# Geryon Project

LLM agent for hypothesis generation on cancer clinicogenomic data (MSK-IMPACT). Each
hypothesis is a Python script the model writes and runs in a Docker sandbox. A narrator
summarizes the result, and an agentic critic tries to falsify it. The README covers
usage; this file covers what matters for changing the code.

## Structure

```
src/geryon/
├── codeflow/   # the loop: agent.py (generator), critic.py, narrate.py, prompts.py,
│               #   chains.py, context.py (prior-hypothesis summaries), store.py (JSONL),
│               #   runner.py (CLI entry point, data-dir resolution)
├── sandbox/    # runner.py (host-side docker run), runtime.py (in-container
│               #   `geryon_runtime`: db() + report()), result.py, Dockerfile
├── tools/      # list_tables / describe_table / query_data for the agents
├── db/         # DuckDB over the parquet dir
├── etl/        # parquet conversion, profiling, patient split, data_version.py
├── llm/        # providers (Bedrock, OpenAI, Anthropic), prompt caching, tracing
├── workflow/   # SessionConfig
├── cli/        # hypothesis viewer (flask), list_sessions
└── plot/       # cost-over-time plot
nextflow/       # etl.nf (TSV -> parquet -> split), plot.nf
scripts/        # etl.sh, plot.sh, check_split_stable.py, verify_sandbox.py, view_data.py
```

## Key Patterns

**Code is the deliverable.** A `CodeHypothesis` (`codeflow/models.py`) is the script plus
the `IterationResult` it reported through `geryon_runtime.report()`. Every analytic field
in the result is nullable, because "no clean effect size" is a legitimate outcome.
`sandbox/runtime.py` is copied into the image on its own and must not import from
`geryon`.

**Generator and critic share tools.** `codeflow/_shared.py` holds the LLM factory, the
read-only exploration tools and `run_python`. The generator adds `submit` and
`get_script`; the critic adds `submit_critique`. Both are LangGraph ReAct agents with
tool calling.

**Failures are loud.** An error in generation, narration or the critic ends the session
with a nonzero exit code. Hypotheses already submitted are on disk, and so are the
critiques that finished. Don't add `except Exception` handlers that warn and carry on,
and don't write fallback values that look like real output (the critic used to fill in
neutral 2/2/2 scores). What *does* go back to the model is its own mistakes: bad SQL, a
crashing script, a timeout, bad tool arguments. The tools return those as strings, and
`ToolNode` uses LangGraph's default handler, which re-raises everything else. An
iteration where the model submits nothing is a legitimate outcome and the session
continues.

**Holdout enforced at the data layer.** The inner loop must only ever see the
*exploration* set, never validation. Hypotheses are free-form Python in a sandbox, so
there is no per-query chokepoint where a filter could be applied. Instead the parquet is
split physically: `etl/create_patient_split.py` labels patients `explore` (80%) /
`validation` (20%), and `etl/split_by_patient.py` writes `<version>/explore/` and
`<version>/validation/`, each filtered to its split and carrying regenerated profiles and
a `SPLIT` marker. The session reads `explore/` only. Validation is absent from both the
DuckDB views and the read-only sandbox mount. `runner.resolve_explore_dir` hard-fails on
an unsplit dir, and `CodeWorkflow.__init__` refuses any dir not marked `explore`.

Traps:
- CNA is keyed by sample even though its column is *named* `PATIENT_ID` (it holds sample
  barcodes). See `SAMPLE_KEY_COLUMNS`.
- A new sample-keyed source file must be added to `SAMPLE_KEY_COLUMNS`. Otherwise it is
  treated as metadata and copied **unfiltered into both splits**, which leaks the
  holdout without any error.
- The split is seeded but depends on row order in `data_clinical_patient.txt`. Reorder
  that file and patients move.

Code says `explore`; prose says "exploration".

**Named data versions.** An ETL run publishes to `~/data/geryon_data/<version>/`. A human
picks the version name (`medonc-pfs-2026-08`); it isn't the run date. Build a named
version from its own copy of the source tree so the canonical `msk_solid_heme/` stays
clean:

```bash
make etl ARGS="--version medonc-pfs-2026-08 --data_root ~/data/msk-impact/msk_solid_heme_medonc"
```

`--version` defaults to today's date, so a plain `make etl` behaves as it always has.
`VERSION.json` at the version root (copied into `explore/` and `validation/`) records the
name, source tree and seed. `resolve_data_version` reads it, and every hypothesis stores
it. Auto-detection of the "latest" dir considers **only** date-named dirs; otherwise a
named version like `medonc-…` would sort after every date and silently become "latest".
After building a new version, run `scripts/check_split_stable.py <a> <b>` to confirm
that no patient crossed the explore/validation boundary. If one did, results are not
comparable across the two versions. `publishDir` doesn't overwrite, so rebuilding under
an existing name keeps the old files. Use a new name.

**Chains.** A chain is a separate line of investigation. `--chain <name>` (or
`make run CHAIN=<name>`) shows the generator **only that chain's** prior hypotheses, so
a focused investigation builds on its own work without being flooded by the main line.
`get_script` deliberately still resolves ids from any chain: the boundary governs what
is pushed into the prompt, not what can be pulled. Session dirs are not partitioned by
chain. The chain is recorded in the JSONL header and filtered on read.

A chain is defined by `chains/<name>.md` (the dir can be moved with `--chains-dir`). It
has optional frontmatter pinning `data_version`, then free prose appended to the
generator, critic and narrator **system** prompts, which sit before the cache
breakpoint. The critic needs the prose too: without it, it scores novelty against the
general literature and dismisses focused hypotheses as textbook. Prompting is the only
steering lever. Sampling params are deliberately omitted because Claude 4.x returns a
400 when they're set. `main` has no file, so the open-ended chain keeps its original
behavior. Sessions written before chains existed read as `main`. No chain files live in
this repo today. The one real chain (`medonc-pfs-os`) lives with the PFS paper in
`~/dev/pfs/cdm-pfs-modeling-project/misc_analysis/pfs_os_hyp_gen/`, which drives this repo through
`--chains-dir` / `--output-dir`.

## Known loose ends

- `explore/` has a `.profile.json` for only half the tables. The Nextflow
  `splitByPatient` step stages parquet only, so the other tables fall back to live
  sampling in `describe_table`. It works fine; it's just untidy.

## Testing

- **Unit tests** live in `src/` next to the code they test, named `*_test.py` (e.g., `agent_test.py`). Run with `make test` (`pytest src/`). CI (`.github/workflows/ci.yml`) runs ruff, mypy, and these on every push to `main` and every PR.
- **Integration tests** (multi-module, not unit) are a separate category that will live outside `src/`. There are none yet — add the location and runner when the first one is written, don't scaffold it ahead of time.

## Code Style

**No vacuous comments** - Don't add comments that just restate what the code does. Only comment if adding non-obvious context (why, not what)

## Safety — Never Invoke LLM Workflows

NEVER run commands that trigger external LLM API calls (Bedrock, OpenAI, Anthropic, Ollama, etc.). This includes:
- `run.sh`, `scripts/run.sh`, or any wrapper that launches a session
- `uv run python -m geryon` or any direct invocation of the workflow
- Any script or command that calls the LLM providers in `geryon/llm/`

These calls are billable and should only be triggered by the human operator. Stick to code edits, tests, and read-only exploration.

## Safety — Never Perform Git Operations

NEVER run git commands that change repository state, and never ask to. The human operator handles all git themselves. This includes (non-exhaustive):
- `git commit`, `git add`, `git push`, `git pull`, `git merge`, `git rebase`, `git reset`, `git stash`, `git cherry-pick`
- Creating, deleting, renaming, switching, or checking out branches (`git branch`, `git checkout`, `git switch`)
- Creating tags, editing git config, or anything else that mutates the repo or its history

Do not do any of this of your own volition, and do not ask whether you should — just leave git alone entirely. Read-only inspection (`git status`, `git log`, `git diff`, `git show`) is fine. If a task seems to need a commit, branch, or other git action, stop and let the human do it.

## Safety — Don't Delete Miscellaneous Files

Removing code as a legitimate part of the task at hand is fine — delete dead modules,
drop obsolete source files, and clean up in an approved refactor without a confirmation
round-trip. This rule is about *miscellaneous* files, not task-scoped code changes.

NEVER delete or remove files that are outside the scope of what you're working on without
first explicitly asking the human and getting a clear yes. In particular, protect anything
you didn't create or touch as part of the task, and anything that looks valuable —
data, results, outputs, configs, someone else's work. This covers `rm`, `rm -rf`, `mv`
that overwrites or discards such a file, `git clean`, truncating/emptying a file, or any
command whose effect is to remove or destroy a file's contents.

A user saying they don't need a file, don't want to keep it, or that it's temporary is NOT
permission to delete it — that is context, not an instruction to act. When it's an
out-of-scope or valuable file, ask first ("want me to delete X?") and wait for an explicit
yes. When in doubt, leave the file in place and let the human delete it themselves.
