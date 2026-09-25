# Geryon

An LLM agent that generates and stress-tests hypotheses on cancer clinicogenomic data
(MSK-IMPACT via cBioPortal). Early and changing fast; interfaces below will break.

Each hypothesis is a Python script the model writes and runs in a locked-down Docker
sandbox, with the data mounted read-only and no network. The script reports an effect
size, CI and p-value. A second agent, the critic, then gets the same tools and tries to
break the finding by running its own analyses.

## How a session works

Each iteration has three roles:

1. **Generator.** Explores the data with read-only tools (`list_tables`,
   `describe_table`, `query_data` for SQL), tries analyses with `run_python`, and calls
   `submit` when it has a result worth keeping. It sees one-line summaries of earlier
   hypotheses and can pull any of them in full with `get_script` to refine it.
2. **Narrator.** Writes a plain-language account of the result and its limitations.
3. **Critic** (optional, `--critic-cycles 1`). Same tools as the generator. It probes for
   confounding and bias, then scores the hypothesis for trustworthiness, confound risk
   and novelty, and says whether it holds up.

Hypotheses are appended to `geryon_data/sessions/<date>/<id>/hypotheses.jsonl`, next to
the session's `config.json` and a `trace.jsonl` of every tool call.

## Setup

Needs [uv](https://docs.astral.sh/uv/), Docker and Nextflow.

```bash
make dev             # uv sync --all-extras, install pre-commit hooks
make sandbox-build   # build the geryon-sandbox Docker image
make etl             # cBioPortal TSVs -> parquet, split into explore/validation
```

The ETL reads from `params.data_root` in `nextflow/etl.nf` and writes to
`~/data/geryon_data/<version>/`.

## Running

```bash
make run                         # 10 iterations on AWS Bedrock, critic on, tee'd to ./out
make run ITERS=1                 # quick smoke run
make run CHAIN=<name> ITERS=5    # a focused line of investigation (see Chains)
```

`make run` makes billable LLM calls. It forwards only `ITERS` and `CHAIN`. For other
options (provider, model, data version, output dir) call the runner directly; see
`uv run python -m geryon.codeflow.runner --help`. The providers are `aws_bedrock` (the
default), `anthropic` and `openai`.

```bash
make viewer          # browse hypotheses at http://localhost:8765
make data            # poke at the parquet tables in harlequin
make plot            # Nextflow plot pipeline -> plots/
```

## Data

**The validation holdout is enforced physically.** The ETL assigns 80% of patients to
`explore/` and 20% to `validation/`, as separate parquet directories. A session mounts
only `explore/`. The runner refuses a directory that isn't marked as the explore split,
so nothing the model runs can reach the validation patients.

**Data versions.** The ETL output directory is named by `--version`, which defaults to
today's date. Give a version a real name when it's built from a variant source tree:

```bash
make etl ARGS="--version medonc-pfs-2026-08 --data_root ~/data/msk-impact/msk_solid_heme_medonc"
uv run python scripts/check_split_stable.py 2026-06-30 medonc-pfs-2026-08
```

The second command confirms that no patient changed sides of the explore/validation
split between versions. If one did, results from the two versions aren't comparable.
When no version is requested, the runner picks the latest *date-named* directory. A
named version is used only when asked for by name.

**Chains.** A chain is a separate line of investigation. Its generator sees only its own
earlier hypotheses. A chain is defined by `chains/<name>.md`:

```markdown
---
data_version: medonc-pfs-2026-08
---
Prose that is appended to the generator, critic and narrator system prompts.
```

The default chain, `main`, has no file and is fully open-ended. Chain files don't have
to live in this repo: `--chains-dir` and `--output-dir` let another project keep its
prompts and sessions under its own version control.

## Development

```bash
make test            # pytest src/ (unit tests sit next to the code as *_test.py)
make mypy
make format
make backup          # commit and push geryon_data/ to its own git remote
make restore         # clone geryon_data/ from that remote
```

CI runs ruff, mypy and the tests on every PR.

## Sample output

A real hypothesis from an earlier run, as `make viewer` renders it (lightly abridged).
The generator found an apparently strong result, and the critic took it apart with
landmark analyses. Catching that kind of failure is the critic's job.

---

> ### SMARCA4 mutation as a predictive biomarker for immunotherapy benefit in NSCLC (TMB-adjusted)
> `iteration 1` · `aws_bedrock/claude-opus-4-8` · ran in 1.5s
>
> **Description.** In NSCLC, SMARCA4 mutations are prognostically adverse without
> immunotherapy (HR=1.36) but neutral under IO (HR=0.95). The SMARCA4×IO interaction
> (HR=0.70, 95% CI 0.59–0.83, p=4.3e-5) suggests SMARCA4-mutant patients derive
> disproportionate relative benefit from IO. Persists after adjusting for log(TMB),
> STK11, KEAP1, TP53, KRAS, stage IV, sex, and age (n=11,619; 913 SMARCA4-mutant).
>
> **Result**
> | | |
> |---|---|
> | effect size | **0.70** — interaction HR (SMARCA4 × IO) |
> | 95% CI | 0.585 – 0.828 |
> | p-value | 4.3e-05 |
> | n | 913 mutant / 10,706 wild-type |
>
> **Narrative.** Without IO, SMARCA4 is adverse (HR=1.36); under IO the disadvantage
> is essentially eliminated (HR=0.95). The interaction is highly significant and
> survives adjustment for TMB (mutant median 11.4 vs 5.3 mut/Mb), co-occurring drivers,
> stage, sex, and age (concordance 0.68).
> *Limitations (model-flagged):* observational/non-randomized; treatment-selection &
> immortal-time bias; PD-L1 not included; OS measured from diagnosis not treatment start; …
>
> **Critic** — trust **1/3** · confound-risk **3/3** · novelty **2/3** · **holds up: no**
>
> > The SMARCA4×IO interaction is almost certainly an artifact of **immortal-time bias**.
> > *Landmark analysis destroys the effect* (3-mo: HR=0.89, p=0.29; 6-mo: HR=0.91, p=0.40;
> > stage-IV + 3-mo landmark: HR=1.19 — reverses). The IO main effect HR>1 in every model,
> > implausible for an effective therapy: the IO variable is capturing "sicker patients who
> > need systemic therapy." 61% of IO patients start treatment *after* the reference time.
> >
> > *Checks run:* landmark analysis at 3/6/12 mo · stage-IV-restricted · IO treatment-timing ·
> > stage & event-rate distributions by group.
> > *Suggested fix:* time-varying IO covariate (or proper landmarking), restrict to stage-IV
> > patients on systemic therapy, propensity-match on indication.
