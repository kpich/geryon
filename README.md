# Geryon

v much in dev/beta -- check back later hopefully.

LLM tool for exploring cancer clinicogenomics data and generating hypotheses.
Each hypothesis is arbitrary Python the model writes and runs in a locked-down
Docker sandbox (data mounted read-only, no network): explore the data, write an
analysis, get an effect size + p-value, iterate.

In the sandbox the model has `list_tables` / `describe_table` / `query_data`
(read-only SQL), `run_python` (execute a script), and `submit` (run + store a
hypothesis). See [Sample output](#sample-output) for a real example of what it
currently produces.

## Setup

```bash
make dev            # uv sync --all-extras + pre-commit
make sandbox-build  # build the geryon-sandbox Docker image (once; needs Docker)
make etl            # Nextflow: cbioportal TSVs -> parquet (see nextflow/etl.nf for paths)
```

Parquet data lives in `~/data/geryon_data/`; the latest subdir is auto-detected.

## Run

```bash
make run            # aws_bedrock + default model; stdout tee'd to ./out
```

Override settings inline, e.g. `make run ITERS=1`. Providers:
`aws_bedrock` (default), `anthropic`, `openai` — see the `run` target in the Makefile.
Sessions are written to `geryon_data/code_sessions/`.

```bash
make viewer         # browse hypotheses (http://localhost:8765)
make data           # examine raw ETL/derived data (harlequin)
make plot           # Nextflow plot pipeline -> plots/
```

## Development

```bash
make test mypy format
make backup         # push geryon_data to its git remote
make restore        # clone geryon_data from remote
```

## Sample output

A real hypothesis from a run, as rendered in `make viewer` (lightly abridged).
**Subject to heavy change — this is likely already outdated.** Note especially
the agentic critic, which writes and runs its *own* code to attack the finding;
here it ran landmark analyses and concluded the effect was an artifact.

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
