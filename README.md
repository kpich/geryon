# Geryon

An LLM agent that generates hypotheses from a tabular dataset and then tries to knock
them down. Early and changing fast.

Each hypothesis is a Python script the model writes and runs in a Docker sandbox. A
critic agent then runs its own analyses to try to break it.

## Setup

Needs [uv](https://docs.astral.sh/uv/), Docker and Nextflow.

```bash
make dev             # install deps and pre-commit hooks
make sandbox-build   # build the sandbox image
make etl             # raw TSVs -> parquet, with a held-out validation split
```

## Usage

```bash
make run ITERS=3     # run a session (billable LLM calls)
make viewer          # browse results at http://localhost:8765
make check           # lint, types, tests
```

More options: `uv run python -m geryon.codeflow.runner --help`.

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
