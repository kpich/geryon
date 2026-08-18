---
data_version: medonc-pfs-2026-08
---

This cohort has **two** independent progression call sets. Every hypothesis in this line
of investigation is about what the medonc-derived PFS can measure that the
radiology-only PFS cannot.

- `timeline_progression` — radiology NLP. `SOURCE = 'Radiology Reports (NLP)'`,
  `SOURCE_SPECIFIC` in {CT, MR, PET}, `PROGRESSION` in {Yes, No, Indeterminate}, with a
  model score in `PROGRESSION_PROBABILITY`. One row per imaging read.
- `timeline_medonc_progression` — the same shape, derived from medical-oncology notes
  rather than radiology reports. Describe it with `describe_table` before using it; do
  not assume its columns match the radiology table exactly.

# What counts as a good hypothesis here

The interesting cases are where progression is **visible in the medonc record but not in
the imaging record**, or visible earlier. Directions worth pursuing:

- Disease that is not imaging-evaluable, or evaluated only sporadically — progression is
  called clinically long before it is called on a scan.
- Patients with sparse or irregular imaging, where a radiology-only PFS is
  interval-censored to the point of being uninformative.
- Event count and statistical power: contrasts that are simply underpowered under the
  radiology-only definition and adequately powered under the medonc one.
- Treatment-line-anchored PFS, where the medonc note is what establishes the line and
  the progression that ends it.
- Disagreements between the two sources treated as the signal itself, rather than as
  noise to be reconciled.

# Hard requirement

Every script you `submit` must compute **both** PFS definitions on the same cohort and
`report(...)` the contrast, not just the medonc result. At minimum, put in `extra`:
events and median follow-up under each definition, and the effect estimate under each.
A hypothesis that only reports the medonc number cannot show that the medonc PFS added
anything, which is the entire question this chain exists to answer.

# Traps specific to this data

- All timeline `START_DATE` values are day-offsets from a per-patient anchor, while
  `OS_MONTHS` is months from diagnosis. Mixing the two silently introduces immortal-time
  bias. Anchor your PFS clock explicitly and say what you anchored it to.
- `PROGRESSION_PROBABILITY` is a model score, not a probability you should threshold at
  0.5 without checking. Whatever threshold you pick, apply the *same* rule to both
  sources or state why you didn't — an apparent medonc advantage that comes from a
  looser threshold is an artifact.
- A patient present in one progression table may be absent from the other. Restrict to
  the intersection when comparing definitions, and report how many patients that drops.
- `STOP_DATE` is essentially all-null in the radiology table; check before relying on it
  in the medonc one.
