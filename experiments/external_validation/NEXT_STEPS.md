# External validation: use treatment-outcome cohorts, not TCGA

**Problem with the first attempt (TCGA):** TCGA is treatment-naïve primary tumors.
It can only measure *prognosis* (does gene X associate with survival in general).
Geryon's hypotheses are *predictive* (does X modify response to a treatment). TCGA
structurally cannot test that — wrong resource. The TCGA biomarker→OS forest plot
(`out/forest.png`) was a plumbing demo, not validation of the enterprise. Drop it.

**Right resource:** external cohorts with treatment + genomics + outcome together.
These are public on the cBioPortal REST API (same client we already wrote,
`cbioportal.py`). Each is IO-treated patients with mutation calls AND
survival-from-treatment — same shape as our `survival_from_treatment` hypotheses,
so re-running a hypothesis here measures HR for *survival on immunotherapy* = the
actual claim.

Candidate IO cohorts (verified on cbioportal API):

| studyId | n | cohort |
|---|---|---|
| mel_dfci_2019 | 144 | melanoma on IO (DFCI) |
| mel_iatlas_riaz_nivolumab_2017 | 107 | melanoma, nivolumab |
| mel_iatlas_gide_2019 | 91 | melanoma, IO trial |
| mel_ucla_2016 | 38 | melanoma on IO |
| blca_dfarber_mskcc_2014 | 50 | bladder on IO |
| ccrcc_dfci_2019 | 35 | ccRCC on IO |

(also: `tmb_mskcc_2018`, `nsclc_pd1_msk_2018` — but MSK, so NOT institutionally
external; same institution as our training data.)

**Honest limits (put on poster):**

1. Most are single-arm (everyone got IO) → measures biomarker→outcome *among
   IO-treated*, not biomarker *modifies* IO benefit vs no-IO (would need a comparator
   arm). BUT our own MSK `survival_from_treatment` hypotheses have the same
   limitation, so it's apples-to-apples — a faithful external mirror of what we
   already do.
2. Granular co-mutation hypotheses are underpowered at n=40–140. Coarse single-gene
   IO-response contrasts (esp. melanoma) are what'll show signal externally.

**Corpus split:**

- Immunotherapy-response subset (melanoma / bladder / RCC IO) → validatable TODAY
  against the cohorts above.
- Granular / non-IO treatment hypotheses → need AACR GENIE BPC (multi-institution,
  curated treatment+outcome; Synapse registered-access, not a quick API pull).

**Next action:** repoint this experiment at the IO cohorts. Same machinery; x-axis
becomes "survival on immunotherapy"; tests the real hypotheses. Retire the TCGA
forest plot.
