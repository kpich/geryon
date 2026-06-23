# Truncation audit (quantify-before-fixing)

Measures how much the stored treatment-hypothesis HRs would move if
`survival_from_treatment` / `progression_from_treatment` applied the same
left-truncation the OS handler already does. **Diagnostic only — changes no engine
code.** Decide from the numbers whether the handler fix is worth doing.

For each stored treatment hypothesis it re-fits the same cohort two ways:
- **untrunc** — no `entry_time` (reproduces the stored fit)
- **trunc** — `entry_time = max(0, -tx_start/30.44)` (immortal time before sequencing)

## Run

```bash
cd experiments/truncation_audit
PYTHONPATH=. ../../.venv/bin/python audit.py            # -> out/truncation_audit.csv
# options: --split all (default 'train', matches original scoring)
#          --parquet-dir /path (default: latest ~/data/geryon_data/*)
#          --data-dir /path/to/geryon_data
```

## What to read in the summary it prints

- **reproduction check** `|hr_untrunc - stored_hr|` — should be ~0. If it's large, the
  re-execution doesn't match how the hypotheses were originally scored (split mismatch,
  replayed-view drift) and the rest is suspect — tell me and we debug before trusting it.
- **`|log2(HR_trunc/HR_untrunc)|`** — the bias magnitude. median ~0 = immaterial;
  0.585 = 1.5×, 1.0 = 2× typical move.
- **HR crossed 1** — hypotheses whose effect *direction* flips. These are the scary ones.
- **significance flipped at p<0.05** — how many change called/not-called.

Per-hypothesis detail (incl. `frac_preseq_a/b` = fraction of each arm treated before
sequencing) is in `out/truncation_audit.csv`.

## Caveat

I have not run this. It duplicates the two handlers' SQL locally (engine untouched),
so eyeball the reproduction-check line on the first run before trusting the deltas.
If `submit_hypothesis` originally scored on a different split than `train`, pass the
matching `--split`.
