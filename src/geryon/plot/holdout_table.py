"""Join the validation rerun with the explore results, critiques and baselines.

Writes two CSVs for the holdout plots:

- ``--table``: one row per rerun hypothesis, with its explore result, validation
  result, critic scores, and the validation q-value (BH across every rerun that
  reported a p-value).
- ``--forecasts``: long format, one row per (hypothesis, method), with a forecast of
  the validation effect size and an 80% interval. Rows exist only where the explore
  run reported an effect size. The methods:

  - ``critic``: the critic's ``predicted_holdout_*``.
  - ``explore``: the explore estimate, with the 80% interval that predicts a fresh
    estimate on a sample a quarter the size. Its variance is se_explore² +
    se_validation², and se_validation² ≈ 4·se_explore² under the 80/20 split, so the
    interval is the explore SE widened by √5. The SE comes from the explore CI, read as
    95%, or from the p-value when there's no CI. Ratio effects use the log scale.
  - ``no_effect``: no effect (1 for a ratio, 0 otherwise), a point with no interval.
"""

import argparse
import json
import math
from pathlib import Path
import re
from statistics import NormalDist

import pandas as pd
from statsmodels.stats.multitest import multipletests  # type: ignore[import-untyped]

from geryon.plot._critiques import load_hypotheses

LEVEL = 0.8
# Explore patients per validation patient (create_patient_split holds out 20%).
SIZE_RATIO = 4.0
_Z = NormalDist().inv_cdf
_RATIO_ABBREV = re.compile(r"(?<![A-Za-z])(HR|OR|RR)s?(?![A-Za-z])")
_RATIO_WORD = re.compile(r"hazard|odds|ratio", re.IGNORECASE)


def is_ratio(effect_size_type: str | None) -> bool:
    """Whether the free-text effect type names a ratio, which lives on a log scale."""
    if not effect_size_type:
        return False
    return bool(
        _RATIO_ABBREV.search(effect_size_type) or _RATIO_WORD.search(effect_size_type)
    )


def explore_se(
    effect: float,
    lower: float | None,
    upper: float | None,
    p: float | None,
    ratio: bool,
) -> float | None:
    """SE of the explore estimate on the analysis scale (log for ratios)."""
    f = math.log if ratio else float
    if lower is not None and upper is not None:
        return (f(upper) - f(lower)) / (2 * _Z(0.975))
    if p is not None and 0 < p < 1 and f(effect) != 0:
        return abs(f(effect)) / _Z(1 - p / 2)
    return None


def explore_forecast(
    effect: float,
    lower: float | None,
    upper: float | None,
    p: float | None,
    ratio: bool,
) -> tuple[float, float | None, float | None]:
    se = explore_se(effect, lower, upper, p, ratio)
    if se is None:
        return effect, None, None
    half = _Z(0.5 + LEVEL / 2) * se * math.sqrt(1 + SIZE_RATIO)
    if ratio:
        centre = math.log(effect)
        return effect, math.exp(centre - half), math.exp(centre + half)
    return effect, effect - half, effect + half


def build(data_dir: Path, runs_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    hyps = {h.hypothesis_id: h for h in load_hypotheses(data_dir)}
    runs = [json.loads(line) for line in runs_path.read_text().splitlines()]

    rows, forecasts = [], []
    for run in runs:
        h = hyps[run["hypothesis_id"]]
        ex, val, c = h.result, run["result"] or {}, h.critique
        ratio = is_ratio(ex.effect_size_type if ex else None)
        rows.append(
            {
                "hypothesis_id": h.hypothesis_id,
                "title": h.title,
                "chain": h.chain,
                "trustworthiness": c.trustworthiness if c else None,
                "confound_risk": c.confound_risk if c else None,
                "novelty": c.novelty if c else None,
                "effect_size_type": ex.effect_size_type if ex else None,
                "ratio": ratio,
                "explore_effect": ex.effect_size if ex else None,
                "explore_lower": ex.ci_lower if ex else None,
                "explore_upper": ex.ci_upper if ex else None,
                "explore_p": ex.p_value if ex else None,
                "val_success": run["success"],
                "val_effect": val.get("effect_size"),
                "val_lower": val.get("ci_lower"),
                "val_upper": val.get("ci_upper"),
                "val_p": val.get("p_value"),
            }
        )
        if ex is None or ex.effect_size is None:
            continue
        if ratio and ex.effect_size <= 0:
            raise ValueError(f"{h.hypothesis_id}: ratio effect {ex.effect_size} <= 0")
        point, lo, hi = explore_forecast(
            ex.effect_size, ex.ci_lower, ex.ci_upper, ex.p_value, ratio
        )
        hid = h.hypothesis_id
        forecasts.append((hid, "explore", point, lo, hi))
        forecasts.append((hid, "no_effect", 1.0 if ratio else 0.0, None, None))
        if c is not None and c.predicted_holdout_effect is not None:
            forecasts.append(
                (
                    hid,
                    "critic",
                    c.predicted_holdout_effect,
                    c.predicted_holdout_lower,
                    c.predicted_holdout_upper,
                )
            )

    table = pd.DataFrame(rows)
    has_p = table["val_p"].notna()
    table["val_q"] = None
    if has_p.any():
        table.loc[has_p, "val_q"] = multipletests(
            table.loc[has_p, "val_p"], method="fdr_bh"
        )[1]
    fc = pd.DataFrame(
        forecasts, columns=["hypothesis_id", "method", "predicted", "lower", "upper"]
    )
    return table, fc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--table", required=True, type=Path)
    parser.add_argument("--forecasts", required=True, type=Path)
    args = parser.parse_args()
    table, fc = build(args.data_dir, args.runs)
    table.to_csv(args.table, index=False)
    fc.to_csv(args.forecasts, index=False)


if __name__ == "__main__":
    main()
