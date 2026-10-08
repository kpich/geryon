"""Join the validation rerun with the explore results, critiques and baselines.

Writes two CSVs for the holdout plots:

- ``--table``: one row per rerun hypothesis, with its explore result, validation
  result, critic scores, and BH q-values for each split (across every rerun
  hypothesis that reported a p-value on that split).
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
  - ``expectation``: the blind expectation and its 80% interval, which was asked for
    the explore estimate, so it is narrower than one for the smaller validation set.

The table also carries each hypothesis's blind expectation, its ``prediction_z``, and
its ``replication_z``.
"""

import argparse
import json
import math
from pathlib import Path
import re
from statistics import NormalDist

import pandas as pd
from statsmodels.stats.multitest import multipletests  # type: ignore[import-untyped]

from geryon.codeflow.models import Expectation
from geryon.plot._critiques import load_hypotheses
from geryon.sandbox.result import IterationResult

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


def expectation_sd(lower: float, upper: float, ratio: bool) -> float:
    """SD of the blind expectation on the analysis scale, its 80% interval read as
    normal."""
    f = math.log if ratio else float
    return (f(upper) - f(lower)) / (2 * _Z(0.5 + LEVEL / 2))


def prediction_z(
    result: IterationResult | None, expectation: Expectation | None
) -> float | None:
    """|explore estimate − blind expectation| in SDs: a z-score, not a surprisal.

    The SD combines the estimate's SE with the expectation's own spread, both read as
    normal on the analysis scale (log for ratios), so neither a noisy estimate nor a
    vague expectation scores high. Absolute, since which group is the reference is
    arbitrary. None without an expectation or an SE.
    """
    if result is None or expectation is None or result.effect_size is None:
        return None
    ratio = is_ratio(result.effect_size_type)
    values = [result.effect_size, expectation.effect, expectation.lower]
    if ratio and min(values) <= 0:
        raise ValueError(f"ratio effect or expectation <= 0: {values}")
    se = explore_se(
        result.effect_size, result.ci_lower, result.ci_upper, result.p_value, ratio
    )
    if se is None:
        return None
    f = math.log if ratio else float
    spread = expectation_sd(expectation.lower, expectation.upper, ratio)
    return abs(f(result.effect_size) - f(expectation.effect)) / math.hypot(se, spread)


def replication_z(
    explore: IterationResult | None, val: dict, ratio: bool
) -> float | None:
    """|explore − validation| over the SE of their difference.

    Scores a null and an effect alike: a hypothesis replicates when the rerun lands
    near its explore estimate, wherever that is. Significance would call a replicated
    null a failure. None when either split lacks an effect or an SE, or a ratio rerun
    came out degenerate (an effect or bound <= 0, or not finite, e.g. no events).
    """
    if explore is None or explore.effect_size is None:
        return None
    v_effect, v_lo, v_hi, v_p = (
        val.get("effect_size"),
        val.get("ci_lower"),
        val.get("ci_upper"),
        val.get("p_value"),
    )
    if v_effect is None:
        return None
    v_values = [x for x in (v_effect, v_lo, v_hi) if x is not None]
    if not all(math.isfinite(x) for x in v_values):
        return None
    if ratio and min(v_values) <= 0:
        return None
    se_ex = explore_se(
        explore.effect_size, explore.ci_lower, explore.ci_upper, explore.p_value, ratio
    )
    se_val = explore_se(v_effect, v_lo, v_hi, v_p, ratio)
    if se_ex is None or se_val is None:
        return None
    f = math.log if ratio else float
    return abs(f(explore.effect_size) - f(v_effect)) / math.hypot(se_ex, se_val)


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


def bh_q(p: pd.Series) -> pd.Series:
    """BH q-values over the non-null p-values; null where p is null."""
    q = pd.Series(None, index=p.index, dtype=float)
    has_p = p.notna()
    if has_p.any():
        q[has_p] = multipletests(p[has_p], method="fdr_bh")[1]
    return q


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
                "created_at": h.created_at.isoformat(),
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
                "expected_effect": h.expectation.effect if h.expectation else None,
                "expected_lower": h.expectation.lower if h.expectation else None,
                "expected_upper": h.expectation.upper if h.expectation else None,
                "prediction_z": prediction_z(ex, h.expectation),
                "replication_z": replication_z(ex, val, ratio),
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
        if h.expectation is not None:
            e = h.expectation
            forecasts.append((hid, "expectation", e.effect, e.lower, e.upper))
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
    for split in ["explore", "val"]:
        table[f"{split}_q"] = bh_q(table[f"{split}_p"])
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
