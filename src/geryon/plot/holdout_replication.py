"""Do the critic's judgements predict replication better than the explore p-value?

Among explore-significant hypotheses (p<0.05) whose rerun reported a p-value, a
hypothesis replicates if the validation p<0.05 with the same sign. Rows missing any
predictor are dropped so every AUC is over the same hypotheses.

Left: AUC for replication of each predictor, with 95% bootstrap intervals. The naive
one is explore |z|. "Critic forecast" is the replication probability implied by the
critic's predicted held-out effect, at the validation SE (2× the explore SE, since
validation is a quarter the size). "|z| + ratings" is a ridge logistic regression on
|z| and the three ratings, scored leave-one-out; it is left out until at least two
hypotheses replicated and two didn't. An AUC below 0.5 means a higher
rating goes with *less* replication.
Right: the number of replications expected if the explore estimates were true, if
the critic's forecasts were true, and observed.
"""

import math
from statistics import NormalDist

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from geryon.plot._critiques import BAD, GOOD, NEUTRAL
from geryon.plot._holdout import METHOD_COLORS, parse_args
from geryon.plot.holdout_table import SIZE_RATIO, explore_se

_N = NormalDist()
_ALPHA = 0.05
_BOOT = 4000
_RIDGE = 1.0


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """Mann-Whitney AUC, ties counted as half."""
    pos, neg = score[label], score[~label]
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0) + 0.5 * (diff == 0)).mean())


def ridge_logistic(x: np.ndarray, y: np.ndarray, lam: float = _RIDGE) -> np.ndarray:
    """Coefficients (intercept first) of an L2-penalized logistic fit by Newton."""
    X = np.column_stack([np.ones(len(x)), x])
    beta = np.zeros(X.shape[1])
    penalty = lam * np.eye(X.shape[1])
    penalty[0, 0] = 0
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ beta))
        grad = X.T @ (p - y) + penalty @ beta
        hess = X.T @ (X * (p * (1 - p))[:, None]) + penalty
        step = np.linalg.solve(hess, grad)
        beta -= step
        if np.abs(step).max() < 1e-8:
            break
    return beta


def loo_logistic(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Holding out the only member of a class leaves a one-class fit whose
    # unpenalized intercept diverges.
    if min(y.sum(), (~y).sum()) < 2:
        raise ValueError("LOO logistic needs at least two of each class")
    # A constant column (every critic gave novelty 2) can't be standardized.
    sd = x.std(axis=0)
    x = (x[:, sd > 0] - x[:, sd > 0].mean(axis=0)) / sd[sd > 0]
    out = np.empty(len(y))
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        beta = ridge_logistic(x[keep], y[keep].astype(float))
        out[i] = beta[0] + x[i] @ beta[1:]
    return out


def _replication_prob(effect: float, se_val: float) -> float:
    """P(validation p<α, same sign) if `effect` (analysis scale) were the truth."""
    return 1 - _N.cdf(_N.inv_cdf(1 - _ALPHA / 2) - abs(effect) / se_val)


def build_frame(table: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
    critic = forecasts[forecasts["method"] == "critic"].set_index("hypothesis_id")
    rows = []
    for r in table.to_dict("records"):
        if pd.isna(r["explore_p"]) or r["explore_p"] >= _ALPHA or pd.isna(r["val_p"]):
            continue
        if pd.isna(r["explore_effect"]) or pd.isna(r["val_effect"]):
            continue
        f = math.log if r["ratio"] else float
        se = explore_se(
            r["explore_effect"],
            None if pd.isna(r["explore_lower"]) else r["explore_lower"],
            None if pd.isna(r["explore_upper"]) else r["explore_upper"],
            r["explore_p"],
            r["ratio"],
        )
        if se is None or r["hypothesis_id"] not in critic.index:
            continue
        se_val = se * math.sqrt(SIZE_RATIO)
        ex, val = f(r["explore_effect"]), f(r["val_effect"])
        crit = f(critic.loc[r["hypothesis_id"], "predicted"])
        rows.append(
            {
                "replicated": bool(r["val_p"] < _ALPHA and ex * val > 0),
                "z": abs(ex) / se,
                "p_explore_true": _replication_prob(ex, se_val),
                # A critic forecast on the other side of no effect can't replicate.
                "p_critic_true": _replication_prob(crit, se_val)
                if crit * ex > 0
                else 0.0,
                "trustworthiness": r["trustworthiness"],
                "confound_risk": r["confound_risk"],
                "novelty": r["novelty"],
            }
        )
    return pd.DataFrame(rows).dropna()


def main() -> None:
    args = parse_args("Plot how well each signal predicts replication")
    df = build_frame(pd.read_csv(args.table), pd.read_csv(args.forecasts))
    y = df["replicated"].to_numpy()
    if len(df) == 0 or y.all() or not y.any():
        # An AUC needs a replicated and a non-replicated hypothesis; early in a batch
        # there may be neither.
        fig, ax = plt.subplots(figsize=(10, 3.8))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            f"Not enough hypotheses yet: {len(df)} explore-significant with a "
            f"validation result, {int(y.sum())} replicated.\nNeeds at least one "
            f"that replicated and one that didn't.",
            ha="center",
            va="center",
        )
        plt.savefig(args.output, bbox_inches="tight", transparent=True)
        return
    ratings = ["trustworthiness", "confound_risk", "novelty"]
    predictors = {
        "Explore |z|  (naive)": df["z"].to_numpy(),
        "Critic forecast": df["p_critic_true"].to_numpy(),
        "Trustworthiness": df["trustworthiness"].to_numpy(),
        "Confound risk": df["confound_risk"].to_numpy(),
        "Novelty": df["novelty"].to_numpy(),
    }
    colors = [METHOD_COLORS["explore"], GOOD, GOOD, GOOD, GOOD]
    if min(y.sum(), (~y).sum()) >= 2:
        predictors["|z| + ratings  (LOO)"] = loo_logistic(
            df[["z", *ratings]].to_numpy(), y
        )
        colors.append(NEUTRAL)

    rng = np.random.default_rng(42)
    fig, (ax_auc, ax_n) = plt.subplots(
        1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [2, 1]}
    )
    names = list(predictors)
    ys = np.arange(len(names))[::-1]
    for yi, name, color in zip(ys, names, colors, strict=True):
        s = predictors[name]
        boots = []
        for _ in range(_BOOT):
            idx = rng.integers(0, len(y), size=len(y))
            if y[idx].all() or not y[idx].any():
                continue
            boots.append(auc(s[idx], y[idx]))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        a = auc(s, y)
        ax_auc.plot([lo, hi], [yi, yi], color=color, linewidth=2)
        ax_auc.scatter(a, yi, color=color, s=30, zorder=3)
        ax_auc.annotate(
            f"{a:.2f}",
            (a, yi),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    ax_auc.axvline(0.5, color="black", linestyle="--", linewidth=1)
    ax_auc.set_yticks(ys, names, fontsize=9)
    ax_auc.set_xlim(0, 1)
    ax_auc.set_xlabel("AUC for replication")
    ax_auc.grid(True, alpha=0.3, axis="x")
    ax_auc.set_title(
        f"N={len(y)} explore-significant, {int(y.sum())} replicated", fontsize=10
    )

    counts = [
        ("If explore\nwere true", df["p_explore_true"].sum(), METHOD_COLORS["explore"]),
        ("If critic\nwere true", df["p_critic_true"].sum(), GOOD),
        ("Observed", float(y.sum()), BAD),
    ]
    for x, (_, n, color) in enumerate(counts):
        ax_n.bar(x, n, color=color, alpha=0.7, width=0.6)
        ax_n.text(x, n, f"{n:.1f}", ha="center", va="bottom", fontsize=9)
    ax_n.set_xticks(range(len(counts)), [c[0] for c in counts], fontsize=8)
    ax_n.set_ylabel("Replications")
    ax_n.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(args.output, bbox_inches="tight", transparent=True)


if __name__ == "__main__":
    main()
