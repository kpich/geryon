"""Flatten session dirs into dataframes for ad-hoc poking. Not part of the package."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

SESSIONS = Path(__file__).resolve().parent.parent / "geryon_data" / "sessions"


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sessions(root: Path = SESSIONS) -> pd.DataFrame:
    rows = []
    for cfg_path in sorted(root.glob("*/*/config.json")):
        d = cfg_path.parent
        cfg = json.loads(cfg_path.read_text())
        trace = _jsonl(d / "trace.jsonl")
        events = [t["event"] for t in trace]
        hyp = d / "hypotheses.jsonl"
        n_hyp = (
            sum(r["record_type"] == "hypothesis" for r in _jsonl(hyp))
            if hyp.exists()
            else 0
        )
        rows.append(
            {
                "session_id": cfg["session_id"],
                "created_at": pd.Timestamp(cfg["created_at"]),
                "code_version": cfg["code_version"],
                "max_iterations": cfg["max_iterations"],
                "iterations_started": events.count("iteration_start"),
                "ended": ("session_end" in events),
                "n_hyp": n_hyp,
                "last_ts": trace[-1]["ts"],
            }
        )
    return pd.DataFrame(rows).sort_values("created_at")


def hypotheses(root: Path = SESSIONS) -> pd.DataFrame:
    rows = []
    for path in sorted(root.glob("*/*/hypotheses.jsonl")):
        for r in _jsonl(path):
            if r["record_type"] != "hypothesis":
                continue
            h = r["data"]
            res = h.get("result") or {}
            crit = h.get("critique") or {}
            narr = h.get("narrative") or {}
            rows.append(
                {
                    "hypothesis_id": h["hypothesis_id"],
                    "session_id": h["session_id"],
                    "created_at": pd.Timestamp(h["created_at"]),
                    "iteration": h["iteration"],
                    "refines": h.get("refines"),
                    "title": h["title"],
                    "code": h["code"],
                    "code_lines": h["code"].count("\n"),
                    "success": h.get("success"),
                    "duration": h.get("duration_seconds"),
                    "effect_size": res.get("effect_size"),
                    "effect_size_type": res.get("effect_size_type"),
                    "p_value": res.get("p_value"),
                    "n_a": res.get("n_a"),
                    "n_b": res.get("n_b"),
                    "narrator_model": narr.get("model"),
                    "has_critique": bool(crit),
                    "trust": crit.get("trustworthiness"),
                    "confound": crit.get("confound_risk"),
                    "novelty": crit.get("novelty"),
                    "holds_up": crit.get("holds_up"),
                    "n_tests": len(crit.get("tests_run") or []),
                    "headline": crit.get("headline"),
                    "pred": crit.get("predicted_holdout_effect"),
                }
            )
    return pd.DataFrame(rows).sort_values("created_at").reset_index(drop=True)


def detail(root: Path = SESSIONS) -> pd.DataFrame:
    rows = []
    for path in sorted(root.glob("*/*/detail.jsonl")):
        sid = path.parent.name
        for r in _jsonl(path):
            r["session_id"] = sid
            rows.append(r)
    return pd.DataFrame(rows)


def usage(root: Path = SESSIONS) -> pd.DataFrame:
    rows = []
    for path in sorted(root.glob("*/*/trace.jsonl")):
        for t in _jsonl(path):
            if t["event"] == "generation_usage":
                t["session_id"] = path.parent.name
                rows.append(t)
    return pd.DataFrame(rows)


def calls(root: Path = SESSIONS) -> pd.DataFrame:
    """One row per model reply, with its conversation and whether the fallback wrote it.

    detail.jsonl has no hypothesis id. Each phase's conversation starts with a system
    message, and a session's conversations alternate generation/critic in iteration
    order, so the k-th pair belongs to the session's k-th hypothesis.
    """
    d = detail(root)
    d["conv"] = (d["type"] == "system").cumsum()
    hyps = hypotheses(root)
    conv_to_hyp = {}
    for sid, convs in d[d["type"] == "system"].groupby("session_id")["conv"]:
        ids = hyps[hyps["session_id"] == sid].sort_values("created_at")["hypothesis_id"]
        convs = list(convs)
        assert len(convs) == 2 * len(ids), (sid, len(convs), len(ids))
        for k, hid in enumerate(ids):
            conv_to_hyp[convs[2 * k]] = hid
            conv_to_hyp[convs[2 * k + 1]] = hid
    ai = d[(d["type"] == "ai") & d["model"].notna()].copy()
    ai["hypothesis_id"] = ai["conv"].map(conv_to_hyp)
    ai["fallback"] = ai["model"].str.contains("opus-4-8")
    ai["pos"] = ai.groupby("conv").cumcount()
    ai["n_calls"] = ai.groupby("conv")["pos"].transform("max") + 1
    return ai.merge(hyps[["hypothesis_id", "created_at"]], on="hypothesis_id")


HOLDOUT_TABLE = (
    SESSIONS.parent.parent / "plots" / "critic_holdout" / "data" / "holdout_table.csv"
)


def generator_tool_counts(root: Path = SESSIONS) -> pd.DataFrame:
    """Tool calls the generator made per hypothesis, one column per tool."""
    d = detail(root)
    d["conv"] = (d["type"] == "system").cumsum()
    gen_conv = (
        calls(root)
        .query("phase == 'generation'")
        .groupby("hypothesis_id")["conv"]
        .first()
    )
    tools = d[(d["type"] == "tool") & (d["phase"] == "generation")]
    counts = tools.groupby("conv")["name"].value_counts().unstack(fill_value=0)
    return counts.reindex(gen_conv.values).set_axis(gen_conv.index).reset_index()


def holdout(root: Path = SESSIONS) -> pd.DataFrame:
    """Ratio-effect hypotheses with explore and validation estimates, on the z scale.

    SE comes from the reported 95% CI on the log scale, so rows without a CI are
    dropped.
    """
    t = pd.read_csv(HOLDOUT_TABLE)
    t = t[t["ratio"] & t["val_effect"].notna()].copy()
    for s in ("explore", "val"):
        t[f"log_{s}"] = np.log(t[f"{s}_effect"])
        t[f"se_{s}"] = (np.log(t[f"{s}_upper"]) - np.log(t[f"{s}_lower"])) / (2 * 1.96)
        t[f"z_{s}"] = t[f"log_{s}"] / t[f"se_{s}"]
    t = t.dropna(subset=["z_explore", "z_val"])
    h = hypotheses(root).drop(
        columns=["title", "novelty", "trustworthiness"], errors="ignore"
    )
    return t.merge(h, on="hypothesis_id").merge(
        generator_tool_counts(root), on="hypothesis_id", how="left"
    )
