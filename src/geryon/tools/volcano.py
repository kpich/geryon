"""Volcano scan tool: scan a categorical column, run one-vs-rest comparisons."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import pandas as pd
from pydantic import TypeAdapter
from scipy.stats import false_discovery_control  # type: ignore[import-untyped]

from geryon.db import Database
from geryon.engine.executor import HypothesisExecutor
from geryon.engine.registry import (
    TABLE_PATIENT_ID_JOINS,
    get_method_implementation,
    get_outcome_handler,
)
from geryon.lang.methods import ComparisonMethod
from geryon.lang.spec import Outcome

P_MIN = sys.float_info.min


def _parse_outcome(outcome_spec: str) -> Outcome:
    adapter: TypeAdapter[Outcome] = TypeAdapter(Outcome)
    return adapter.validate_python(json.loads(outcome_spec))


def _cache_key(
    group_table: str,
    group_column: str,
    outcome_spec: str,
    method: str,
    min_group_size: int,
    max_groups: int,
) -> str:
    payload = json.dumps(
        {
            "group_table": group_table,
            "group_column": group_column,
            "outcome_spec": outcome_spec,
            "method": method,
            "min_group_size": min_group_size,
            "max_groups": max_groups,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _compare_one(
    grp: str,
    group_ids: frozenset,
    all_ids: frozenset,
    outcome: Outcome,
    method_enum: ComparisonMethod,
    executor: HypothesisExecutor,
    min_group_size: int,
    all_outcome_data: pd.DataFrame | None = None,
) -> dict | None:
    if all_outcome_data is not None:
        cohort_a_df = all_outcome_data[all_outcome_data["PATIENT_ID"].isin(group_ids)]
        cohort_b_df = all_outcome_data[
            all_outcome_data["PATIENT_ID"].isin(all_ids - group_ids)
        ]
        if len(cohort_a_df) < min_group_size:
            return None
        try:
            stats = get_method_implementation(method_enum).calculate(
                cohort_a_df, cohort_b_df
            )
            p_value = stats.get("p_value")
            if p_value is None:
                return None
            return {
                "group": grp,
                "n_group": len(cohort_a_df),
                "n_rest": len(cohort_b_df),
                "hazard_ratio": stats.get("hazard_ratio"),
                "p_value": p_value,
            }
        except Exception:
            return None
    else:
        rest_ids = list(all_ids - group_ids)
        result = executor.compare_ids(list(group_ids), rest_ids, outcome, method_enum)
        if (
            not result.success
            or result.cohort_a_size < min_group_size
            or result.p_value is None
        ):
            return None
        return {
            "group": grp,
            "n_group": result.cohort_a_size,
            "n_rest": result.cohort_b_size,
            "hazard_ratio": result.hazard_ratio,
            "p_value": result.p_value,
        }


def scan_groupby(
    db: Database,
    executor: HypothesisExecutor,
    group_table: str,
    group_column: str,
    outcome_spec: str,
    method: str = "hazard_ratio_cox",
    min_group_size: int = 10,
    top_n: int = 20,
    max_groups: int = 200,
    cache_dir: Path | None = None,
) -> str:
    """Scan all unique values of a categorical column, one-vs-rest comparisons.

    Returns the top hits ranked by significance with FDR correction, formatted
    as a markdown table suitable for volcano plot inspection.
    """
    outcome = _parse_outcome(outcome_spec)
    method_enum = ComparisonMethod(method)

    # Cache check
    successful: list[dict] | None = None
    cache_file: Path | None = None
    if cache_dir is not None:
        key = _cache_key(
            group_table, group_column, outcome_spec, method, min_group_size, max_groups
        )
        cache_file = Path(cache_dir) / "volcano_cache" / f"{key}.json"
        if cache_file.exists():
            with contextlib.suppress(Exception):
                successful = json.loads(cache_file.read_text())

    if successful is None:
        # Batch ID resolution: one SQL query for all group → patient mappings
        join_info = TABLE_PATIENT_ID_JOINS.get(group_table)
        if join_info is not None:
            sample_key, join_table, join_column = join_info
            sql = (
                f'SELECT t."{group_column}" AS grp, cs."PATIENT_ID" '
                f'FROM "{group_table}" t '
                f'JOIN "{join_table}" cs ON t."{sample_key}" = cs."{join_column}" '
                f'WHERE t."{group_column}" IS NOT NULL'
            )
        else:
            sql = (
                f'SELECT "{group_column}" AS grp, "PATIENT_ID" '
                f'FROM "{group_table}" '
                f'WHERE "{group_column}" IS NOT NULL'
            )

        df = db.execute(sql)
        if df.empty:
            return "No data found for the specified table/column."

        grp_to_ids: dict[str, frozenset[str]] = {
            str(grp): frozenset(sub["PATIENT_ID"]) for grp, sub in df.groupby("grp")
        }

        # Limit to max_groups most-frequent values
        counts = Counter({grp: len(ids) for grp, ids in grp_to_ids.items()})
        top_groups = [grp for grp, _ in counts.most_common(max_groups)]

        # Base population: all patients (complement denominator)
        all_ids = frozenset(
            db.execute('SELECT "PATIENT_ID" FROM "clinical_patient"')[
                "PATIENT_ID"
            ].tolist()
        )

        # Pre-load outcome data for all patients in one query; falls back to per-group
        # queries (slow path) if the handler doesn't support bulk loading or the query
        # fails (e.g., table missing, schema mismatch).
        outcome_handler = get_outcome_handler(outcome)
        all_outcome_data: pd.DataFrame | None = None
        if hasattr(outcome_handler, "load_all_data"):
            with contextlib.suppress(Exception):
                all_outcome_data = outcome_handler.load_all_data(outcome, db)

        # Pre-filter groups below min_group_size, then run comparisons in parallel
        candidates = [
            (grp, grp_to_ids[grp])
            for grp in top_groups
            if len(grp_to_ids[grp]) >= min_group_size
        ]

        successful = []
        if candidates:
            n_workers = min(len(candidates), os.cpu_count() or 4)
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = [
                    pool.submit(
                        _compare_one,
                        grp,
                        ids,
                        all_ids,
                        outcome,
                        method_enum,
                        executor,
                        min_group_size,
                        all_outcome_data,
                    )
                    for grp, ids in candidates
                ]
                successful = [r for f in futures if (r := f.result()) is not None]

        # Cache write
        if cache_file is not None:
            with contextlib.suppress(Exception):
                cache_file.parent.mkdir(exist_ok=True)
                cache_file.write_text(json.dumps(successful))

    if not successful:
        return "No successful comparisons found."

    # Clip p-values to avoid BH underflow on literal 0.0 from Cox
    raw_ps = [r["p_value"] for r in successful]
    clipped_ps = [max(p, P_MIN) for p in raw_ps]
    q_values = false_discovery_control(clipped_ps, method="bh")

    for row, clipped_p, q in zip(successful, clipped_ps, q_values, strict=False):
        row["p_value_clipped"] = clipped_p
        row["q_value"] = q
        hr = row["hazard_ratio"]
        row["log2_hr"] = math.log2(hr) if hr is not None and hr > 0 else None

    # Sort: p_value_clipped ascending, then |log2_hr| descending (breaks ties)
    successful.sort(
        key=lambda r: (
            r["p_value_clipped"],
            -(abs(r["log2_hr"]) if r["log2_hr"] is not None else 0.0),
        )
    )
    top_rows = successful[:top_n]

    return _format_markdown(top_rows, method_enum)


def _format_markdown(rows: list[dict], method_enum: ComparisonMethod) -> str:
    if method_enum == ComparisonMethod.HAZARD_RATIO_COX:
        lines = [
            "| Group | N_group | N_rest | HR | log2(HR) | p_value | q_value |",
            "|-------|---------|--------|----|----------|---------|---------|",
        ]
        for r in rows:
            hr = f"{r['hazard_ratio']:.3f}" if r["hazard_ratio"] is not None else "NA"
            log2hr = f"{r['log2_hr']:.3f}" if r["log2_hr"] is not None else "NA"
            raw_p = r["p_value"]
            p_str = f"<{P_MIN:.0e}" if raw_p == 0.0 else f"{raw_p:.3e}"
            q_str = f"{r['q_value']:.3e}"
            lines.append(
                f"| {r['group']} | {r['n_group']} | {r['n_rest']} | "
                f"{hr} | {log2hr} | {p_str} | {q_str} |"
            )
    else:
        lines = [
            "| Group | N_group | N_rest | p_value | q_value |",
            "|-------|---------|--------|---------|---------|",
        ]
        for r in rows:
            raw_p = r["p_value"]
            p_str = f"<{P_MIN:.0e}" if raw_p == 0.0 else f"{raw_p:.3e}"
            q_str = f"{r['q_value']:.3e}"
            lines.append(
                f"| {r['group']} | {r['n_group']} | {r['n_rest']} | "
                f"{p_str} | {q_str} |"
            )
    return "\n".join(lines)
