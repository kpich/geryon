"""Run external validation: biomarker -> OS hazard ratio on matching TCGA study.

For each biomarker extracted from the Geryon corpus, fetch the matching TCGA
PanCancer Atlas cohort from cBioPortal and compute an altered-vs-wildtype overall
survival hazard ratio using Geryon's own Cox method (geryon.engine.methods.
CoxHazardRatioMethod) -- the same statistical core the live engine uses.

Standard (non-truncated) Cox is correct here: TCGA time-zero is diagnosis, so the
sequencing-anchored left-truncation Geryon applies to MSK-IMPACT does not apply.

Outputs out/results.csv. No LLM calls; read-only against the public API.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import os

from biomarkers import Biomarker, extract_biomarkers
import cbioportal
import pandas as pd  # type: ignore
from statsmodels.stats.multitest import multipletests  # type: ignore

from geryon.engine.methods import CoxHazardRatioMethod

# Stability guards: tiny altered arms give wild, uninterpretable HRs.
MIN_ALTERED = 5
MIN_EVENTS_PER_ARM = 3

HERE = os.path.dirname(os.path.abspath(__file__))


def _altered_patients(
    bm: Biomarker,
    entrez: int,
    muts: dict[int, set[str]],
    cnas: dict[int, dict[str, int]],
) -> set[str]:
    if bm.alt_kind == "mut":
        return set(muts.get(entrez, set()))
    gene_cna = cnas.get(entrez, {})
    want = 2 if bm.cna_dir > 0 else -2
    return {pid for pid, alt in gene_cna.items() if alt == want}


def _os_frame(os_map: dict[str, tuple[float, int]], pids: set[str]) -> pd.DataFrame:
    rows = [
        {"PATIENT_ID": p, "time": os_map[p][0], "event": os_map[p][1]}
        for p in pids
        if p in os_map and os_map[p][0] > 0
    ]
    return pd.DataFrame(rows)


def compute_rows(biomarkers: list[Biomarker]) -> list[dict]:
    entrez = cbioportal.resolve_entrez([b.gene for b in biomarkers])

    by_study: dict[str, list[Biomarker]] = defaultdict(list)
    for b in biomarkers:
        by_study[b.study_id].append(b)

    cox = CoxHazardRatioMethod()
    rows: list[dict] = []

    for study, bms in by_study.items():
        print(f"[{study}] {len(bms)} biomarkers")
        os_map = cbioportal.fetch_os(study)
        mut_ids = [
            entrez[b.gene] for b in bms if b.alt_kind == "mut" and b.gene in entrez
        ]
        cna_ids = [
            entrez[b.gene] for b in bms if b.alt_kind == "cna" and b.gene in entrez
        ]
        muts = cbioportal.fetch_mutated_patients(study, mut_ids)
        cnas = cbioportal.fetch_cna_patients(study, cna_ids)

        for b in bms:
            row = {
                "label": b.label,
                "cancer_type": b.cancer_type,
                "gene": b.gene,
                "alt_kind": b.alt_kind,
                "study_id": study,
                "hazard_ratio": None,
                "ci_lower": None,
                "ci_upper": None,
                "p_value": None,
                "n_altered": 0,
                "n_wildtype": 0,
                "status": "ok",
            }
            if b.gene not in entrez:
                row["status"] = "gene_not_found"
                rows.append(row)
                continue

            altered = _altered_patients(b, entrez[b.gene], muts, cnas)
            all_pids = set(os_map)
            alt_df = _os_frame(os_map, altered & all_pids)
            wt_df = _os_frame(os_map, all_pids - altered)
            row["n_altered"] = len(alt_df)
            row["n_wildtype"] = len(wt_df)

            if len(alt_df) < MIN_ALTERED:
                row["status"] = "too_few_altered"
                rows.append(row)
                continue
            if (
                alt_df["event"].sum() < MIN_EVENTS_PER_ARM
                or wt_df["event"].sum() < MIN_EVENTS_PER_ARM
            ):
                row["status"] = "too_few_events"
                rows.append(row)
                continue

            try:
                res = cox.calculate(alt_df, wt_df)
                row.update(
                    hazard_ratio=res["hazard_ratio"],
                    ci_lower=res["confidence_interval_lower"],
                    ci_upper=res["confidence_interval_upper"],
                    p_value=res["p_value"],
                )
            except Exception as e:  # lifelines convergence / singular fits
                row["status"] = f"cox_failed: {type(e).__name__}"
            rows.append(row)
    return rows


def add_qvalues(rows: list[dict]) -> None:
    ok = [r for r in rows if r["p_value"] is not None]
    if not ok:
        return
    qs = multipletests([r["p_value"] for r in ok], method="fdr_bh")[1]
    for r, q in zip(ok, qs, strict=False):
        r["q_value"] = float(q)
    for r in rows:
        r.setdefault("q_value", None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data-dir",
        default=os.path.expanduser("~/dev/cycl/geryon_data"),
        help="Geryon data dir containing sessions/",
    )
    ap.add_argument("--output", default=os.path.join(HERE, "out", "results.csv"))
    args = ap.parse_args()

    biomarkers = extract_biomarkers(args.data_dir)
    print(f"Extracted {len(biomarkers)} biomarkers from corpus")
    rows = compute_rows(biomarkers)
    add_qvalues(rows)

    fields = [
        "label",
        "cancer_type",
        "gene",
        "alt_kind",
        "study_id",
        "hazard_ratio",
        "ci_lower",
        "ci_upper",
        "p_value",
        "q_value",
        "n_altered",
        "n_wildtype",
        "status",
    ]
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})

    ok = [r for r in rows if r["p_value"] is not None]
    sig = [r for r in ok if r["q_value"] is not None and r["q_value"] < 0.05]
    print(f"\nWrote {args.output}")
    print(f"  {len(ok)}/{len(rows)} biomarkers tested; {len(sig)} significant (q<0.05)")


if __name__ == "__main__":
    main()
