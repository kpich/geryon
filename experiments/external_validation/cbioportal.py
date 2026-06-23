"""Minimal cBioPortal public REST API client (stdlib only, no new deps).

Fetches just what the external-validation forest plot needs:
  - hugo -> entrez gene id resolution
  - per-study overall survival (parsed to time/event)
  - per-gene mutated patient sets
  - per-gene discrete CNA (deep del / amp) patient sets

Public endpoint, no auth. See https://www.cbioportal.org/api/swagger-ui/
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

API = "https://www.cbioportal.org/api"


def _request(path: str, body: object | None = None, retries: int = 3) -> object:
    url = f"{API}{path}"
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode())
        except (urllib.error.URLError, TimeoutError) as e:  # pragma: no cover - net
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"cBioPortal request failed: {url}: {last_err}")


def resolve_entrez(hugo_symbols: list[str]) -> dict[str, int]:
    """Map HUGO symbols to Entrez gene ids (one call for all genes)."""
    if not hugo_symbols:
        return {}
    rows = _request(
        "/genes/fetch?geneIdType=HUGO_GENE_SYMBOL", sorted(set(hugo_symbols))
    )
    assert isinstance(rows, list)
    return {r["hugoGeneSymbol"]: r["entrezGeneId"] for r in rows}


def fetch_os(study_id: str) -> dict[str, tuple[float, int]]:
    """Return {patient_id: (os_months, event)} for a study.

    OS_STATUS is cBioPortal-encoded ("1:DECEASED"/"0:LIVING"); event is the
    leading digit, matching geryon's OverallSurvivalHandler convention.
    """

    def _attr(attr: str) -> dict[str, str]:
        rows = _request(
            f"/studies/{study_id}/clinical-data"
            f"?clinicalDataType=PATIENT&attributeId={attr}&pageSize=100000"
        )
        assert isinstance(rows, list)
        return {r["patientId"]: r["value"] for r in rows}

    months = _attr("OS_MONTHS")
    status = _attr("OS_STATUS")
    out: dict[str, tuple[float, int]] = {}
    for pid, raw_m in months.items():
        raw_s = status.get(pid)
        if raw_s is None:
            continue
        try:
            m = float(raw_m)
            ev = int(str(raw_s).split(":", 1)[0])
        except (ValueError, TypeError):
            continue
        out[pid] = (m, ev)
    return out


def fetch_mutated_patients(study_id: str, entrez_ids: list[int]) -> dict[int, set[str]]:
    """Return {entrez_id: set(patient_id mutated)} for a study (one call)."""
    if not entrez_ids:
        return {}
    rows = _request(
        f"/molecular-profiles/{study_id}_mutations/mutations/fetch?projection=ID",
        {"sampleListId": f"{study_id}_all", "entrezGeneIds": sorted(set(entrez_ids))},
    )
    assert isinstance(rows, list)
    out: dict[int, set[str]] = {e: set() for e in entrez_ids}
    for r in rows:
        out.setdefault(r["entrezGeneId"], set()).add(r["patientId"])
    return out


def fetch_cna_patients(
    study_id: str, entrez_ids: list[int]
) -> dict[int, dict[str, int]]:
    """Return {entrez_id: {patient_id: alteration}} where alteration in {-2,2}.

    Aggregates to patient level taking the most extreme alteration across the
    patient's samples.
    """
    if not entrez_ids:
        return {}
    rows = _request(
        f"/molecular-profiles/{study_id}_gistic/discrete-copy-number/fetch"
        f"?discreteCopyNumberEventType=HOMDEL_AND_AMP&projection=ID",
        {"sampleListId": f"{study_id}_all", "entrezGeneIds": sorted(set(entrez_ids))},
    )
    assert isinstance(rows, list)
    out: dict[int, dict[str, int]] = {e: {} for e in entrez_ids}
    for r in rows:
        gene = out.setdefault(r["entrezGeneId"], {})
        pid = r["patientId"]
        alt = int(r["alteration"])
        if pid not in gene or abs(alt) > abs(gene[pid]):
            gene[pid] = alt
    return out
