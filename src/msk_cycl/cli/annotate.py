"""Browser-based annotation server for labeling hypotheses.

Local HTTP server using Python stdlib. Serves a single HTML page that
lets reviewers rate hypotheses on multiple dimensions instead of typing
label strings at a terminal prompt.
"""

from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import webbrowser

from msk_cycl.labeling.labeled_store import LabeledStore
from msk_cycl.labeling.labels import RATING_DIMENSIONS, HypothesisRating
from msk_cycl.labeling.storage import HypothesisStore


def _load_unlabeled(output_dir: Path, labeled_store: LabeledStore) -> list[dict]:
    """Load all hypotheses from session JSONLs, filter to unlabeled, newest first."""
    labeled_ids = labeled_store.labeled_ids()

    seen: dict[str, dict] = {}
    for jsonl_file in sorted(output_dir.rglob("hypotheses.jsonl")):
        store = HypothesisStore(jsonl_file.parent)
        try:
            hypotheses = store.load()
        except Exception:
            continue

        for hyp in hypotheses:
            hid = hyp.hypothesis_id
            if hid in labeled_ids or hid in seen:
                continue
            if not hyp.rating.is_pending and hyp.labeled_by != "llm_critic":
                continue

            seen[hid] = {
                "hypothesis_id": hid,
                "session_id": hyp.session_id,
                "created_at": str(hyp.created_at),
                "cohort_a_description": hyp.proposal.cohort_a_description,
                "cohort_b_description": hyp.proposal.cohort_b_description,
                "outcome_description": hyp.proposal.outcome_description,
                "rationale": hyp.proposal.rationale,
                "cohort_a_size": hyp.result.cohort_a_size,
                "cohort_b_size": hyp.result.cohort_b_size,
                "hazard_ratio": hyp.result.hazard_ratio,
                "confidence_interval_lower": hyp.result.confidence_interval_lower,
                "confidence_interval_upper": hyp.result.confidence_interval_upper,
                "p_value": hyp.result.p_value,
                "spec": hyp.spec.model_dump(),
                "summary": hyp.narrative.summary,
                "findings": hyp.narrative.findings,
                "limitations": hyp.narrative.limitations,
                "clinical_relevance": hyp.narrative.clinical_relevance,
                "iteration": hyp.iteration,
                "labeled_by": hyp.labeled_by,
                "critic_rating": (
                    {
                        "novelty": hyp.rating.novelty,
                        "uncontrolled": hyp.rating.uncontrolled,
                        "trustworthiness": hyp.rating.trustworthiness,
                        "is_duplicate": hyp.rating.is_duplicate,
                    }
                    if hyp.labeled_by == "llm_critic"
                    else None
                ),
                "critic_notes": (hyp.notes if hyp.labeled_by == "llm_critic" else None),
            }

    items = list(seen.values())
    items.sort(key=lambda x: x["created_at"], reverse=True)
    return items


def _count_all(output_dir: Path, labeled_store: LabeledStore) -> dict:
    """Return total / labeled / pending counts."""
    labeled_ids = labeled_store.labeled_ids()

    all_ids: set[str] = set()
    for jsonl_file in sorted(output_dir.rglob("hypotheses.jsonl")):
        store = HypothesisStore(jsonl_file.parent)
        try:
            hypotheses = store.load()
        except Exception:
            continue
        for hyp in hypotheses:
            all_ids.add(hyp.hypothesis_id)

    total = len(all_ids)
    labeled = len(labeled_ids & all_ids)
    return {"total": total, "labeled": labeled, "pending": total - labeled}


# ruff: noqa: E501
HTML_PAGE = (
    """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CYCL Hypothesis Annotator</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: #f5f5f5; color: #333; line-height: 1.5; padding: 20px; }
  h1 { text-align: center; margin-bottom: 8px; }
  #stats { text-align: center; color: #666; margin-bottom: 24px; font-size: 14px; }
  .empty { text-align: center; color: #999; margin-top: 60px; font-size: 18px; }
  .card { background: #fff; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,.12);
          padding: 24px; margin: 0 auto 24px; max-width: 800px; }
  .card h2 { font-size: 14px; color: #999; margin-bottom: 12px; }
  .field { margin-bottom: 12px; }
  .field-label { font-weight: 600; font-size: 13px; color: #666;
                 text-transform: uppercase; letter-spacing: .5px;
                 margin-bottom: 2px; }
  .field-value { font-size: 15px; }
  table.stats { border-collapse: collapse; width: 100%; margin: 8px 0; }
  table.stats th, table.stats td { padding: 6px 12px; text-align: left;
                                    border-bottom: 1px solid #eee; font-size: 14px; }
  table.stats th { color: #666; font-weight: 600; }
  .dim-row { display: flex; align-items: center; gap: 12px; margin: 10px 0; }
  .dim-label { min-width: 130px; font-weight: 600; font-size: 14px; }
  .dim-end { font-size: 12px; color: #888; min-width: 90px; }
  .dim-end.right { text-align: right; }
  .dim-buttons { display: flex; gap: 4px; }
  .dim-buttons button { width: 36px; height: 36px; border: 2px solid #ddd;
                         border-radius: 6px; background: #fff; cursor: pointer;
                         font-size: 15px; font-weight: 600; transition: all .15s; }
  .dim-buttons button:hover { border-color: #999; }
  .dim-buttons button.selected { border-color: #2563eb; background: #eff6ff;
                                   color: #2563eb; }
  .dup-row { display: flex; align-items: center; gap: 8px; margin: 10px 0; }
  .dup-row label { font-size: 14px; cursor: pointer; display: flex;
                   align-items: center; gap: 6px; }
  textarea { width: 100%; height: 60px; border: 1px solid #ddd; border-radius: 6px;
             padding: 8px; font-family: inherit; font-size: 14px; resize: vertical; }
  button.submit { display: block; margin: 16px auto 0; padding: 10px 32px;
                  background: #2563eb; color: #fff; border: none; border-radius: 6px;
                  font-size: 15px; font-weight: 600; cursor: pointer; }
  button.submit:hover { background: #1d4ed8; }
  button.submit:disabled { background: #93c5fd; cursor: not-allowed; }
  .msg { text-align: center; padding: 12px; border-radius: 6px; margin: 0 auto 16px;
         max-width: 800px; font-size: 14px; }
  .msg.ok { background: #dcfce7; color: #166534; }
  .msg.err { background: #fee2e2; color: #991b1b; }
  pre.spec { background: #f8f8f8; border: 1px solid #e0e0e0;
             border-radius: 6px; padding: 12px; font-size: 13px;
             overflow-x: auto; line-height: 1.4; }
</style>
</head>
<body>
<h1>CYCL Hypothesis Annotator</h1>
<div id="stats"></div>
<div id="msg"></div>
<div id="cards"></div>

<script>
const DIMENSIONS = """
    + json.dumps(
        {
            k: {
                "label": v["label"],
                "levels": {str(lk): lv for lk, lv in v["levels"].items()},
            }
            for k, v in RATING_DIMENSIONS.items()
        }
    )
    + """;

const DIM_ENDPOINTS = {
  novelty: ["Unsurprising", "Intriguing"],
  uncontrolled: ["Clean", "Heavily mixed"],
  trustworthiness: ["Suspect", "Credible"],
};

async function loadStats() {
  const r = await fetch("/api/stats");
  const s = await r.json();
  document.getElementById("stats").textContent =
    s.total + " total | " + s.labeled + " labeled | " + s.pending + " pending";
}

async function loadHypotheses() {
  const r = await fetch("/api/hypotheses");
  const hyps = await r.json();
  const container = document.getElementById("cards");
  container.innerHTML = "";

  if (hyps.length === 0) {
    container.innerHTML = '<div class="empty">No unlabeled hypotheses.</div>';
    return;
  }

  hyps.forEach((h, idx) => {
    const card = document.createElement("div");
    card.className = "card";
    card.id = "card-" + h.hypothesis_id;

    const fmtVal = (v, digits) => v != null ? Number(v).toFixed(digits) : "N/A";

    const hid_short = h.hypothesis_id.substring(0, 8);
    const iterTag = h.iteration ? ` | iter ${h.iteration}` : "";
    const criticBadge = h.labeled_by === "llm_critic"
      ? ' <span style="background:#f59e0b;color:#fff;font-size:10px;font-variant:small-caps;padding:2px 6px;border-radius:3px;margin-left:6px">LLM CRITIC</span>'
      : "";

    let dimHTML = "";
    for (const [key, dim] of Object.entries(DIMENSIONS)) {
      const ends = DIM_ENDPOINTS[key] || ["", ""];
      dimHTML += `<div class="dim-row">
        <span class="dim-label">${dim.label}</span>
        <span class="dim-end">${ends[0]}</span>
        <span class="dim-buttons">
          ${[1,2,3].map(n =>
            `<button type="button" data-dim="${key}" data-val="${n}"
                     title="${dim.levels[n]}"
                     onclick="selectDim(this,'${h.hypothesis_id}','${key}',${n})">${n}</button>`
          ).join("")}
        </span>
        <span class="dim-end right">${ends[1]}</span>
      </div>`;
    }

    card.innerHTML = `
      <h2>#${idx + 1} &mdash; ${hid_short}...${iterTag} &mdash; ${h.created_at}${criticBadge}</h2>
      <div class="field">
        <div class="field-label">Cohort A</div>
        <div class="field-value">${esc(h.cohort_a_description)}</div>
      </div>
      <div class="field">
        <div class="field-label">Cohort B</div>
        <div class="field-value">${esc(h.cohort_b_description)}</div>
      </div>
      <div class="field">
        <div class="field-label">Rationale</div>
        <div class="field-value">${esc(h.rationale)}</div>
      </div>
      <table class="stats">
        <tr><th>N (A)</th><th>N (B)</th><th>HR</th><th>95% CI</th><th>p-value</th></tr>
        <tr>
          <td>${h.cohort_a_size}</td>
          <td>${h.cohort_b_size}</td>
          <td>${fmtVal(h.hazard_ratio, 3)}</td>
          <td>[${fmtVal(h.confidence_interval_lower, 3)},
              ${fmtVal(h.confidence_interval_upper, 3)}]</td>
          <td>${fmtVal(h.p_value, 4)}</td>
        </tr>
      </table>
      <div class="field">
        <div class="field-label">Spec</div>
        <pre class="spec">${esc(
          JSON.stringify(h.spec, null, 2)
        )}</pre>
      </div>
      <div class="field">
        <div class="field-label">Summary</div>
        <div class="field-value">${esc(h.summary)}</div>
      </div>
      <div class="field">
        <div class="field-label">Ratings</div>
        ${dimHTML}
        <div class="dup-row">
          <label><input type="checkbox" id="dup-${h.hypothesis_id}"> Duplicate</label>
        </div>
      </div>
      <div class="field">
        <div class="field-label">Notes (optional)</div>
        <textarea id="notes-${h.hypothesis_id}"></textarea>
      </div>
      <button class="submit"
        onclick="submitRating('${h.hypothesis_id}')">Submit</button>
    `;
    container.appendChild(card);

    // Pre-fill critic ratings
    if (h.critic_rating) {
      if (!selections[h.hypothesis_id]) selections[h.hypothesis_id] = {};
      for (const [dim, val] of Object.entries(h.critic_rating)) {
        if (dim === "is_duplicate") {
          if (val) {
            const dupEl = document.getElementById("dup-" + h.hypothesis_id);
            if (dupEl) dupEl.checked = true;
          }
        } else if (val != null) {
          selections[h.hypothesis_id][dim] = val;
          const btn = card.querySelector(
                `button[data-dim="${dim}"][data-val="${val}"]`);
          if (btn) btn.classList.add("selected");
        }
      }
    }
    if (h.critic_notes) {
      const notesEl = document.getElementById("notes-" + h.hypothesis_id);
      if (notesEl) notesEl.value = h.critic_notes;
    }
  });
}

const selections = {};

function selectDim(btn, hid, dim, val) {
  if (!selections[hid]) selections[hid] = {};
  selections[hid][dim] = val;
  const row = btn.parentElement;
  row.querySelectorAll("button").forEach(b => b.classList.remove("selected"));
  btn.classList.add("selected");
}

function esc(s) {
  if (s == null) return "";
  const d = document.createElement("div");
  d.textContent = String(s);
  return d.innerHTML;
}

async function submitRating(hid) {
  const sel = selections[hid] || {};
  const dupEl = document.getElementById("dup-" + hid);
  const isDup = dupEl ? dupEl.checked : false;

  const hasAnyDim = sel.novelty || sel.uncontrolled || sel.trustworthiness || isDup;
  if (!hasAnyDim) { showMsg("Rate at least one dimension.", true); return; }

  const notes = document.getElementById("notes-" + hid).value.trim();
  const btn = document.querySelector("#card-" + hid + " button.submit");
  btn.disabled = true;
  btn.textContent = "Saving...";

  const r = await fetch("/api/label", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      hypothesis_id: hid,
      novelty: sel.novelty || null,
      uncontrolled: sel.uncontrolled || null,
      trustworthiness: sel.trustworthiness || null,
      is_duplicate: isDup || null,
      notes: notes || null,
    }),
  });

  if (r.ok) {
    document.getElementById("card-" + hid).remove();
    showMsg("Rated " + hid.substring(0, 8) + "...", false);
    loadStats();
    if (!document.querySelector(".card")) {
      document.getElementById("cards").innerHTML =
        '<div class="empty">No unlabeled hypotheses.</div>';
    }
  } else {
    const e = await r.text();
    showMsg("Error: " + e, true);
    btn.disabled = false;
    btn.textContent = "Submit";
  }
}

function showMsg(text, isErr) {
  const el = document.getElementById("msg");
  el.className = "msg " + (isErr ? "err" : "ok");
  el.textContent = text;
  setTimeout(() => { el.textContent = ""; el.className = ""; }, 4000);
}

loadStats();
loadHypotheses();
</script>
</body>
</html>"""
)


def _make_handler(output_dir: Path, labeled_store: LabeledStore):
    """Create request handler class with closure over config."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/":
                self._serve_html()
            elif self.path == "/api/hypotheses":
                self._serve_hypotheses()
            elif self.path == "/api/stats":
                self._serve_stats()
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path == "/api/label":
                self._handle_label()
            else:
                self.send_error(404)

        def _serve_html(self):
            body = HTML_PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _serve_hypotheses(self):
            data = _load_unlabeled(output_dir, labeled_store)
            self._json_response(data)

        def _serve_stats(self):
            data = _count_all(output_dir, labeled_store)
            self._json_response(data)

        def _handle_label(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))

            hid = body.get("hypothesis_id")
            if not hid:
                self.send_error(400, "Missing hypothesis_id")
                return

            rating = HypothesisRating(
                novelty=body.get("novelty"),
                uncontrolled=body.get("uncontrolled"),
                trustworthiness=body.get("trustworthiness"),
                is_duplicate=body.get("is_duplicate"),
            )

            if rating.is_pending:
                self.send_error(400, "At least one dimension must be rated")
                return

            hyp = self._find_hypothesis(hid)
            if hyp is None:
                self.send_error(404, f"Hypothesis {hid} not found")
                return

            hyp.rating = rating
            hyp.notes = body.get("notes")
            hyp.labeled_at = datetime.now(UTC)
            hyp.labeled_by = "annotator"

            labeled_store.save(hyp)
            self._json_response({"ok": True})

        def _find_hypothesis(self, hypothesis_id: str):
            for jsonl_file in output_dir.rglob("hypotheses.jsonl"):
                store = HypothesisStore(jsonl_file.parent)
                try:
                    for hyp in store.load():
                        if hyp.hypothesis_id == hypothesis_id:
                            return hyp
                except Exception:
                    continue
            return None

        def _json_response(self, data, status=200):
            body = json.dumps(data, default=str).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    return Handler


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Browser-based hypothesis annotator")
    parser.add_argument(
        "--output-dir",
        default="cycl_run_outputs",
        help="Directory containing session JSONL files (default: cycl_run_outputs/)",
    )
    parser.add_argument(
        "--labeled-dir",
        default="labeled_hypotheses",
        help="Directory for labeled hypothesis JSON files "
        "(default: labeled_hypotheses/)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to serve on (default: 8765)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    labeled_store = LabeledStore(Path(args.labeled_dir))

    handler = _make_handler(output_dir, labeled_store)
    server = HTTPServer(("localhost", args.port), handler)

    url = f"http://localhost:{args.port}"
    print(f"Annotation server running at {url}")
    print("Press Ctrl+C to stop.\n")
    webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
