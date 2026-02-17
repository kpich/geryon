"""Browser-based annotation server for labeling hypotheses.

Local HTTP server using Python stdlib. Serves a single HTML page that
lets reviewers label hypotheses via radio buttons instead of typing
label strings at a terminal prompt.
"""

from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import webbrowser

from msk_cycl.labeling.labeled_store import LabeledStore
from msk_cycl.labeling.labels import HypothesisLabel
from msk_cycl.labeling.storage import HypothesisStore


def _load_unlabeled(output_dir: Path, labeled_store: LabeledStore) -> list[dict]:
    """Load all hypotheses from session JSONLs, filter to unlabeled, newest first."""
    labeled_ids = labeled_store.labeled_ids()

    seen: dict[str, dict] = {}
    for jsonl_file in sorted(output_dir.rglob("*.jsonl")):
        session_id = jsonl_file.stem.replace("_hypotheses", "")
        store = HypothesisStore(jsonl_file.parent)
        try:
            hypotheses = store.load_session(session_id)
        except Exception:
            continue

        for hyp in hypotheses:
            hid = hyp.hypothesis_id
            if hid in labeled_ids or hid in seen:
                continue
            if hyp.label != HypothesisLabel.PENDING:
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
            }

    items = list(seen.values())
    items.sort(key=lambda x: x["created_at"], reverse=True)
    return items


def _count_all(output_dir: Path, labeled_store: LabeledStore) -> dict:
    """Return total / labeled / pending counts."""
    labeled_ids = labeled_store.labeled_ids()

    all_ids: set[str] = set()
    for jsonl_file in sorted(output_dir.rglob("*.jsonl")):
        session_id = jsonl_file.stem.replace("_hypotheses", "")
        store = HypothesisStore(jsonl_file.parent)
        try:
            hypotheses = store.load_session(session_id)
        except Exception:
            continue
        for hyp in hypotheses:
            all_ids.add(hyp.hypothesis_id)

    total = len(all_ids)
    labeled = len(labeled_ids & all_ids)
    return {"total": total, "labeled": labeled, "pending": total - labeled}


_LABELS = [label for label in HypothesisLabel if label != HypothesisLabel.PENDING]

_LABEL_DISPLAY = {
    "correct": "Correct",
    "red_herring": "Red Herring",
    "confounded": "Confounded",
    "data_issue": "Data Issue",
    "duplicate": "Duplicate",
    "not_novel": "Not Novel",
}

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
  .labels { display: flex; flex-wrap: wrap; gap: 8px; margin: 16px 0; }
  .labels label { display: flex; align-items: center; gap: 6px; padding: 8px 14px;
                  border: 2px solid #ddd; border-radius: 6px; cursor: pointer;
                  font-size: 14px; transition: all .15s; }
  .labels label:hover { border-color: #999; }
  .labels input[type=radio] { accent-color: #2563eb; }
  .labels input[type=radio]:checked + span { font-weight: 600; }
  .labels label:has(input:checked) { border-color: #2563eb; background: #eff6ff; }
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
const LABELS = """
    + json.dumps(
        [{"value": lb.value, "display": _LABEL_DISPLAY[lb.value]} for lb in _LABELS]
    )
    + """;

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
    card.innerHTML = `
      <h2>#${idx + 1} &mdash; ${hid_short}... &mdash; ${h.created_at}</h2>
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
        <div class="field-label">Label</div>
        <div class="labels">
          ${LABELS.map(lb =>
            '<label><input type="radio" name="label-'
            + h.hypothesis_id + '" value="' + lb.value
            + '"><span>' + lb.display
            + '</span></label>').join("")}
        </div>
      </div>
      <div class="field">
        <div class="field-label">Notes (optional)</div>
        <textarea id="notes-${h.hypothesis_id}"></textarea>
      </div>
      <button class="submit" onclick="submitLabel('${h.hypothesis_id}')">Submit</button>
    `;
    container.appendChild(card);
  });
}

function esc(s) {
  if (s == null) return "";
  const d = document.createElement("div");
  d.textContent = String(s);
  return d.innerHTML;
}

async function submitLabel(hid) {
  const sel = document.querySelector('input[name="label-' + hid + '"]:checked');
  if (!sel) { showMsg("Select a label first.", true); return; }

  const notes = document.getElementById("notes-" + hid).value.trim();
  const btn = document.querySelector("#card-" + hid + " button.submit");
  btn.disabled = true;
  btn.textContent = "Saving...";

  const r = await fetch("/api/label", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({hypothesis_id: hid, label: sel.value, notes: notes || null}),
  });

  if (r.ok) {
    document.getElementById("card-" + hid).remove();
    showMsg("Labeled " + hid.substring(0, 8) + "...", false);
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
            label_val = body.get("label")
            notes = body.get("notes")

            if not hid or not label_val:
                self.send_error(400, "Missing hypothesis_id or label")
                return

            try:
                label = HypothesisLabel(label_val)
            except ValueError:
                self.send_error(400, f"Invalid label: {label_val}")
                return

            hyp = self._find_hypothesis(hid)
            if hyp is None:
                self.send_error(404, f"Hypothesis {hid} not found")
                return

            hyp.label = label
            hyp.label_notes = notes
            hyp.labeled_at = datetime.now(UTC)
            hyp.labeled_by = "annotator"

            labeled_store.save(hyp)
            self._json_response({"ok": True})

        def _find_hypothesis(self, hypothesis_id: str):
            for jsonl_file in output_dir.rglob("*.jsonl"):
                session_id = jsonl_file.stem.replace("_hypotheses", "")
                store = HypothesisStore(jsonl_file.parent)
                try:
                    for hyp in store.load_session(session_id):
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
