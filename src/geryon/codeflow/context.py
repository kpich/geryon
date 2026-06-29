"""Format prior code hypotheses for injection into the generator prompt.

Only short summaries are reinjected (the full script/result is fetchable on demand
via the get_script tool), so context doesn't grow with full history.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path

from geryon.codeflow.models import CodeHypothesis
from geryon.codeflow.store import HYPOTHESES_FILENAME, CodeHypothesisStore

MAX_CONTEXT_HYPOTHESES = 200


@dataclass
class PreviousContext:
    text: str
    ids: list[str] = field(default_factory=list)


def _format_entry(hyp: CodeHypothesis) -> str:
    """One compact line: id, summary, headline numbers, lineage."""
    body = ""
    if hyp.narrative and hyp.narrative.context_summary:
        body = hyp.narrative.context_summary
    elif hyp.narrative and hyp.narrative.summary:
        body = hyp.narrative.summary
    else:
        body = hyp.title

    stat = ""
    if hyp.result and hyp.result.effect_size is not None:
        stat = f" [effect={hyp.result.effect_size:.3g}"
        if hyp.result.p_value is not None:
            stat += f", p={hyp.result.p_value:.2g}"
        stat += "]"
    elif not hyp.success:
        stat = " [FAILED]"

    refines = f" (refines {hyp.refines[:8]})" if hyp.refines else ""
    return f"[{hyp.short_id()}] {body}{stat}{refines}"


def format_previous_hypotheses(hypotheses: list[CodeHypothesis]) -> PreviousContext:
    """Render prior hypotheses as a compact newest-first list, capped."""
    if not hypotheses:
        return PreviousContext(text="**No previous hypotheses yet.**")

    ordered = list(reversed(hypotheses))  # JSONL is oldest-first
    included = ordered[:MAX_CONTEXT_HYPOTHESES]
    overflow = len(ordered) - len(included)

    lines = [
        "**PREVIOUSLY TESTED HYPOTHESES "
        "(avoid duplicates; refine strong ones via get_script + submit(refines=...)):**"
    ]
    for idx, hyp in enumerate(included, 1):
        lines.append(f"{idx}. {_format_entry(hyp)}")
    if overflow > 0:
        lines.append(f"... and {overflow} more")

    return PreviousContext(
        text="\n".join(lines), ids=[h.hypothesis_id for h in included]
    )


def load_prior_hypotheses(
    output_dir: Path | None, current_session_id: str
) -> list[CodeHypothesis]:
    """Load code hypotheses from prior sessions under output_dir.

    Skips JSONL files that are not in the codeflow format (e.g. legacy sessions).
    """
    if output_dir is None or not output_dir.exists():
        return []
    out: list[CodeHypothesis] = []
    for jsonl_file in sorted(output_dir.rglob(HYPOTHESES_FILENAME)):
        if not _is_codeflow_file(jsonl_file):
            continue
        try:
            hyps = CodeHypothesisStore(jsonl_file.parent).load()
        except Exception:
            continue
        out.extend(h for h in hyps if h.session_id != current_session_id)
    return out


def _is_codeflow_file(path: Path) -> bool:
    """True if the file's metadata header marks it as codeflow format."""
    try:
        with open(path) as f:
            first = json.loads(f.readline())
        return first.get("format") == "codeflow"
    except (OSError, json.JSONDecodeError):
        return False
