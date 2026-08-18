"""Format prior code hypotheses for injection into the generator prompt.

Only short summaries are reinjected (the full script/result is fetchable on demand
via the get_script tool), so context doesn't grow with full history.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path

from geryon.codeflow.chains import DEFAULT_CHAIN
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
    output_dir: Path | None,
    current_session_id: str,
    chain: str | None = None,
) -> list[CodeHypothesis]:
    """Load code hypotheses from prior sessions under output_dir.

    Skips JSONL files that are not in the codeflow format (e.g. legacy sessions).
    When ``chain`` is given, only sessions belonging to that chain are returned — this
    is what keeps a separate line of investigation from being flooded by the main one.
    Sessions written before chains existed have no ``chain`` in their header and count
    as :data:`~geryon.codeflow.chains.DEFAULT_CHAIN`. Pass ``None`` to load every chain.
    """
    if output_dir is None or not output_dir.exists():
        return []
    out: list[CodeHypothesis] = []
    for jsonl_file in sorted(output_dir.rglob(HYPOTHESES_FILENAME)):
        header = _read_header(jsonl_file)
        if header is None or header.get("format") != "codeflow":
            continue
        if chain is not None and header.get("chain", DEFAULT_CHAIN) != chain:
            continue
        try:
            hyps = CodeHypothesisStore(jsonl_file.parent).load()
        except Exception:
            continue
        out.extend(h for h in hyps if h.session_id != current_session_id)
    return out


def _read_header(path: Path) -> dict | None:
    """Parse a JSONL store's metadata header, or None if it isn't readable."""
    try:
        with open(path) as f:
            header = json.loads(f.readline())
    except (OSError, json.JSONDecodeError):
        return None
    return header if isinstance(header, dict) else None
