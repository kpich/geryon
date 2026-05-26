"""Format prior hypotheses for injection into the LLM prompt."""

from dataclasses import dataclass, field
from pathlib import Path

from geryon.labeling.models import LabeledHypothesis
from geryon.labeling.storage import HypothesisStore

MAX_CONTEXT_HYPOTHESES = 200


@dataclass
class PreviousHypothesesContext:
    """Formatted context and metadata about previous hypotheses."""

    text: str
    ids: list[str] = field(default_factory=list)


def short_id(hypothesis_id: str) -> str:
    """Return the first 8 characters of a hypothesis ID."""
    return hypothesis_id[:8]


def _format_rating_tag(hyp: LabeledHypothesis) -> str:
    """Build compact [novelty=2, trust=3, dup] tag from rated dimensions."""
    rating = hyp.effective_rating
    parts: list[str] = []
    if rating.novelty is not None:
        parts.append(f"novelty={rating.novelty}")
    if rating.uncontrolled is not None:
        parts.append(f"uncontrolled={rating.uncontrolled}")
    if rating.trustworthiness is not None:
        parts.append(f"trust={rating.trustworthiness}")
    if rating.is_duplicate is True:
        parts.append("dup")
    return "[" + ", ".join(parts) + "]" if parts else ""


def _format_refines_tag(hyp: LabeledHypothesis) -> str:
    """Build optional (refines xxxxxxxx) tag."""
    ref = hyp.proposal.refines_hypothesis
    if ref:
        return f" (refines {short_id(ref)})"
    return ""


def _format_entry(hyp: LabeledHypothesis) -> str:
    """Build a compact single-line entry for the context block."""
    sid = short_id(hyp.hypothesis_id)

    ctx = hyp.narrative.context_summary if hyp.narrative else None
    if ctx:
        body = ctx
    else:
        desc = (
            f"{hyp.proposal.cohort_a_description} vs "
            f"{hyp.proposal.cohort_b_description}"
        )
        summary = (
            hyp.narrative.summary if hyp.narrative and hyp.narrative.summary else ""
        )
        body = f"{desc}" + (f" | {summary}" if summary else "")

    rating_tag = _format_rating_tag(hyp)
    refines = _format_refines_tag(hyp)
    notes = hyp.human_notes or hyp.notes

    entry = f"[{sid}] {body}"
    if rating_tag:
        entry += f" {rating_tag}"
    if refines:
        entry += refines
    if notes:
        entry += f" — {notes}"
    return entry


def load_prior_hypotheses(
    output_dir: Path | None, current_session_id: str
) -> list[LabeledHypothesis]:
    """Load hypotheses from prior sessions under output_dir.

    Parameters
    ----------
    output_dir : Path | None
        Parent output directory containing session subdirectories
    current_session_id : str
        Session ID to exclude (current session)

    Returns
    -------
    list[LabeledHypothesis]
        Hypotheses from all prior sessions
    """
    if output_dir is None or not output_dir.exists():
        return []
    all_hyps = []
    for jsonl_file in sorted(output_dir.rglob("hypotheses.jsonl")):
        store = HypothesisStore(jsonl_file.parent)
        try:
            hyps = store.load()
        except Exception:
            continue
        for h in hyps:
            if h.session_id != current_session_id:
                all_hyps.append(h)
    return all_hyps


def format_previous_hypotheses(
    labeled: list[LabeledHypothesis],
    session_previous: list[LabeledHypothesis],
) -> PreviousHypothesesContext:
    """Format all prior hypotheses as a compact flat list for LLM context.

    Parameters
    ----------
    labeled : list[LabeledHypothesis]
        Previously rated hypotheses from the labeled store
    session_previous : list[LabeledHypothesis]
        Hypotheses generated earlier in this session (may overlap with labeled)
    """
    labeled_by_id = {h.hypothesis_id: h for h in labeled}

    # Merge: prefer labeled-store version (has human ratings) over session version
    seen: set[str] = set()
    merged: list[LabeledHypothesis] = []
    for h in session_previous:
        hid = h.hypothesis_id
        if hid not in seen:
            seen.add(hid)
            merged.append(labeled_by_id.get(hid, h))
    # Include any labeled hypotheses not already in session_previous
    for h in labeled:
        if h.hypothesis_id not in seen:
            seen.add(h.hypothesis_id)
            merged.append(h)

    if not merged:
        return PreviousHypothesesContext(text="**No previous hypotheses yet.**")

    # Newest first: session_previous is oldest-first (chronological JSONL order)
    merged = list(reversed(merged))

    cap = MAX_CONTEXT_HYPOTHESES
    included = merged[:cap]
    overflow = len(merged) - cap

    lines: list[str] = [
        "**PREVIOUSLY TESTED HYPOTHESES (avoid duplicates, learn from ratings):**"
    ]
    for idx, hyp in enumerate(included, 1):
        lines.append(f"{idx}. {_format_entry(hyp)}")
    if overflow > 0:
        lines.append(f"... and {overflow} more")

    return PreviousHypothesesContext(
        text="\n".join(lines),
        ids=[h.hypothesis_id for h in included],
    )
