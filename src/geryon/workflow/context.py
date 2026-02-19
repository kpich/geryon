"""Format prior hypotheses for injection into the LLM prompt."""

from geryon.labeling.models import LabeledHypothesis

MAX_RATED = 50
MAX_SESSION = 20


def short_id(hypothesis_id: str) -> str:
    """Return the first 8 characters of a hypothesis ID."""
    return hypothesis_id[:8]


def _format_rating_tag(hyp: LabeledHypothesis) -> str:
    """Build compact [novelty=2, trust=3, dup] tag from rated dimensions."""
    parts: list[str] = []
    if hyp.rating.novelty is not None:
        parts.append(f"novelty={hyp.rating.novelty}")
    if hyp.rating.uncontrolled is not None:
        parts.append(f"uncontrolled={hyp.rating.uncontrolled}")
    if hyp.rating.trustworthiness is not None:
        parts.append(f"trust={hyp.rating.trustworthiness}")
    if hyp.rating.is_duplicate is True:
        parts.append("dup")
    return "[" + ", ".join(parts) + "]"


def _format_refines_tag(hyp: LabeledHypothesis) -> str:
    """Build optional (refines xxxxxxxx) tag."""
    ref = hyp.proposal.refines_hypothesis
    if ref:
        return f" (refines {short_id(ref)})"
    return ""


def _format_summary_suffix(hyp: LabeledHypothesis) -> str:
    """Build optional | summary suffix from narrative."""
    if hyp.narrative and hyp.narrative.summary:
        return f" | {hyp.narrative.summary}"
    return ""


def format_previous_hypotheses(
    labeled: list[LabeledHypothesis],
    session_previous: list[LabeledHypothesis],
) -> str:
    """Format labeled and same-session hypotheses for LLM context.

    Parameters
    ----------
    labeled : list[LabeledHypothesis]
        Previously rated hypotheses from the labeled store
    session_previous : list[LabeledHypothesis]
        Hypotheses generated earlier in this session (may overlap with labeled)
    """
    labeled_ids = {h.hypothesis_id for h in labeled}

    rated = [h for h in labeled if not h.rating.is_pending]

    unrated_session = [
        h for h in session_previous if h.hypothesis_id not in labeled_ids
    ]

    if not rated and not unrated_session:
        return "**No previous hypotheses yet.**"

    lines: list[str] = []
    idx = 1

    if rated:
        lines.append("**PREVIOUSLY RATED HYPOTHESES (learn from this feedback):**")
        for hyp in rated[:MAX_RATED]:
            sid = short_id(hyp.hypothesis_id)
            desc = (
                f"{hyp.proposal.cohort_a_description} vs "
                f"{hyp.proposal.cohort_b_description}"
            )
            tag = _format_rating_tag(hyp)
            refines = _format_refines_tag(hyp)
            summary = _format_summary_suffix(hyp)
            entry = f"{idx}. [{sid}] {desc} {tag}{refines}"
            if hyp.notes:
                entry += f" — {hyp.notes}"
            entry += summary
            lines.append(entry)
            idx += 1
        if len(rated) > MAX_RATED:
            lines.append(f"... and {len(rated) - MAX_RATED} more rated hypotheses")

    if unrated_session:
        lines.append("**PREVIOUSLY TESTED THIS SESSION (avoid duplicates):**")
        for hyp in unrated_session[:MAX_SESSION]:
            sid = short_id(hyp.hypothesis_id)
            desc = (
                f"{hyp.proposal.cohort_a_description} vs "
                f"{hyp.proposal.cohort_b_description}"
            )
            refines = _format_refines_tag(hyp)
            summary = _format_summary_suffix(hyp)
            entry = f"{idx}. [{sid}] {desc}"
            if not hyp.rating.is_pending:
                tag = _format_rating_tag(hyp)
                source = "critic" if hyp.labeled_by == "llm_critic" else "auto"
                entry += f" {tag} [{source}]"
                if hyp.notes:
                    entry += f" — {hyp.notes}"
            entry += refines + summary
            lines.append(entry)
            idx += 1
        if len(unrated_session) > MAX_SESSION:
            lines.append(f"... and {len(unrated_session) - MAX_SESSION} more")

    return "\n".join(lines)
