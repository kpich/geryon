"""Format prior hypotheses for injection into the LLM prompt."""

from dataclasses import dataclass, field
from pathlib import Path

from geryon.labeling.models import LabeledHypothesis
from geryon.labeling.storage import HypothesisStore

MAX_RATED = 50
MAX_SESSION = 20


@dataclass
class PreviousHypothesesContext:
    """Formatted context and metadata about previous hypotheses."""

    text: str
    rated_ids: list[str] = field(default_factory=list)
    unrated_session_ids: list[str] = field(default_factory=list)


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
    return "[" + ", ".join(parts) + "]"


def _source_label(hyp: LabeledHypothesis) -> str:
    """Return 'human', 'critic', or 'auto' based on rating source."""
    hr = hyp.human_rating
    if hr is not None and not hr.is_pending:
        return "human"
    if hyp.labeled_by == "llm_critic":
        return "critic"
    return "auto"


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
    """Format labeled and same-session hypotheses for LLM context.

    Parameters
    ----------
    labeled : list[LabeledHypothesis]
        Previously rated hypotheses from the labeled store
    session_previous : list[LabeledHypothesis]
        Hypotheses generated earlier in this session (may overlap with labeled)
    """
    labeled_ids = {h.hypothesis_id for h in labeled}

    human_rated = [h for h in labeled if not h.effective_rating.is_pending]

    # Critic-rated session hypotheses go into the rated bucket (learn from feedback)
    critic_rated = [
        h
        for h in session_previous
        if h.hypothesis_id not in labeled_ids and not h.effective_rating.is_pending
    ]
    truly_unrated = [
        h
        for h in session_previous
        if h.hypothesis_id not in labeled_ids and h.effective_rating.is_pending
    ]

    # Human-rated first (higher signal), then critic-rated
    rated = human_rated + critic_rated
    unrated_session = truly_unrated

    rated_ids = [h.hypothesis_id for h in rated[:MAX_RATED]]
    unrated_session_ids = [h.hypothesis_id for h in unrated_session[:MAX_SESSION]]

    if not rated and not unrated_session:
        return PreviousHypothesesContext(
            text="**No previous hypotheses yet.**",
        )

    lines: list[str] = []
    idx = 1

    if rated:
        lines.append("**PREVIOUSLY RATED HYPOTHESES (learn from this feedback):**")
        for hyp in rated[:MAX_RATED]:
            sid = hyp.hypothesis_id
            desc = (
                f"{hyp.proposal.cohort_a_description} vs "
                f"{hyp.proposal.cohort_b_description}"
            )
            tag = _format_rating_tag(hyp)
            refines = _format_refines_tag(hyp)
            summary = _format_summary_suffix(hyp)
            src = _source_label(hyp)
            entry = f"{idx}. [{sid}] {desc} {tag} [{src}]{refines}"
            notes = hyp.human_notes or hyp.notes
            if notes:
                entry += f" — {notes}"
            entry += summary
            lines.append(entry)
            idx += 1
        if len(rated) > MAX_RATED:
            lines.append(f"... and {len(rated) - MAX_RATED} more rated hypotheses")

    if unrated_session:
        lines.append("**PREVIOUSLY TESTED (avoid duplicates):**")
        for hyp in unrated_session[:MAX_SESSION]:
            sid = hyp.hypothesis_id
            desc = (
                f"{hyp.proposal.cohort_a_description} vs "
                f"{hyp.proposal.cohort_b_description}"
            )
            refines = _format_refines_tag(hyp)
            summary = _format_summary_suffix(hyp)
            entry = f"{idx}. [{sid}] {desc}"
            if not hyp.effective_rating.is_pending:
                tag = _format_rating_tag(hyp)
                src = _source_label(hyp)
                entry += f" {tag} [{src}]"
                notes = hyp.human_notes or hyp.notes
                if notes:
                    entry += f" — {notes}"
            entry += refines + summary
            lines.append(entry)
            idx += 1
        if len(unrated_session) > MAX_SESSION:
            lines.append(f"... and {len(unrated_session) - MAX_SESSION} more")

    return PreviousHypothesesContext(
        text="\n".join(lines),
        rated_ids=rated_ids,
        unrated_session_ids=unrated_session_ids,
    )
