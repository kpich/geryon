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
    return f"[{hyp.short_id()}] {body}{stat}{refines}{_format_verdict(hyp)}"


_HOLDS_UP_LABEL = {True: "held", False: "did NOT hold", None: "untested"}


def _format_verdict(hyp: CodeHypothesis) -> str:
    c = hyp.critique
    if c is None:
        return ""
    tag = (
        f"critic T{c.trustworthiness}/C{c.confound_risk}/N{c.novelty}, "
        f"{_HOLDS_UP_LABEL[c.holds_up]}"
    )
    if c.headline:
        tag += f": {c.headline}"
    return f" {{{tag}}}"


def format_previous_hypotheses(hypotheses: list[CodeHypothesis]) -> PreviousContext:
    """Render prior hypotheses as a compact newest-first list, capped.

    Only the list: the heading and the key to the critic tags are prose in the
    prompt set's ``generator_user.md``.
    """
    if not hypotheses:
        return PreviousContext(text="(none yet)")

    ordered = list(reversed(hypotheses))  # JSONL is oldest-first
    included = ordered[:MAX_CONTEXT_HYPOTHESES]
    overflow = len(ordered) - len(included)

    lines = []
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

    Skips JSONL files whose header isn't codeflow format. A codeflow session that
    fails to load raises, so it can't silently drop out of the prompt context. With
    ``chain`` set, returns only that chain's sessions; ``None`` loads every chain.
    Sessions written before chains existed have no ``chain`` in their header and
    count as :data:`~geryon.codeflow.chains.DEFAULT_CHAIN`.
    """
    if output_dir is None or not output_dir.exists():
        return []
    out: list[CodeHypothesis] = []
    for jsonl_file in sorted(output_dir.rglob(HYPOTHESES_FILENAME)):
        header = _read_header(jsonl_file)
        if not isinstance(header, dict) or header.get("format") != "codeflow":
            continue
        if chain is not None and header.get("chain", DEFAULT_CHAIN) != chain:
            continue
        hyps = CodeHypothesisStore(jsonl_file.parent).load()
        out.extend(h for h in hyps if h.session_id != current_session_id)
    return out


def _read_header(path: Path) -> object:
    with open(path) as f:
        return json.loads(f.readline())
