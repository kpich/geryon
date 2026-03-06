"""LLM ranker — synthesizes rated pool and identifies top candidates."""

from dataclasses import dataclass, field
import json

from geryon.labeling.models import LabeledHypothesis
from geryon.llm.conversation_logger import SessionTracer
from geryon.llm.providers.base import ChatMessage, LLMProvider


@dataclass
class RankerResult:
    top_general: list[str] = field(default_factory=list)
    top_refinement: list[str] = field(default_factory=list)
    synthesis: str = ""


RANKER_SYSTEM_PROMPT = """\
You are an expert oncologist synthesizing results from a session of tested cancer \
genomics hypotheses. Each hypothesis has already been rated by a critic on three \
dimensions (novelty, uncontrolled confounders, trustworthiness, each 1-3).

Your tasks:
1. Identify the **top 10** most scientifically promising hypotheses overall — \
prioritize credible statistics (trustworthiness=3), clinically meaningful findings \
(novelty>=2), and non-trivial cohort definitions.
2. Identify the **top 2** most worth refining — high interest but confounded \
(uncontrolled >= 2); worth re-running with tighter cohorts per the \
"Suggested fix" in the notes.
3. Write a 3-5 sentence synthesis summarizing patterns across the session, promising \
directions, and notable gaps.

Return ONLY valid JSON with no prose before or after:
```json
{
  "top_general": ["<full-id>", ...],
  "top_refinement": ["<full-id>", ...],
  "synthesis": "3-5 sentence summary of patterns, promising directions, gaps"
}
```

Rules:
- `top_general`: up to 10 full hypothesis IDs, ordered by priority
- `top_refinement`: up to 2 full hypothesis IDs; only include hypotheses with \
uncontrolled >= 2
- Use the full ID from the "ID:" line in each block, not the short 8-char prefix"""


class HypothesisRanker:
    """Synthesizes a rated hypothesis pool and identifies top candidates."""

    def __init__(self, provider: LLMProvider, logger: SessionTracer | None = None):
        self.provider = provider
        self.logger = logger

    def rank(
        self,
        hypotheses: list[LabeledHypothesis],
        priority_ids: set[str] | None = None,
    ) -> RankerResult:
        """Rank hypotheses. Returns RankerResult with selected IDs and synthesis."""
        if not hypotheses:
            return RankerResult(synthesis="No hypotheses to rank.")

        user_prompt = self._build_user_prompt(hypotheses, priority_ids)
        messages = [
            ChatMessage(role="system", content=RANKER_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_prompt),
        ]

        response = self.provider.generate(messages, temperature=0.2, cache_system=True)
        result = self._parse_response(response.content, hypotheses)

        if self.logger:
            usage_total = (response.usage or {}).get("total_tokens")
            self.logger.log_ranker(
                count=len(hypotheses),
                tokens=usage_total,
                top_general=result.top_general,
                top_refinement=result.top_refinement,
                raw_response=response.content,
            )

        return result

    def _build_user_prompt(
        self,
        hypotheses: list[LabeledHypothesis],
        priority_ids: set[str] | None = None,
    ) -> str:
        priority_ids = priority_ids or set()
        priority_blocks: list[str] = []
        compact_lines: list[str] = []

        for hyp in hypotheses:
            r = hyp.result
            stats = (
                f"HR={r.hazard_ratio:.3f}" if r.hazard_ratio is not None else "HR=N/A"
            )
            if r.confidence_interval_lower is not None:
                stats += (
                    f", 95%CI=[{r.confidence_interval_lower:.3f},"
                    f" {r.confidence_interval_upper:.3f}]"
                )
            if r.p_value is not None:
                stats += f", p={r.p_value:.4f}"
            stats += f", N_A={r.cohort_a_size}, N_B={r.cohort_b_size}"

            rating = hyp.effective_rating
            rating_str = (
                f"novelty={rating.novelty}, uncontrolled={rating.uncontrolled}, "
                f"trust={rating.trustworthiness}"
            )
            if rating.is_duplicate:
                rating_str += ", is_duplicate=true"

            short_id = hyp.hypothesis_id[:8]
            notes = hyp.notes or "(none)"

            if hyp.hypothesis_id in priority_ids:
                block = (
                    f"## Hypothesis {short_id}\n"
                    f"ID: {hyp.hypothesis_id}\n"
                    f"A: {hyp.proposal.cohort_a_description} | "
                    f"B: {hyp.proposal.cohort_b_description}\n"
                    f"Stats: {stats} | Ratings: [{rating_str}]\n"
                    f"Notes: {notes} | Summary: {hyp.narrative.summary}"
                )
                priority_blocks.append(block)
            else:
                line = (
                    f"[{short_id}] {hyp.proposal.cohort_a_description} vs "
                    f"{hyp.proposal.cohort_b_description} [{rating_str}] | {notes}"
                )
                compact_lines.append(line)

        sections: list[str] = priority_blocks + compact_lines
        return (
            "Synthesize the following rated hypotheses. Return JSON.\n\n"
            + "\n\n".join(sections)
        )

    def _parse_response(
        self, content: str, hypotheses: list[LabeledHypothesis]
    ) -> RankerResult:
        """Parse ranker JSON response with ID prefix resolution and fallback."""
        prefix_to_id = {h.hypothesis_id[:8]: h.hypothesis_id for h in hypotheses}

        def resolve_ids(raw_ids: list) -> list[str]:
            return [prefix_to_id.get(rid, rid) for rid in raw_ids]

        try:
            text = content.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                start = 0
                end = len(lines)
                for i, line in enumerate(lines):
                    if line.strip().startswith("```"):
                        if start == 0:
                            start = i + 1
                        else:
                            end = i
                            break
                text = "\n".join(lines[start:end])

            data = json.loads(text)
            return RankerResult(
                top_general=resolve_ids(data.get("top_general", [])),
                top_refinement=resolve_ids(data.get("top_refinement", [])),
                synthesis=data.get("synthesis", ""),
            )
        except Exception:
            return RankerResult(
                top_general=[h.hypothesis_id for h in hypotheses[:10]],
                top_refinement=[],
                synthesis="parse failure",
            )

    def format_for_prompt(
        self, result: RankerResult, hypotheses_by_id: dict[str, LabeledHypothesis]
    ) -> str:
        """Format RankerResult as a text block for injection into the next prompt."""
        lines: list[str] = []
        lines.append("**SYNTHESIS OF CURRENT FINDINGS:**")
        lines.append(result.synthesis)
        lines.append("")

        if result.top_general:
            lines.append("**TOP CANDIDATES TO EXPLORE FURTHER:**")
            for i, hid in enumerate(result.top_general, 1):
                hyp = hypotheses_by_id.get(hid)
                if hyp is None:
                    lines.append(f"{i}. [{hid[:8]}] (not found)")
                    continue
                rating = hyp.effective_rating
                rating_str = (
                    f"novelty={rating.novelty}, "
                    f"uncontrolled={rating.uncontrolled}, "
                    f"trust={rating.trustworthiness}"
                )
                desc = (
                    f"{hyp.proposal.cohort_a_description} vs "
                    f"{hyp.proposal.cohort_b_description}"
                )
                lines.append(
                    f"{i}. [{hid[:8]}] {desc} [{rating_str}] — {hyp.narrative.summary}"
                )
            lines.append("")

        if result.top_refinement:
            lines.append(
                "**TOP CANDIDATES FOR REFINEMENT (confounded, worth fixing):**"
            )
            for i, hid in enumerate(result.top_refinement, 1):
                hyp = hypotheses_by_id.get(hid)
                if hyp is None:
                    lines.append(f"{i}. [{hid[:8]}] (not found)")
                    continue
                rating = hyp.effective_rating
                rating_str = (
                    f"novelty={rating.novelty}, "
                    f"uncontrolled={rating.uncontrolled}, "
                    f"trust={rating.trustworthiness}"
                )
                desc = (
                    f"{hyp.proposal.cohort_a_description} vs "
                    f"{hyp.proposal.cohort_b_description}"
                )
                notes = hyp.notes or ""
                lines.append(f"{i}. [{hid[:8]}] {desc} [{rating_str}] — {notes}")

        return "\n".join(lines)
