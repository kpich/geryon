"""LLM critic — auto-rates hypotheses between iterations."""

from datetime import UTC, datetime
import json

from pydantic import BaseModel

from geryon.legacy.labeling.labels import RATING_DIMENSIONS, HypothesisRating
from geryon.legacy.labeling.models import LabeledHypothesis
from geryon.llm.conversation_logger import SessionTracer
from geryon.llm.providers.base import ChatMessage, LLMProvider


class CriticRating(BaseModel):
    """Rating for a single hypothesis from the critic."""

    hypothesis_id: str
    novelty: int
    uncontrolled: int
    trustworthiness: int
    is_duplicate: bool = False
    notes: str = ""


class CriticResponse(BaseModel):
    """Batch response from the critic."""

    ratings: list[CriticRating]


def _build_dimensions_description() -> str:
    lines: list[str] = []
    for key, dim in RATING_DIMENSIONS.items():
        lines.append(f"**{dim['label']}** (`{key}`, 1-3):")
        for level, desc in dim["levels"].items():
            lines.append(f"  {level} = {desc}")
    return "\n".join(lines)


CRITIC_SYSTEM_PROMPT = f"""You are an expert oncologist and biostatistician reviewing
cancer genomics hypotheses. Rate each hypothesis on the following dimensions:

{_build_dimensions_description()}

**is_duplicate** (bool): True if this hypothesis is essentially the same comparison
as another hypothesis in the batch or is trivially redundant.

CALIBRATION:
- For uncontrolled and trustworthiness, most hypotheses should receive 2s; reserve
  1s for clearly poor results and 3s for genuinely impressive/concerning ones.
- NOVELTY is judged on STRUCTURE, not effect size. A "volcano-cell" hypothesis — a
  single gene-mut-vs-WT contrast inside a (cancer type ∩ drug) slice — gets
  novelty=1 EVEN WITH A STRONG HR OR TINY p-VALUE, because a 1-D scan already
  enumerates every such cell; the LLM added nothing. Do not let good statistics pull
  a volcano-cell's novelty above 1.
- novelty=2 requires at least one structural dimension a 1-D scan cannot produce
  (co-mutation/epistasis, derived/temporal cohort, pathway burden, or
  clinical-subgroup × marker). novelty=3 is reserved for structurally rich AND
  biologically surprising contrasts.
- Think like a reviewer: could this exact comparison have fallen out of an automated
  one-column scan? If yes, it is not novel.
- A hypothesis with tiny cohorts or extreme HR values gets trustworthiness=1.

ACTIONABLE NOTES:
When uncontrolled >= 2, end your notes with a single "Suggested fix:" line
giving the most specific actionable filter to add to both cohorts. Examples:
  "Suggested fix: restrict both cohorts to STAGE_HIGHEST_RECORDED = 'Stage 4'"
  "Suggested fix: restrict both cohorts to timeline_treatment.SUBTYPE = 'Immuno'"
  "Suggested fix: add MSI_TYPE filter to isolate MSS patients in both cohorts"

Return ONLY valid JSON matching this schema:
```json
{{
  "ratings": [
    {{
      "hypothesis_id": "...",
      "novelty": 2,
      "uncontrolled": 2,
      "trustworthiness": 2,
      "is_duplicate": false,
      "notes": "brief explanation"
    }}
  ]
}}
```
"""


class HypothesisCritic:
    """Batch-rates hypotheses using an LLM provider."""

    def __init__(self, provider: LLMProvider, logger: SessionTracer | None = None):
        self.provider = provider
        self.logger = logger

    def rate(self, hypotheses: list[LabeledHypothesis]) -> list[LabeledHypothesis]:
        """Rate a batch of hypotheses. Mutates and returns the same list."""
        if not hypotheses:
            return hypotheses

        user_prompt = self._build_user_prompt(hypotheses)
        messages = [
            ChatMessage(role="system", content=CRITIC_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_prompt),
        ]

        response = self.provider.generate(messages, temperature=0.2, cache_system=True)
        ratings = self._parse_response(response.content, hypotheses)

        rating_by_id = {r.hypothesis_id: r for r in ratings}
        now = datetime.now(UTC)

        for hyp in hypotheses:
            cr = rating_by_id.get(hyp.hypothesis_id)
            if cr is None:
                continue
            hyp.rating = HypothesisRating(
                novelty=cr.novelty,
                uncontrolled=cr.uncontrolled,
                trustworthiness=cr.trustworthiness,
                is_duplicate=cr.is_duplicate or None,
            )
            hyp.labeled_by = "llm_critic"
            hyp.labeled_at = now
            hyp.notes = cr.notes or None

        if self.logger:
            usage_total = (response.usage or {}).get("total_tokens")
            self.logger.log_critic(
                count=len(hypotheses),
                tokens=usage_total,
                ratings=[r.model_dump() for r in ratings],
                raw_response=response.content,
            )

        return hypotheses

    def _build_user_prompt(self, hypotheses: list[LabeledHypothesis]) -> str:
        sections: list[str] = []
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

            section = (
                f"## Hypothesis {hyp.hypothesis_id}\n"
                f"**Cohort A**: {hyp.proposal.cohort_a_description}\n"
                f"**Cohort B**: {hyp.proposal.cohort_b_description}\n"
                f"**Rationale**: {hyp.proposal.rationale}\n"
                f"**Stats**: {stats}\n"
                f"**Summary**: {hyp.narrative.summary}\n"
                f"**Findings**: {hyp.narrative.findings}\n"
                f"**Limitations**: {', '.join(hyp.narrative.limitations)}"
            )
            sections.append(section)

        return (
            "Rate each hypothesis below. Return JSON with one rating per "
            "hypothesis.\n\n" + "\n\n".join(sections)
        )

    def _parse_response(
        self, content: str, hypotheses: list[LabeledHypothesis]
    ) -> list[CriticRating]:
        """Parse critic JSON response, falling back to neutral defaults."""
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
            resp = CriticResponse(**data)
            return resp.ratings
        except Exception:
            return [
                CriticRating(
                    hypothesis_id=h.hypothesis_id,
                    novelty=2,
                    uncontrolled=2,
                    trustworthiness=2,
                    is_duplicate=False,
                    notes="critic parse failure — neutral defaults applied",
                )
                for h in hypotheses
            ]
