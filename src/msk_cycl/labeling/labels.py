"""Multi-dimensional rating model for hypothesis quality."""

from pydantic import BaseModel, field_validator

RATING_DIMENSIONS: dict[str, dict] = {
    "novelty": {
        "label": "Novelty",
        "levels": {
            1: "Known/obvious finding — good sanity check but not new knowledge",
            2: "Somewhat expected — plausible based on existing literature",
            3: "Surprising or non-obvious — potential new insight",
        },
    },
    "uncontrolled": {
        "label": "Uncontrolled",
        "levels": {
            1: "Clean comparison — cohorts are well-separated on the "
            "variable of interest",
            2: "Partially confounded — likely some mixing but main "
            "signal probably real",
            3: "Heavily confounded — effect is probably a mixture "
            "of uncontrolled populations",
        },
    },
    "trustworthiness": {
        "label": "Trustworthiness",
        "levels": {
            1: "Likely spurious — wouldn't trust this result even directionally",
            2: "Uncertain — could be real but not convincing",
            3: "Credible — result is likely real and meaningful",
        },
    },
}


class HypothesisRating(BaseModel):
    """Multi-axis rating for a hypothesis.

    Each dimension is independently optional (None = not yet rated).
    """

    novelty: int | None = None
    uncontrolled: int | None = None
    trustworthiness: int | None = None
    is_duplicate: bool | None = None
    is_na: bool | None = None

    @field_validator("novelty", "uncontrolled", "trustworthiness")
    @classmethod
    def _check_range(cls, v: int | None) -> int | None:
        if v is not None and v not in (1, 2, 3):
            raise ValueError("Rating must be 1, 2, or 3")
        return v

    @property
    def is_pending(self) -> bool:
        return all(
            v is None
            for v in (
                self.novelty,
                self.uncontrolled,
                self.trustworthiness,
                self.is_duplicate,
                self.is_na,
            )
        )
