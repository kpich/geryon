"""Standardized machine-readable result for a single iteration.

``IterationResult`` is what a script reports via the in-container ``report()``
helper (written to ``/scratch/result.json``). ``ScriptRun`` is what the host-side
runner returns: the parsed result plus execution metadata (stdout, exit code, ...).

Reporting is optional and partial — "no clean effect size" is a first-class case,
so every analytic field is nullable. The ``extra`` dict is the extensibility hatch
for method-specific numbers (mirrors the legacy ``ComparisonResult.extra_stats``).
"""

from pydantic import BaseModel, Field

# Keys written by report() into result.json. Kept in sync with runtime.py, which
# is standalone (not importable from this package) inside the container.
RESULT_FIELDS = (
    "effect_size",
    "effect_size_type",
    "p_value",
    "ci_lower",
    "ci_upper",
    "n_a",
    "n_b",
    "summary",
    "extra",
)


class IterationResult(BaseModel):
    """The analytic result a script chose to report."""

    effect_size: float | None = Field(
        default=None, description="Point estimate, e.g. a hazard ratio or mean diff"
    )
    effect_size_type: str | None = Field(
        default=None,
        description="What the effect size is, e.g. 'hazard_ratio', 'odds_ratio'",
    )
    p_value: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    n_a: int | None = Field(default=None, description="Sample size of group A")
    n_b: int | None = Field(default=None, description="Sample size of group B")
    summary: str | None = Field(
        default=None, description="One-line plain-language result the script reported"
    )
    extra: dict[str, float] = Field(default_factory=dict)


class ScriptRun(BaseModel):
    """Outcome of running one script in the sandbox."""

    success: bool = Field(..., description="Script exited 0 and did not time out")
    exit_code: int | None = None
    timed_out: bool = False
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = 0.0

    # Parsed from /scratch/result.json if the script called report(); else None.
    result: IterationResult | None = None

    # Host-side failure (docker unavailable, result.json malformed, ...). Distinct
    # from a non-zero exit, which is captured via exit_code/stderr.
    error: str | None = None
