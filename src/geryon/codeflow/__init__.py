"""Code-first hypothesis generation.

The deliverable per iteration is a Python script (run in the Docker sandbox), not a
restricted formal spec. Each script reports a standardized
:class:`~geryon.sandbox.result.IterationResult`. Derivation = fetch a prior script and
remix it.
"""

from geryon.codeflow.models import CodeCritique, CodeHypothesis, CodeNarrative
from geryon.codeflow.store import CodeHypothesisStore

__all__ = [
    "CodeHypothesis",
    "CodeNarrative",
    "CodeCritique",
    "CodeHypothesisStore",
]
