"""Code-first hypothesis generation.

Each hypothesis is a Python script run in the Docker sandbox, which reports a
:class:`~geryon.sandbox.result.IterationResult`. Refining a hypothesis means fetching
its script and editing it.
"""

from geryon.codeflow.models import CodeCritique, CodeHypothesis, CodeNarrative
from geryon.codeflow.store import CodeHypothesisStore

__all__ = [
    "CodeHypothesis",
    "CodeNarrative",
    "CodeCritique",
    "CodeHypothesisStore",
]
