"""Sandboxed execution of LLM-authored Python against the genomics data.

Each hypothesis is arbitrary Python run in a Docker container that sees only the
data (read-only) and a scratch directory. The script reports an
:class:`IterationResult` through ``geryon_runtime.report()``.
"""

from geryon.sandbox.result import IterationResult, ScriptRun
from geryon.sandbox.runner import (
    DEFAULT_IMAGE,
    SandboxError,
    SandboxLimits,
    ensure_sandbox,
    run_script,
)

__all__ = [
    "IterationResult",
    "ScriptRun",
    "SandboxError",
    "SandboxLimits",
    "DEFAULT_IMAGE",
    "ensure_sandbox",
    "run_script",
]
