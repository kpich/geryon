"""Sandboxed execution of LLM-authored Python against the genomics data.

Each hypothesis is now arbitrary Python run inside a Docker container that only
sees the data (read-only) and a scratch directory. The container reports a
standardized :class:`IterationResult` via the injected ``report()`` helper.
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
