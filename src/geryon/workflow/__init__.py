"""Workflow orchestration for hypothesis generation."""

from geryon.workflow.session import Session, SessionConfig

__all__ = ["SessionConfig", "Session", "AutonomousWorkflow"]


def __getattr__(name: str):
    if name == "AutonomousWorkflow":
        from geryon.workflow.autonomous import AutonomousWorkflow

        return AutonomousWorkflow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
