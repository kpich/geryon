"""Host-side driver: run an LLM-authored script in the Docker sandbox.

The container only ever sees the data dir (read-only) and a fresh scratch dir, with
networking disabled and resource limits applied. Arbitrary host files are simply
not present in the container, so they cannot be read.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import uuid

from geryon.sandbox.result import IterationResult, ScriptRun

DEFAULT_IMAGE = "geryon-sandbox"
DEFAULT_TIMEOUT_SECONDS = 180


class SandboxError(RuntimeError):
    """Host-side sandbox problem (docker missing, image not built, ...)."""


@dataclass(frozen=True)
class SandboxLimits:
    """Resource limits passed to ``docker run``."""

    memory: str = "2g"
    cpus: str = "2"
    pids: int = 256


def docker_available() -> bool:
    """True if the docker CLI is on PATH and the daemon answers."""
    if shutil.which("docker") is None:
        return False
    proc = subprocess.run(
        ["docker", "info"],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0


def image_exists(image: str = DEFAULT_IMAGE) -> bool:
    """True if the sandbox image has been built locally."""
    proc = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0


def ensure_sandbox(image: str = DEFAULT_IMAGE) -> None:
    """Raise SandboxError unless docker + the built image are usable.

    Call this once at session start to fail fast before spending LLM calls.
    """
    _preflight(image)


def _preflight(image: str) -> None:
    if shutil.which("docker") is None:
        raise SandboxError(
            "docker CLI not found on PATH. Install Docker Desktop and ensure "
            "`docker` is available."
        )
    if not docker_available():
        raise SandboxError(
            "Docker daemon is not responding. Is Docker Desktop running?"
        )
    if not image_exists(image):
        raise SandboxError(
            f"Sandbox image '{image}' not found. Build it with:\n"
            f"    docker build -t {image} src/geryon/sandbox"
        )


def run_script(
    code: str,
    data_dir: str | Path,
    *,
    image: str = DEFAULT_IMAGE,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    limits: SandboxLimits | None = None,
) -> ScriptRun:
    """Run ``code`` in the sandbox and return its outcome.

    Parameters
    ----------
    code:
        The Python source to execute. It may ``from geryon_runtime import db, report``.
    data_dir:
        Host directory of parquet files; mounted read-only at ``/data``.
    image:
        Docker image tag (default ``geryon-sandbox``).
    timeout:
        Wall-clock seconds before the container is force-killed.
    limits:
        Resource limits; defaults to :class:`SandboxLimits`.

    Raises
    ------
    SandboxError
        If docker / the image is unavailable (an environment problem, distinct
        from the script itself failing).
    """
    limits = limits or SandboxLimits()
    data_path = Path(data_dir).expanduser().resolve()
    if not data_path.is_dir():
        raise SandboxError(f"data_dir is not a directory: {data_path}")

    _preflight(image)

    container_name = f"geryon-run-{uuid.uuid4().hex[:12]}"

    with tempfile.TemporaryDirectory(prefix="geryon-scratch-") as scratch:
        scratch_path = Path(scratch)
        (scratch_path / "script.py").write_text(code)

        cmd = [
            "docker",
            "run",
            "--rm",
            "--name",
            container_name,
            "--network",
            "none",
            "--security-opt",
            "no-new-privileges",
            "--memory",
            limits.memory,
            "--cpus",
            limits.cpus,
            "--pids-limit",
            str(limits.pids),
            "-v",
            f"{data_path}:/data:ro",
            "-v",
            f"{scratch_path}:/scratch",
            image,
            "python",
            "/scratch/script.py",
        ]

        start = time.monotonic()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            _force_remove(container_name)
            return ScriptRun(
                success=False,
                timed_out=True,
                stdout=_as_text(exc.stdout),
                stderr=_as_text(exc.stderr),
                duration_seconds=time.monotonic() - start,
                error=f"Script exceeded {timeout}s timeout",
            )

        duration = time.monotonic() - start
        result, parse_error = _read_result(scratch_path / "result.json")

    return ScriptRun(
        success=proc.returncode == 0,
        exit_code=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        duration_seconds=duration,
        result=result,
        error=parse_error,
    )


def _read_result(path: Path) -> tuple[IterationResult | None, str | None]:
    """Parse result.json if the script reported one."""
    if not path.exists():
        return None, None
    try:
        data = json.loads(path.read_text())
        return IterationResult.model_validate(data), None
    except (json.JSONDecodeError, ValueError) as exc:
        return None, f"Could not parse reported result: {exc}"


def _force_remove(container_name: str) -> None:
    subprocess.run(
        ["docker", "rm", "-f", container_name],
        capture_output=True,
        text=True,
    )


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value
