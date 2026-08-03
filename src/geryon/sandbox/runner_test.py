"""Tests for the sandbox runner.

These do not invoke Docker — they exercise preflight failures, result parsing, and
helpers. End-to-end execution against a real container is covered by the manual
verification in the plan (and an integration test once the image is built).
"""

from pathlib import Path

import pytest

from geryon.sandbox import runner
from geryon.sandbox.result import IterationResult
from geryon.sandbox.runner import SandboxError, SandboxLimits, run_script


def test_limits_defaults():
    limits = SandboxLimits()
    assert limits.memory == "2g"
    assert limits.cpus == "2"
    assert limits.pids == 256


def test_run_script_rejects_nonexistent_data_dir(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(runner.shutil, "which", lambda _: "/usr/bin/docker")
    with pytest.raises(SandboxError, match="not a directory"):
        run_script("print(1)", tmp_path / "missing")


def test_run_script_errors_when_docker_missing(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(runner.shutil, "which", lambda _: None)
    with pytest.raises(SandboxError, match="docker CLI not found"):
        run_script("print(1)", tmp_path)


def test_run_script_errors_when_image_missing(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(runner.shutil, "which", lambda _: "/usr/bin/docker")
    monkeypatch.setattr(runner, "docker_available", lambda: True)
    monkeypatch.setattr(runner, "image_exists", lambda *a, **k: False)
    with pytest.raises(SandboxError, match="not found"):
        run_script("print(1)", tmp_path)


@pytest.mark.parametrize(
    "image,expected",
    [
        ("geryon-sandbox", "geryon-sandbox:latest"),
        ("geryon-sandbox:v1", "geryon-sandbox:v1"),
        ("localhost:5000/geryon-sandbox", "localhost:5000/geryon-sandbox:latest"),
        ("registry.io/team/img:dev", "registry.io/team/img:dev"),
        ("geryon-sandbox@sha256:abc", "geryon-sandbox@sha256:abc"),
    ],
)
def test_with_default_tag(image: str, expected: str):
    assert runner._with_default_tag(image) == expected


def test_image_exists_inspects_tagged_reference(monkeypatch):
    seen: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        seen.append(cmd)
        return type("Proc", (), {"returncode": 0})()

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    assert runner.image_exists("geryon-sandbox")
    assert seen == [["docker", "image", "inspect", "geryon-sandbox:latest"]]


def test_read_result_missing_file(tmp_path: Path):
    result, error = runner._read_result(tmp_path / "result.json")
    assert result is None
    assert error is None


def test_read_result_valid(tmp_path: Path):
    path = tmp_path / "result.json"
    path.write_text('{"effect_size": 1.8, "p_value": 0.01, "extra": {"fdr": 0.02}}')
    result, error = runner._read_result(path)
    assert error is None
    assert isinstance(result, IterationResult)
    assert result.effect_size == 1.8
    assert result.extra == {"fdr": 0.02}


def test_read_result_malformed(tmp_path: Path):
    path = tmp_path / "result.json"
    path.write_text("{not json")
    result, error = runner._read_result(path)
    assert result is None
    assert error is not None
    assert "parse" in error.lower()


def test_as_text_handles_bytes_and_none():
    assert runner._as_text(None) == ""
    assert runner._as_text(b"hi") == "hi"
    assert runner._as_text("hi") == "hi"
