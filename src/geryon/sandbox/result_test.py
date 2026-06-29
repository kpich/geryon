"""Tests for the standardized result models."""

from geryon.sandbox.result import IterationResult, ScriptRun


def test_iteration_result_all_fields_optional():
    r = IterationResult()
    assert r.effect_size is None
    assert r.p_value is None
    assert r.extra == {}


def test_iteration_result_partial():
    r = IterationResult(summary="no clean effect size", n_a=120)
    assert r.summary == "no clean effect size"
    assert r.n_a == 120
    assert r.effect_size is None


def test_iteration_result_full_roundtrip():
    r = IterationResult(
        effect_size=1.8,
        effect_size_type="hazard_ratio",
        p_value=0.003,
        ci_lower=1.2,
        ci_upper=2.7,
        n_a=200,
        n_b=180,
        summary="TP53-mut worse OS",
        extra={"fdr": 0.01},
    )
    dumped = r.model_dump()
    assert IterationResult.model_validate(dumped) == r


def test_script_run_defaults():
    run = ScriptRun(success=True)
    assert run.timed_out is False
    assert run.stdout == ""
    assert run.result is None
    assert run.error is None
