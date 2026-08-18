"""Tests for the code hypothesis JSONL store."""

from pathlib import Path

from geryon.codeflow.models import CodeHypothesis, CodeNarrative
from geryon.codeflow.store import CodeHypothesisStore
from geryon.sandbox.result import IterationResult


def _hyp(hid: str, session: str = "s1", **kw) -> CodeHypothesis:
    defaults = {
        "hypothesis_id": hid,
        "session_id": session,
        "title": "t",
        "description": "d",
        "rationale": "r",
        "code": "print(1)",
        "success": True,
    }
    defaults.update(kw)
    return CodeHypothesis(**defaults)  # type: ignore[arg-type]


def test_save_and_load_roundtrip(tmp_path: Path):
    store = CodeHypothesisStore(tmp_path)
    h = _hyp(
        "abc",
        result=IterationResult(effect_size=1.8, p_value=0.01),
        narrative=CodeNarrative(summary="s", findings="f", context_summary="cs"),
    )
    store.save(h)

    loaded = store.load()
    assert len(loaded) == 1
    assert loaded[0].hypothesis_id == "abc"
    assert loaded[0].result is not None
    assert loaded[0].result.effect_size == 1.8
    assert loaded[0].narrative is not None
    assert loaded[0].narrative.context_summary == "cs"


def test_metadata_header_marks_codeflow_format(tmp_path: Path):
    store = CodeHypothesisStore(tmp_path)
    store.save(_hyp("a"))
    first_line = (tmp_path / "hypotheses.jsonl").read_text().splitlines()[0]
    assert '"record_type":"metadata"' in first_line.replace(" ", "")
    assert '"format":"codeflow"' in first_line.replace(" ", "")


def test_load_empty(tmp_path: Path):
    assert CodeHypothesisStore(tmp_path).load() == []


def test_save_all_rewrites(tmp_path: Path):
    store = CodeHypothesisStore(tmp_path)
    store.save(_hyp("a"))
    store.save(_hyp("b"))
    store.save_all([_hyp("a", title="updated")])

    loaded = store.load()
    assert len(loaded) == 1
    assert loaded[0].title == "updated"


def test_header_records_the_chain(tmp_path: Path):
    store = CodeHypothesisStore(tmp_path, chain="medonc-pfs")
    store.save(_hyp("a"))
    first_line = (tmp_path / "hypotheses.jsonl").read_text().splitlines()[0]
    assert '"chain":"medonc-pfs"' in first_line.replace(" ", "")


def test_header_defaults_to_main(tmp_path: Path):
    CodeHypothesisStore(tmp_path).save(_hyp("a"))
    first_line = (tmp_path / "hypotheses.jsonl").read_text().splitlines()[0]
    assert '"chain":"main"' in first_line.replace(" ", "")


def test_save_all_preserves_the_chain(tmp_path: Path):
    """The critic rewrites the file; the header must not silently revert to main."""
    store = CodeHypothesisStore(tmp_path, chain="medonc-pfs")
    store.save(_hyp("a"))
    store.save_all([_hyp("a", title="critiqued")])

    first_line = (tmp_path / "hypotheses.jsonl").read_text().splitlines()[0]
    assert '"chain":"medonc-pfs"' in first_line.replace(" ", "")


def test_hypothesis_carries_chain_and_data_version(tmp_path: Path):
    store = CodeHypothesisStore(tmp_path, chain="medonc-pfs")
    store.save(_hyp("a", chain="medonc-pfs", data_version="medonc-pfs-2026-08"))

    loaded = store.load()[0]
    assert loaded.chain == "medonc-pfs"
    assert loaded.data_version == "medonc-pfs-2026-08"


def test_result_none_roundtrips(tmp_path: Path):
    store = CodeHypothesisStore(tmp_path)
    store.save(_hyp("a", result=None, success=False, stderr="boom"))
    loaded = store.load()
    assert loaded[0].result is None
    assert loaded[0].success is False
    assert loaded[0].stderr == "boom"
