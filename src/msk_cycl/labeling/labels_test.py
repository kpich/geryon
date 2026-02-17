"""Tests for HypothesisRating model."""

from pydantic import ValidationError
import pytest

from msk_cycl.labeling.labels import RATING_DIMENSIONS, HypothesisRating


def test_default_is_pending():
    r = HypothesisRating()
    assert r.is_pending


def test_single_dimension_not_pending():
    r = HypothesisRating(novelty=2)
    assert not r.is_pending


def test_is_duplicate_only_not_pending():
    r = HypothesisRating(is_duplicate=True)
    assert not r.is_pending


def test_full_rating():
    r = HypothesisRating(
        novelty=1, uncontrolled=3, trustworthiness=2, is_duplicate=False
    )
    assert not r.is_pending
    assert r.novelty == 1
    assert r.uncontrolled == 3
    assert r.trustworthiness == 2
    assert r.is_duplicate is False


def test_valid_range_boundaries():
    for v in (1, 2, 3):
        r = HypothesisRating(novelty=v, uncontrolled=v, trustworthiness=v)
        assert r.novelty == v


def test_invalid_range_zero():
    with pytest.raises(ValidationError):
        HypothesisRating(novelty=0)


def test_invalid_range_four():
    with pytest.raises(ValidationError):
        HypothesisRating(trustworthiness=4)


def test_invalid_range_negative():
    with pytest.raises(ValidationError):
        HypothesisRating(uncontrolled=-1)


def test_partial_rating():
    r = HypothesisRating(novelty=2, trustworthiness=3)
    assert not r.is_pending
    assert r.novelty == 2
    assert r.uncontrolled is None
    assert r.trustworthiness == 3
    assert r.is_duplicate is None


def test_rating_dimensions_has_expected_keys():
    assert set(RATING_DIMENSIONS.keys()) == {
        "novelty",
        "uncontrolled",
        "trustworthiness",
    }
    for dim in RATING_DIMENSIONS.values():
        assert "label" in dim
        assert "levels" in dim
        assert set(dim["levels"].keys()) == {1, 2, 3}
