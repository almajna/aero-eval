"""Tests for the CheckResult dataclass."""

from __future__ import annotations

import pytest

from aero_eval import CheckResult


def test_pass_result_truthy():
    r = CheckResult(name="x", passed=True, score=1.0, message="ok")
    assert bool(r) is True


def test_fail_result_falsy():
    r = CheckResult(name="x", passed=False, score=0.0, message="bad")
    assert bool(r) is False


def test_score_out_of_range_raises():
    with pytest.raises(ValueError):
        CheckResult(name="x", passed=True, score=1.5, message="ok")
    with pytest.raises(ValueError):
        CheckResult(name="x", passed=False, score=-0.1, message="bad")


def test_evidence_defaults_empty():
    r = CheckResult(name="x", passed=True, score=1.0, message="ok")
    assert r.evidence == {}


def test_evidence_accepts_payload():
    r = CheckResult(
        name="x",
        passed=False,
        score=0.0,
        message="bad",
        evidence={"layer": "fc1", "value": 0.0},
    )
    assert r.evidence["layer"] == "fc1"
