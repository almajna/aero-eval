"""Tests for the NaN/Inf check."""

from __future__ import annotations

import torch
from torch import nn

from aero_eval import check_nan_inf


def _toy_model() -> nn.Module:
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))


def test_clean_model_passes():
    model = _toy_model()
    result = check_nan_inf(model)
    assert result.passed
    assert result.score == 1.0
    assert result.evidence["offenders"] == []


def test_nan_in_param_caught():
    model = _toy_model()
    with torch.no_grad():
        model[0].weight[0, 0] = float("nan")
    result = check_nan_inf(model)
    assert not result.passed
    assert result.score == 0.0
    assert any(o["kind"] == "param_nan" for o in result.evidence["offenders"])


def test_inf_in_param_caught():
    model = _toy_model()
    with torch.no_grad():
        model[2].bias[0] = float("inf")
    result = check_nan_inf(model)
    assert not result.passed
    kinds = {o["kind"] for o in result.evidence["offenders"]}
    assert "param_inf" in kinds


def test_nan_in_grad_caught():
    model = _toy_model()
    x = torch.randn(2, 8)
    y = model(x).sum()
    y.backward()
    # Inject NaN into a grad after backward.
    model[0].weight.grad[0, 0] = float("nan")
    result = check_nan_inf(model)
    assert not result.passed
    assert any(o["kind"] == "grad_nan" for o in result.evidence["offenders"])


def test_skip_grads_when_disabled():
    model = _toy_model()
    x = torch.randn(2, 8)
    y = model(x).sum()
    y.backward()
    model[0].weight.grad[0, 0] = float("nan")
    result = check_nan_inf(model, check_grads=False)
    assert result.passed  # grad NaN ignored


def test_first_offender_in_message():
    model = _toy_model()
    with torch.no_grad():
        model[0].weight[0, 0] = float("nan")
    result = check_nan_inf(model)
    assert "0.weight" in result.message
