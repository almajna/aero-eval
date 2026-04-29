"""Tests for the param-update-magnitude check."""

from __future__ import annotations

import torch
from torch import nn

from aero_eval import check_param_update_magnitude, snapshot_params


def _toy_model() -> nn.Module:
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))


def test_normal_step_passes():
    model = _toy_model()
    snap = snapshot_params(model)
    optim = torch.optim.SGD(model.parameters(), lr=1e-2)

    x = torch.randn(4, 8)
    y = torch.randn(4, 4)
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()
    optim.step()

    result = check_param_update_magnitude(model, snap)
    assert result.passed
    assert result.score >= 0.9


def test_no_step_flags_frozen():
    model = _toy_model()
    snap = snapshot_params(model)
    # Don't take any step — every param has delta == 0.
    result = check_param_update_magnitude(model, snap)
    assert not result.passed
    assert len(result.evidence["frozen"]) == sum(1 for p in model.parameters() if p.requires_grad)


def test_explosion_caught():
    model = _toy_model()
    snap = snapshot_params(model)
    # Manually multiply weights by 100x to simulate explosion.
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(100.0)
    result = check_param_update_magnitude(model, snap)
    assert not result.passed
    assert len(result.evidence["exploding"]) > 0


def test_minor_frozen_layer_tolerated():
    """If only one layer is frozen and ratio is under threshold, still passes."""
    model = nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 4),
    )
    # Freeze first layer.
    for p in model[0].parameters():
        p.requires_grad_(False)

    snap = snapshot_params(model)
    optim = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    x = torch.randn(4, 8)
    y = torch.randn(4, 4)
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()
    optim.step()

    result = check_param_update_magnitude(model, snap, max_offender_ratio=0.20)
    # Frozen params are filtered out (requires_grad=False), so no offenders.
    assert result.passed


def test_score_grades_with_offender_ratio():
    model = _toy_model()
    snap = snapshot_params(model)
    # No optimizer step => all frozen => score should be 0.
    result = check_param_update_magnitude(model, snap, max_offender_ratio=0.0)
    assert result.score == 0.0
