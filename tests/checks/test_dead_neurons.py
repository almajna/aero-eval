"""Tests for the dead-neuron / activation-variance check."""

from __future__ import annotations

import torch
from torch import nn

from aero_eval import check_dead_neurons


def _healthy_mlp() -> nn.Module:
    """MLP with weights initialized to produce non-degenerate activations."""
    return nn.Sequential(
        nn.Linear(16, 32),
        nn.GELU(),
        nn.Linear(32, 16),
    )


class _ClampedModel(nn.Module):
    """Adversarial model: clamps every Linear output to a constant.

    Simulates an LLM Goodharting around gradient-stability checks.
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 16)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.clamp(x, min=0.5, max=0.5)  # all variance destroyed
        x = self.fc2(x)
        return x


class _DeadReLUModel(nn.Module):
    """Model where most ReLU pre-activations are negative => dead post-activation."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        # Force a negative bias so ReLU kills nearly everything.
        with torch.no_grad():
            self.fc1.bias.fill_(-100.0)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def test_healthy_mlp_passes():
    model = _healthy_mlp().eval()
    x = torch.randn(8, 16)
    result = check_dead_neurons(model, x)
    assert result.passed
    assert result.score > 0.5


def test_clamped_model_caught():
    model = _ClampedModel().eval()
    x = torch.randn(8, 16)
    result = check_dead_neurons(model, x)
    assert not result.passed
    # fc1 is the clamped layer; all its post-clamp variance is zero, but the
    # check looks at *layer outputs*, which for fc1 is pre-clamp. The clamp
    # affects fc2's *input*. So fc2's output should still have variance —
    # but its *input* has zero variance, meaning fc2's output is just the
    # bias, which has zero per-channel variance across the batch.
    assert "fc2" in result.evidence["failing_layers"]


def test_dead_relu_post_activation_caught():
    """Dead ReLU is upstream of fc2; fc2's input is all-zero, so fc2 output
    becomes just bias, which has zero across-batch variance per channel."""
    model = _DeadReLUModel().eval()
    x = torch.randn(8, 16)
    result = check_dead_neurons(model, x)
    assert not result.passed


def test_excluded_layer_not_flagged():
    model = _ClampedModel().eval()
    x = torch.randn(8, 16)
    result = check_dead_neurons(model, x, excluded_layer_names={"fc2"})
    # With fc2 excluded, only fc1 is instrumented; fc1 sees normal random input
    # and produces normal variance.
    assert result.passed
