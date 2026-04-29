"""Tests for the single-batch overfit check."""

from __future__ import annotations

import torch
from torch import nn

from aero_eval import check_single_batch_overfit


def _classification_batch(n: int = 8, d: int = 16, n_classes: int = 4):
    x = torch.randn(n, d)
    y = torch.randint(0, n_classes, (n,))
    return x, y


def test_healthy_classifier_overfits():
    """A reasonably-sized MLP should overfit a tiny batch easily."""
    model = nn.Sequential(nn.Linear(16, 64), nn.GELU(), nn.Linear(64, 4))
    batch = _classification_batch()
    loss_fn = nn.CrossEntropyLoss()

    result = check_single_batch_overfit(model, batch, loss_fn)
    assert result.passed
    assert result.evidence["final_loss"] < result.evidence["initial_loss"]


def test_frozen_model_fails_overfit():
    """A model whose parameters are all frozen cannot reduce loss."""
    model = nn.Sequential(nn.Linear(16, 64), nn.GELU(), nn.Linear(64, 4))
    for p in model.parameters():
        p.requires_grad_(False)
    batch = _classification_batch()
    loss_fn = nn.CrossEntropyLoss()

    result = check_single_batch_overfit(model, batch, loss_fn, max_steps=20)
    assert not result.passed
    # Loss should be essentially flat.
    assert abs(result.evidence["final_loss"] - result.evidence["initial_loss"]) < 1e-5


def test_loss_history_recorded():
    model = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 4))
    batch = _classification_batch()
    loss_fn = nn.CrossEntropyLoss()

    result = check_single_batch_overfit(model, batch, loss_fn, max_steps=10)
    assert len(result.evidence["loss_history"]) == 10


def test_threshold_tuning_relaxes_pass():
    """A weak model that drops loss only 50% should fail at default 90% but
    pass when threshold is loosened."""
    model = nn.Linear(16, 4)  # Underpowered for the task.
    batch = _classification_batch()
    loss_fn = nn.CrossEntropyLoss()

    strict = check_single_batch_overfit(
        model, batch, loss_fn, max_steps=10, relative_reduction=0.99
    )
    # Reset model state to ensure fair comparison.
    model = nn.Linear(16, 4)
    relaxed = check_single_batch_overfit(
        model, batch, loss_fn, max_steps=200, relative_reduction=0.50
    )
    # Linear model with enough steps should hit 50% reduction.
    assert relaxed.passed
    # Strict (99% in 10 steps) should fail.
    assert not strict.passed


def test_custom_forward_fn_for_hf_style():
    """forward_fn override allows HuggingFace-style batches that return loss."""

    class TinyHfStyle(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(16, 4)

        def forward(self, input_ids=None, labels=None):
            logits = self.fc(input_ids)
            loss = nn.functional.cross_entropy(logits, labels)
            # Return scalar loss directly.
            return loss

    model = TinyHfStyle()
    x, y = _classification_batch()
    batch = {"input_ids": x, "labels": y}

    def fwd(m, b):
        return m(**b), b["labels"]

    result = check_single_batch_overfit(model, batch, loss_fn=None, forward_fn=fwd)
    assert result.passed
