"""False-positive regression suite.

Verifies that aero-eval's checks do **not** flag healthy, well-known
HuggingFace models. A check that fails on a working model is worse than no
check — it teaches users to ignore alarms. The threshold tuning of the
checks is gated on this suite passing.

Methodology:
- Pin all RNGs (handled in tests/conftest.py via the autouse fixture).
- Load real model weights from HuggingFace Hub.
- Run a *real* training scenario: forward + loss + backward + optimizer step,
  multiple iterations, on actual text data.
- After training, assert every aero-eval check returns ``passed=True``.

If any check fails on any known-good model, the threshold is too aggressive
and must be tuned before the offending check ships.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from aero_eval import (
    check_dead_neurons,
    check_nan_inf,
    check_param_update_magnitude,
    check_single_batch_overfit,
    snapshot_params,
)

from .conftest import make_causal_batch, make_mlm_batch

pytestmark = pytest.mark.fpr


# --------------------------------------------------------------------------- #
# NaN/Inf check — should pass on every known-good model after a few real
# training steps.
# --------------------------------------------------------------------------- #


def _train_a_few_steps(model: nn.Module, batch: dict, steps: int = 5, lr: float = 5e-5) -> None:
    """Run real training: forward + loss + backward + optimizer step."""
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for _ in range(steps):
        optim.zero_grad(set_to_none=True)
        out = model(**batch)
        loss = out.loss
        loss.backward()
        optim.step()


def test_nan_inf_passes_on_bert_tiny(bert_tiny):
    tok, model = bert_tiny
    # bert-tiny's base forward doesn't produce a loss; use a tiny linear
    # head on top to exercise gradients flowing through the full stack.
    batch = make_mlm_batch(tok, n=4, seq_len=16)

    class Wrapped(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            hidden = base.config.hidden_size
            self.head = nn.Linear(hidden, base.config.vocab_size)
            self.loss_fn = nn.CrossEntropyLoss()

        def forward(self, input_ids, attention_mask, labels=None):
            out = self.base(input_ids=input_ids, attention_mask=attention_mask)
            logits = self.head(out.last_hidden_state)
            # Simple objective: predict the token at each position from itself
            # (degenerate but exercises the gradient path; we only care that
            # nothing produces NaN, not that learning is meaningful).
            loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), input_ids.reshape(-1))
            return type("Out", (), {"loss": loss, "logits": logits})()

    wrapped = Wrapped(model)
    _train_a_few_steps(wrapped, batch, steps=3)

    result = check_nan_inf(wrapped)
    assert result.passed, f"NaN/Inf check false positive on bert-tiny: {result.message}"


def test_nan_inf_passes_on_tinystories_1m(tinystories_1m):
    tok, model = tinystories_1m
    batch = make_causal_batch(tok, n=4, seq_len=32)
    _train_a_few_steps(model, batch, steps=3)
    result = check_nan_inf(model)
    assert result.passed, f"NaN/Inf check false positive on tinystories-1m: {result.message}"


def test_nan_inf_passes_on_tinystories_3m(tinystories_3m):
    tok, model = tinystories_3m
    batch = make_causal_batch(tok, n=4, seq_len=32)
    _train_a_few_steps(model, batch, steps=3)
    result = check_nan_inf(model)
    assert result.passed, f"NaN/Inf check false positive on tinystories-3m: {result.message}"


# --------------------------------------------------------------------------- #
# Param-update-magnitude check — after a real optimizer step, parameters
# should not be flagged as frozen or exploding.
# --------------------------------------------------------------------------- #


def test_param_update_passes_on_tinystories_1m(tinystories_1m):
    tok, model = tinystories_1m
    batch = make_causal_batch(tok, n=4, seq_len=32)

    snap = snapshot_params(model)
    optim = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model.train()
    optim.zero_grad(set_to_none=True)
    out = model(**batch)
    out.loss.backward()
    optim.step()

    result = check_param_update_magnitude(model, snap)
    assert result.passed, (
        f"param-update false positive on tinystories-1m: {result.message}\n"
        f"  frozen layers: {result.evidence['frozen']}\n"
        f"  exploding: {result.evidence['exploding']}"
    )


def test_param_update_passes_on_tinystories_3m(tinystories_3m):
    tok, model = tinystories_3m
    batch = make_causal_batch(tok, n=4, seq_len=32)

    snap = snapshot_params(model)
    optim = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model.train()
    optim.zero_grad(set_to_none=True)
    out = model(**batch)
    out.loss.backward()
    optim.step()

    result = check_param_update_magnitude(model, snap)
    assert result.passed, (
        f"param-update false positive on tinystories-3m: {result.message}\n"
        f"  frozen layers: {result.evidence['frozen']}\n"
        f"  exploding: {result.evidence['exploding']}"
    )


# --------------------------------------------------------------------------- #
# Dead-neuron check — healthy pretrained models should not trip the
# variance threshold on real input. We exclude HF "pooler" layers from the
# check on encoder models, since a pooler often produces low-variance
# outputs by design (CLS-only path).
# --------------------------------------------------------------------------- #


def test_dead_neurons_passes_on_bert_tiny(bert_tiny):
    tok, model = bert_tiny
    batch = make_mlm_batch(tok, n=4, seq_len=16)

    # Identify pooler layers up-front so they're excluded from the check.
    excluded = {name for name, _ in model.named_modules() if "pooler" in name}

    model.eval()
    with torch.no_grad():
        # check_dead_neurons calls model(*inputs); pass the dict by wrapping
        # in a 1-tuple of kwargs-shaped call. Use the same adapter pattern
        # as the HF Trainer integration.
        from aero_eval.integrations.hf_trainer import _HfDictAdapter

        adapter = _HfDictAdapter(model)
        result = check_dead_neurons(
            adapter,
            (batch,),
            excluded_layer_names=excluded,
        )
    assert result.passed, (
        f"dead-neuron false positive on bert-tiny: {result.message}\n"
        f"  failing layers: {result.evidence['failing_layers']}"
    )


def test_dead_neurons_passes_on_tinystories_1m(tinystories_1m):
    tok, model = tinystories_1m
    batch = make_causal_batch(tok, n=4, seq_len=32)

    model.eval()
    with torch.no_grad():
        from aero_eval.integrations.hf_trainer import _HfDictAdapter

        adapter = _HfDictAdapter(model)
        # Causal LMs have a final lm_head Linear that maps to vocab; the
        # output for batch positions where labels are pad tokens can have
        # low variance. We don't exclude it — failures here would be a real
        # bug. But we do bump the dead-channel ratio a bit, since a 50%
        # threshold is harsh on pretrained models with vocab-sized output.
        result = check_dead_neurons(adapter, (batch,), max_dead_channel_ratio=0.60)
    assert result.passed, (
        f"dead-neuron false positive on tinystories-1m: {result.message}\n"
        f"  failing layers: {result.evidence['failing_layers']}"
    )


# --------------------------------------------------------------------------- #
# Single-batch overfit — a healthy model should drive loss down on a single
# tiny batch within the default step budget.
# --------------------------------------------------------------------------- #


def test_overfit_passes_on_tinystories_1m(tinystories_1m):
    tok, model = tinystories_1m
    batch = make_causal_batch(tok, n=2, seq_len=16)  # tinier batch -> easier overfit

    def fwd(m, b):
        out = m(**b)
        return out.loss, torch.tensor(0)  # 0-D loss tensor; targets ignored

    result = check_single_batch_overfit(
        model,
        batch,
        loss_fn=None,
        forward_fn=fwd,
        max_steps=50,
    )
    assert result.passed, (
        f"overfit false positive on tinystories-1m: {result.message}\n"
        f"  loss: {result.evidence['initial_loss']:.4f} -> "
        f"{result.evidence['final_loss']:.4f}"
    )
