"""Single-batch overfit check — the canonical PyTorch sanity test.

A model that cannot drive loss to near-zero on a single batch in a few dozen
optimizer steps is structurally broken. This is Karpathy's recipe and remains
the cheapest, highest-information check available.

The threshold is **loss-function-aware**: cross-entropy on a 50k-vocab problem
starts near ``ln(50000) ~ 10.8``, while MSE on normalized targets starts near
``1.0``. A single absolute cutoff would be wrong for one or the other. We
instead require a relative reduction from the *initial* loss, plus a fallback
absolute floor expressed as a fraction of that initial loss.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from aero_eval.result import CheckResult

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def check_single_batch_overfit(
    model: nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor] | dict,
    loss_fn: LossFn | None = None,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    max_steps: int = 200,
    relative_reduction: float = 0.90,
    absolute_floor_ratio: float = 0.05,
    forward_fn: Callable[[nn.Module, object], tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> CheckResult:
    """Train ``model`` on a single batch and assert loss collapses.

    Pass criterion (either is sufficient):

    1. Final loss <= ``(1 - relative_reduction) * initial_loss`` — i.e. a 90%
       reduction by default.
    2. Final loss <= ``absolute_floor_ratio * initial_loss`` — i.e. loss
       falls to under 5% of where it started. Effectively the same as (1)
       at default values; kept separate so callers can tune them independently
       (e.g. for MSE on noisy targets you may relax (1) but keep (2) tight).

    Args:
        model: Module to train. Will be mutated (parameter updates applied).
            Caller is responsible for cloning if non-destructive checking is
            required.
        batch: Either a ``(inputs, targets)`` tuple or a dict for HF-style
            models. Custom unpacking via ``forward_fn``.
        loss_fn: Loss callable ``(logits, targets) -> scalar``. Required
            unless ``forward_fn`` is supplied and computes loss internally.
        optimizer: Optimizer. Defaults to ``Adam(lr=5e-3)`` over trainable
            params only. AdamW is intentionally avoided — its weight decay
            fights the overfit we are trying to induce.
        max_steps: Optimization steps to run. 200 is enough for most tiny
            models with the default optimizer; very large models may need
            more.
        relative_reduction: Required fractional decrease from initial loss.
            Default 0.90 = "loss must drop at least 90%".
        absolute_floor_ratio: Alternative pass condition. Final loss must be
            below this fraction of the initial loss.
        forward_fn: Override forward pass. Signature
            ``(model, batch) -> (logits_or_loss, targets)``. If the returned
            first tensor is already a scalar loss, ``loss_fn`` may be None.

    Returns:
        ``CheckResult`` with the loss trajectory in ``evidence["loss_history"]``.
    """
    trainable = [p for p in model.parameters() if p.requires_grad]
    has_trainable = len(trainable) > 0

    if optimizer is None and has_trainable:
        # Adam with a relatively high lr is best for tiny-model overfitting:
        # AdamW's weight decay actively fights overfit, which we *want* here.
        optimizer = torch.optim.Adam(trainable, lr=5e-3)

    def _default_forward(m: nn.Module, b: object) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(b, tuple) and len(b) == 2:
            inputs, targets = b
            logits = m(inputs)
            return logits, targets
        raise ValueError(
            "Default forward expects a (inputs, targets) tuple; "
            "pass forward_fn for other batch shapes."
        )

    fwd = forward_fn or _default_forward

    model.train()
    loss_history: list[float] = []
    initial_loss: float | None = None

    for step in range(max_steps):
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        out, targets = fwd(model, batch)

        # Allow forward_fn to return a pre-computed scalar loss.
        if out.ndim == 0:
            loss = out
        else:
            if loss_fn is None:
                raise ValueError(
                    "loss_fn is required unless forward_fn returns a scalar loss tensor."
                )
            loss = loss_fn(out, targets)

        loss_val = loss.item()
        loss_history.append(loss_val)
        if step == 0:
            initial_loss = loss_val

        # Skip backward/step if the model has no trainable params — the loss
        # tensor will have no grad graph and backward() would crash. We still
        # record the loss so the caller sees the flat trajectory.
        if has_trainable:
            loss.backward()
            assert optimizer is not None
            optimizer.step()

    assert initial_loss is not None
    final_loss = loss_history[-1]

    # Avoid divide-by-zero when initial_loss happens to be 0.
    if initial_loss <= 0:
        passed = final_loss <= 0
        relative_drop = 1.0 if passed else 0.0
    else:
        relative_drop = (initial_loss - final_loss) / initial_loss
        cond_relative = relative_drop >= relative_reduction
        cond_absolute = final_loss <= absolute_floor_ratio * initial_loss
        passed = bool(cond_relative or cond_absolute)

    score = max(0.0, min(1.0, relative_drop))

    msg = (
        f"loss {initial_loss:.4f} -> {final_loss:.4f} "
        f"({relative_drop:.1%} reduction in {max_steps} steps); "
        f"required {relative_reduction:.0%}"
    )

    return CheckResult(
        name="single_batch_overfit",
        passed=passed,
        score=score,
        message=msg,
        evidence={
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "relative_reduction": relative_drop,
            "loss_history": loss_history,
            "max_steps": max_steps,
        },
    )
