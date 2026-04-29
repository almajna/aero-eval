"""Detect frozen or exploding parameter updates between training steps."""

from __future__ import annotations

import torch
from torch import nn

from aero_eval.result import CheckResult


def snapshot_params(model: nn.Module) -> dict[str, torch.Tensor]:
    """Take a detached, cloned copy of every trainable parameter.

    Call before an optimizer step; pass the result to
    :func:`check_param_update_magnitude` after the step.
    """
    return {name: param.detach().clone() for name, param in model.named_parameters()}


def check_param_update_magnitude(
    model: nn.Module,
    pre_step_snapshot: dict[str, torch.Tensor],
    *,
    frozen_threshold: float = 0.0,
    exploding_threshold: float = 10.0,
    max_offender_ratio: float = 0.20,
    eps: float = 1e-12,
) -> CheckResult:
    """Compare current parameters against a pre-step snapshot.

    Computes per-parameter relative update magnitude::

        delta = ||theta_t - theta_{t-1}|| / (||theta_{t-1}|| + eps)

    and flags parameters whose delta is exactly ``frozen_threshold`` (no
    update — dead optimizer, zero gradient, or detached layer) or above
    ``exploding_threshold`` (lr too high, gradient pathology, missing
    normalization).

    The check fails if the *fraction* of offending tensors exceeds
    ``max_offender_ratio``. A handful of frozen embeddings is acceptable;
    half the model not updating is not.

    Args:
        model: Module after the optimizer step has been applied.
        pre_step_snapshot: Mapping returned by :func:`snapshot_params`
            *before* the step.
        frozen_threshold: Treat ``delta == frozen_threshold`` as frozen.
            Default ``0.0`` — any non-zero update is healthy.
        exploding_threshold: Treat ``delta > exploding_threshold`` as
            exploding. Default ``10.0`` — a 1000% per-step update is
            unambiguously broken.
        max_offender_ratio: Max share of trainable params allowed to be in
            the frozen-or-exploding bucket before the check fails.
        eps: Numerical stabilizer for the denominator.
    """
    frozen: list[str] = []
    exploding: list[dict[str, float | str]] = []
    deltas: dict[str, float] = {}
    n_total = 0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name not in pre_step_snapshot:
            continue
        n_total += 1

        prev = pre_step_snapshot[name]
        if prev.shape != param.shape:
            # Shape changed mid-training — treat as exploding (something is wrong).
            exploding.append({"layer": name, "delta": float("inf")})
            continue

        delta_norm = torch.linalg.vector_norm(param.detach() - prev).item()
        prev_norm = torch.linalg.vector_norm(prev).item()
        delta = delta_norm / (prev_norm + eps)
        deltas[name] = delta

        if delta == frozen_threshold:
            frozen.append(name)
        elif delta > exploding_threshold:
            exploding.append({"layer": name, "delta": delta})

    n_offender = len(frozen) + len(exploding)
    ratio = n_offender / n_total if n_total else 0.0
    passed = ratio <= max_offender_ratio
    score = max(0.0, 1.0 - ratio) if n_total else 1.0

    if passed and not n_offender:
        msg = f"All {n_total} trainable parameters updated within healthy range."
    elif passed:
        msg = (
            f"{n_offender}/{n_total} ({ratio:.1%}) params anomalous "
            f"but under {max_offender_ratio:.0%} threshold."
        )
    else:
        msg = (
            f"{n_offender}/{n_total} ({ratio:.1%}) params frozen or exploding — "
            f"exceeds {max_offender_ratio:.0%} threshold."
        )

    return CheckResult(
        name="param_update_magnitude",
        passed=passed,
        score=score,
        message=msg,
        evidence={
            "frozen": frozen,
            "exploding": exploding,
            "n_total_trainable": n_total,
            "offender_ratio": ratio,
            "deltas": deltas,
        },
    )
