"""Detect NaN/Inf contamination in parameters or gradients."""

from __future__ import annotations

import torch
from torch import nn

from aero_eval.result import CheckResult


def check_nan_inf(
    model: nn.Module,
    *,
    check_grads: bool = True,
    check_params: bool = True,
) -> CheckResult:
    """Scan model parameters and gradients for NaN or Inf values.

    Standard early-warning sentinel. NaNs propagate silently through PyTorch
    and corrupt downstream training; catching them at the originating layer
    is the difference between a 5-minute fix and a 6-hour debug session.

    Args:
        model: Any ``nn.Module``. Typically called *after* a backward pass so
            ``.grad`` tensors are populated.
        check_grads: If ``True``, scan ``param.grad`` for each parameter that
            has one. Skip parameters with no gradient (frozen layers).
        check_params: If ``True``, scan ``param.data`` directly. Catches
            corrupted weights independently of gradient flow.

    Returns:
        ``CheckResult`` with ``passed=False`` if any tensor contains NaN/Inf.
        ``evidence`` lists offending layer names with the kind of contamination
        and the tensor shape, so the caller can surface it in logs / UI.
    """
    offenders: list[dict[str, str | list[int]]] = []

    for name, param in model.named_parameters():
        if check_params:
            if torch.isnan(param.data).any():
                offenders.append({"layer": name, "kind": "param_nan", "shape": list(param.shape)})
            elif torch.isinf(param.data).any():
                offenders.append({"layer": name, "kind": "param_inf", "shape": list(param.shape)})

        if check_grads and param.grad is not None:
            if torch.isnan(param.grad).any():
                offenders.append(
                    {"layer": name, "kind": "grad_nan", "shape": list(param.grad.shape)}
                )
            elif torch.isinf(param.grad).any():
                offenders.append(
                    {"layer": name, "kind": "grad_inf", "shape": list(param.grad.shape)}
                )

    passed = len(offenders) == 0
    if passed:
        msg = "No NaN/Inf detected in parameters or gradients."
    else:
        first = offenders[0]
        msg = (
            f"NaN/Inf detected in {len(offenders)} tensor(s); "
            f"first offender: {first['layer']} ({first['kind']})"
        )

    return CheckResult(
        name="nan_inf",
        passed=passed,
        score=1.0 if passed else 0.0,
        message=msg,
        evidence={"offenders": offenders},
    )
