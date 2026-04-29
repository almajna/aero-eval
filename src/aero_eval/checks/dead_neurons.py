"""Detect dead neurons (saturated, clamped, or zero-variance) via forward hooks."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch
from torch import nn

from aero_eval.result import CheckResult

# Layer types whose outputs we care about. Linear and Conv pre-activations are
# the canonical dead-neuron surface. Norm layers have constructed variance ~= 1
# and would false-positive.
_DEFAULT_TARGET_TYPES: tuple[type[nn.Module], ...] = (
    nn.Linear,
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
)


@contextmanager
def _record_outputs(
    model: nn.Module,
    target_types: tuple[type[nn.Module], ...],
    excluded: set[str],
) -> Iterator[dict[str, list[torch.Tensor]]]:
    """Attach forward hooks to every layer of the target types and collect outputs."""
    captures: dict[str, list[torch.Tensor]] = {}
    handles = []

    for name, module in model.named_modules():
        if not isinstance(module, target_types):
            continue
        if name in excluded:
            continue
        captures[name] = []

        def _make_hook(layer_name: str):
            def _hook(_mod, _inp, out):
                # Only collect tensor outputs; some custom layers return tuples.
                if isinstance(out, torch.Tensor):
                    captures[layer_name].append(out.detach())

            return _hook

        handles.append(module.register_forward_hook(_make_hook(name)))

    try:
        yield captures
    finally:
        for h in handles:
            h.remove()


def check_dead_neurons(
    model: nn.Module,
    inputs: tuple[torch.Tensor, ...] | torch.Tensor,
    *,
    variance_threshold: float = 1e-5,
    max_dead_channel_ratio: float = 0.50,
    target_types: tuple[type[nn.Module], ...] = _DEFAULT_TARGET_TYPES,
    excluded_layer_names: set[str] | None = None,
) -> CheckResult:
    """Run the model and check per-channel output variance of target layers.

    A "dead" channel is one whose pre-activation variance across the batch and
    spatial axes is below ``variance_threshold``. This catches:

    - ReLU networks where most units are saturated at zero from initialization
    - Models post-processed with ``torch.clamp`` to fake gradient stability
      (the Goodhart pattern from the V2 plan — clamping kills variance)
    - Layers receiving zeroed-out inputs from upstream bugs

    A check fails for a layer if more than ``max_dead_channel_ratio`` of its
    output channels are dead. The overall check fails if any layer fails.

    Note:
        Intentionally sparse architectures (MoE with low top-k, sparse
        attention) will trigger this. Pass their layer names via
        ``excluded_layer_names``.

    Args:
        model: Module to evaluate. Set to ``eval()`` for stable BN/Dropout
            behavior unless you are explicitly testing training-time dynamics.
        inputs: Forward-pass arguments. A single tensor is treated as
            ``model(inputs)``; a tuple is unpacked as ``model(*inputs)``.
        variance_threshold: Channels with variance below this are dead.
        max_dead_channel_ratio: Max share of dead channels per layer before
            that layer fails.
        target_types: Module types to instrument. Defaults to Linear+Conv.
        excluded_layer_names: Layer names to skip (e.g. MoE routers).
    """
    excluded = excluded_layer_names or set()

    with _record_outputs(model, target_types, excluded) as captures, torch.no_grad():
        if isinstance(inputs, torch.Tensor):
            model(inputs)
        else:
            model(*inputs)

    layer_results: dict[str, dict[str, float]] = {}
    failing_layers: list[str] = []

    for layer_name, outputs in captures.items():
        if not outputs:
            continue
        # Concatenate captures from this forward pass (typically one).
        out = torch.cat([o.float() for o in outputs], dim=0)
        # Treat dim 1 as the channel axis for Conv; for Linear, last dim.
        # Compute variance over all dims except the channel dim.
        if out.ndim >= 2 and any(
            isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d))
            for n, m in model.named_modules()
            if n == layer_name
        ):
            channel_dim = 1
        else:
            channel_dim = out.ndim - 1

        # Move channel dim to front, flatten the rest, compute per-channel var.
        out_perm = out.movedim(channel_dim, 0).reshape(out.shape[channel_dim], -1)
        per_channel_var = out_perm.var(dim=1, unbiased=False)
        n_dead = int((per_channel_var < variance_threshold).sum().item())
        n_channels = per_channel_var.numel()
        dead_ratio = n_dead / n_channels if n_channels else 0.0

        layer_results[layer_name] = {
            "dead_channels": n_dead,
            "total_channels": n_channels,
            "dead_ratio": dead_ratio,
            "min_variance": float(per_channel_var.min().item()),
        }
        if dead_ratio > max_dead_channel_ratio:
            failing_layers.append(layer_name)

    passed = len(failing_layers) == 0
    if not layer_results:
        return CheckResult(
            name="dead_neurons",
            passed=True,
            score=1.0,
            message="No target layers found to instrument.",
            evidence={"layer_results": {}},
        )

    worst_ratio = max((r["dead_ratio"] for r in layer_results.values()), default=0.0)
    score = max(0.0, 1.0 - worst_ratio)

    if passed:
        msg = (
            f"All {len(layer_results)} instrumented layers under "
            f"{max_dead_channel_ratio:.0%} dead-channel threshold."
        )
    else:
        msg = (
            f"{len(failing_layers)}/{len(layer_results)} layers exceed "
            f"{max_dead_channel_ratio:.0%} dead channels; first: {failing_layers[0]}"
        )

    return CheckResult(
        name="dead_neurons",
        passed=passed,
        score=score,
        message=msg,
        evidence={
            "layer_results": layer_results,
            "failing_layers": failing_layers,
            "variance_threshold": variance_threshold,
        },
    )
