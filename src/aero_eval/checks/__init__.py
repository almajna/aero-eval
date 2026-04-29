"""Individual semantic-correctness checks for PyTorch models."""

from __future__ import annotations

from aero_eval.checks.dead_neurons import check_dead_neurons
from aero_eval.checks.nan_inf import check_nan_inf
from aero_eval.checks.overfit import check_single_batch_overfit
from aero_eval.checks.param_update import check_param_update_magnitude

__all__ = [
    "check_dead_neurons",
    "check_nan_inf",
    "check_param_update_magnitude",
    "check_single_batch_overfit",
]
