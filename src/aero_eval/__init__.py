"""aero-eval: adversarial evaluation harness for LLM-generated PyTorch models."""

from aero_eval.checks import (
    check_dead_neurons,
    check_nan_inf,
    check_param_update_magnitude,
    check_single_batch_overfit,
)
from aero_eval.checks.param_update import snapshot_params
from aero_eval.result import CheckResult

__version__ = "0.1.0a1"

__all__ = [
    "CheckResult",
    "__version__",
    "check_dead_neurons",
    "check_nan_inf",
    "check_param_update_magnitude",
    "check_single_batch_overfit",
    "snapshot_params",
]
