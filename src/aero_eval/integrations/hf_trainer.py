"""HuggingFace ``Trainer`` integration.

Provides :class:`AeroEvalCallback`, a drop-in ``TrainerCallback`` that runs
``aero-eval`` checks at well-defined points in the training loop:

- ``on_train_begin``: pre-flight overfit and dead-neuron checks before any GPU
  hours are burned.
- ``on_step_end`` (every ``check_every_n_steps``): streaming NaN/Inf and
  parameter-update-magnitude checks. Catches divergence early.
- ``on_train_end``: final summary log of every check that ran.

The callback never raises by default; it logs failures and optionally sets
``control.should_training_stop = True``. Set ``fail_fast=True`` to raise on
first failure (useful in CI / smoke-test contexts).

This module imports ``transformers`` lazily so users who do not need
HuggingFace integration can install ``aero-eval`` without it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from aero_eval.checks.dead_neurons import check_dead_neurons
from aero_eval.checks.nan_inf import check_nan_inf
from aero_eval.checks.overfit import check_single_batch_overfit
from aero_eval.checks.param_update import check_param_update_magnitude, snapshot_params
from aero_eval.result import CheckResult

if TYPE_CHECKING:
    from transformers import TrainerCallback as _CallbackBase
    from transformers import TrainerControl, TrainerState
    from transformers.training_args import TrainingArguments
else:
    try:
        from transformers import TrainerCallback as _CallbackBase
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "aero_eval.integrations.hf_trainer requires transformers. "
            "Install with: pip install 'aero-eval[hf]'"
        ) from e


logger = logging.getLogger(__name__)


@dataclass
class AeroEvalConfig:
    """Configuration for :class:`AeroEvalCallback`.

    Attributes:
        check_every_n_steps: Cadence for streaming checks (NaN/Inf,
            param-update-magnitude). 0 disables streaming.
        run_overfit_preflight: If ``True``, run the single-batch overfit check
            against the first training batch before any optimizer step. Off by
            default — it modifies model parameters and should only be enabled
            against a freshly-initialized model the user is willing to retrain.
        run_dead_neuron_preflight: If ``True``, run the dead-neuron check on
            the first batch.
        fail_fast: If ``True``, raise ``AeroEvalCheckError`` on first failure.
        stop_on_failure: If ``True`` (default), set
            ``control.should_training_stop = True`` on failure.
        nan_inf_check_grads: Forwarded to :func:`check_nan_inf`.
        param_update_max_offender_ratio: Forwarded to
            :func:`check_param_update_magnitude`.
        dead_neuron_max_dead_ratio: Forwarded to :func:`check_dead_neurons`.
        overfit_max_steps: Forwarded to :func:`check_single_batch_overfit`.
        excluded_dead_neuron_layers: Layer names to skip in the dead-neuron
            check (MoE routers, intentional sparse layers).
    """

    check_every_n_steps: int = 50
    run_overfit_preflight: bool = False
    run_dead_neuron_preflight: bool = True
    fail_fast: bool = False
    stop_on_failure: bool = True
    nan_inf_check_grads: bool = True
    param_update_max_offender_ratio: float = 0.20
    dead_neuron_max_dead_ratio: float = 0.50
    overfit_max_steps: int = 200
    excluded_dead_neuron_layers: set[str] = field(default_factory=set)


class AeroEvalCheckError(RuntimeError):
    """Raised when ``fail_fast=True`` and an ``aero-eval`` check fails."""

    def __init__(self, result: CheckResult):
        super().__init__(f"[aero-eval] {result.name} failed: {result.message}")
        self.result = result


class _HfDictAdapter(nn.Module):
    """Wrap an HF model so it can be invoked positionally with a dict batch.

    ``check_dead_neurons`` calls ``model(*inputs)``; HF models expect
    ``model(**batch)``. This adapter consumes a single positional dict arg and
    forwards it as kwargs. We delegate ``named_modules`` to the inner model so
    that the dead-neuron check's forward hooks attach to the real layers.
    """

    def __init__(self, inner: nn.Module):
        super().__init__()
        # Avoid registering inner as a submodule — we don't want hooks on us.
        object.__setattr__(self, "_inner", inner)

    def named_modules(self, *args, **kwargs):
        return self._inner.named_modules(*args, **kwargs)

    def forward(self, batch: dict[str, Any]) -> Any:
        return self._inner(**batch)


class AeroEvalCallback(_CallbackBase):
    """``transformers.Trainer`` callback that runs ``aero-eval`` checks live.

    Usage::

        from transformers import Trainer
        from aero_eval.integrations.hf_trainer import AeroEvalCallback

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            callbacks=[AeroEvalCallback()],
        )
        trainer.train()

    Customize via :class:`AeroEvalConfig`::

        AeroEvalCallback(config=AeroEvalConfig(check_every_n_steps=10, fail_fast=True))
    """

    def __init__(self, config: AeroEvalConfig | None = None):
        super().__init__()
        self.config = config or AeroEvalConfig()
        self._results: list[CheckResult] = []
        self._pre_step_snapshot: dict[str, torch.Tensor] | None = None

    @property
    def results(self) -> list[CheckResult]:
        """All check results recorded during the training run, in order."""
        return list(self._results)

    # ------------------------------------------------------------------ #
    # TrainerCallback hooks
    # ------------------------------------------------------------------ #

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: nn.Module | None = None,
        train_dataloader: Any = None,
        **kwargs: Any,
    ) -> TrainerControl:
        if model is None or train_dataloader is None:
            logger.warning(
                "[aero-eval] on_train_begin missing model or dataloader; "
                "skipping pre-flight checks."
            )
            return control

        # Distributed safety: only run pre-flight on the main process.
        if hasattr(state, "is_world_process_zero") and not state.is_world_process_zero:
            return control

        first_batch = self._peek_first_batch(train_dataloader)
        if first_batch is None:
            logger.warning("[aero-eval] could not peek first batch; skipping pre-flight.")
            return control

        if self.config.run_dead_neuron_preflight:
            self._run_dead_neuron_preflight(model, first_batch, control)

        if self.config.run_overfit_preflight:
            self._run_overfit_preflight(model, first_batch, control)

        return control

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: nn.Module | None = None,
        **kwargs: Any,
    ) -> TrainerControl:
        # Snapshot before optimizer step so we can compare deltas in on_step_end.
        if (
            model is not None
            and self.config.check_every_n_steps > 0
            and (state.global_step + 1) % self.config.check_every_n_steps == 0
        ):
            self._pre_step_snapshot = snapshot_params(model)
        return control

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: nn.Module | None = None,
        **kwargs: Any,
    ) -> TrainerControl:
        if model is None or self.config.check_every_n_steps <= 0:
            return control
        if state.global_step % self.config.check_every_n_steps != 0:
            return control

        # NaN/Inf check is cheap and always runs on the cadence.
        nan_result = check_nan_inf(model, check_grads=self.config.nan_inf_check_grads)
        self._record(nan_result, control)

        # Param-update check requires the pre-step snapshot.
        if self._pre_step_snapshot is not None:
            update_result = check_param_update_magnitude(
                model,
                self._pre_step_snapshot,
                max_offender_ratio=self.config.param_update_max_offender_ratio,
            )
            self._record(update_result, control)
            self._pre_step_snapshot = None

        return control

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        n_pass = sum(1 for r in self._results if r.passed)
        n_fail = len(self._results) - n_pass
        logger.info("[aero-eval] training complete: %d pass / %d fail", n_pass, n_fail)
        for r in self._results:
            if not r.passed:
                logger.warning("[aero-eval] FAIL %s — %s", r.name, r.message)
        return control

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _peek_first_batch(dataloader: Any) -> Any:
        """Return the first batch from a dataloader without skipping training samples.

        ``iter(dataloader)`` here creates a transient iterator for the peek;
        when the Trainer subsequently calls ``iter(dataloader)`` for the real
        training loop it gets a fresh iterator that yields all samples.
        """
        try:
            return next(iter(dataloader))
        except (StopIteration, TypeError):
            return None

    def _run_dead_neuron_preflight(
        self,
        model: nn.Module,
        batch: Any,
        control: TrainerControl,
    ) -> None:
        target_model, inputs = self._adapt_batch_for_check(model, batch)
        if target_model is None:
            logger.warning(
                "[aero-eval] dead-neuron preflight skipped: unsupported batch type %s",
                type(batch).__name__,
            )
            return

        was_training = model.training
        model.eval()
        try:
            result = check_dead_neurons(
                target_model,
                inputs,
                max_dead_channel_ratio=self.config.dead_neuron_max_dead_ratio,
                excluded_layer_names=self.config.excluded_dead_neuron_layers,
            )
        finally:
            if was_training:
                model.train()
        self._record(result, control)

    def _run_overfit_preflight(
        self,
        model: nn.Module,
        batch: Any,
        control: TrainerControl,
    ) -> None:
        """Run single-batch overfit using the model's own loss computation."""

        def _forward(m: nn.Module, b: Any) -> tuple[torch.Tensor, torch.Tensor]:
            outputs = m(**b) if isinstance(b, dict) else m(b)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs
            if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
                raise ValueError(
                    "[aero-eval] overfit preflight: model did not return a scalar loss. "
                    "Disable run_overfit_preflight if your model has a non-standard "
                    "training signature."
                )
            # check_single_batch_overfit treats 0-D tensors as pre-computed losses;
            # the second tuple element is then ignored.
            return loss, torch.tensor(0)

        result = check_single_batch_overfit(
            model,
            batch,
            loss_fn=None,
            max_steps=self.config.overfit_max_steps,
            forward_fn=_forward,
        )
        self._record(result, control)

    @staticmethod
    def _adapt_batch_for_check(model: nn.Module, batch: Any) -> tuple[nn.Module | None, Any]:
        """Convert a Trainer batch into the (model, inputs) shape checks expect.

        ``check_dead_neurons`` calls ``model(*inputs)`` when ``inputs`` is not
        a single tensor. For a dict batch we wrap the dict in a 1-tuple so the
        call becomes ``adapter(batch_dict)`` and the adapter forwards as
        kwargs.

        Returns ``(None, None)`` if the batch type is unsupported.
        """
        if isinstance(batch, dict):
            return _HfDictAdapter(model), (batch,)
        if isinstance(batch, torch.Tensor):
            return model, batch
        if isinstance(batch, (list, tuple)):
            return model, tuple(batch)
        return None, None

    def _record(self, result: CheckResult, control: TrainerControl) -> None:
        self._results.append(result)
        if result.passed:
            logger.debug("[aero-eval] %s passed: %s", result.name, result.message)
            return

        logger.warning("[aero-eval] %s FAILED: %s", result.name, result.message)
        if self.config.fail_fast:
            raise AeroEvalCheckError(result)
        if self.config.stop_on_failure:
            control.should_training_stop = True
