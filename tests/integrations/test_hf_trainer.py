"""Tests for AeroEvalCallback.

These exercise the callback against a real ``transformers.Trainer`` on a tiny
model so we catch interface drift between releases. Skipped if transformers
is not installed (e.g. base-only install).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

transformers = pytest.importorskip("transformers")
# ---------------------------------------------------------------------------
# Tiny HF-style model + dataset for end-to-end callback exercise
# ---------------------------------------------------------------------------
from dataclasses import dataclass  # noqa: E402

from transformers import Trainer, TrainerControl, TrainerState, TrainingArguments  # noqa: E402
from transformers.modeling_outputs import ModelOutput  # noqa: E402

from aero_eval.integrations.hf_trainer import (  # noqa: E402
    AeroEvalCallback,
    AeroEvalCheckError,
    AeroEvalConfig,
    _HfDictAdapter,
)


@dataclass
class _TinyOutput(ModelOutput):
    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None


class _TinyHfModel(nn.Module):
    """Minimal HF-style model: takes ``input_ids`` + ``labels``, returns a
    ``ModelOutput`` so the Trainer's loss extraction works."""

    def __init__(self, vocab_size: int = 32, d: int = 16, n_classes: int = 4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d)
        self.fc1 = nn.Linear(d, 32)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(32, n_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids=None, labels=None, **kwargs):
        x = self.emb(input_ids).mean(dim=1)
        logits = self.fc2(self.act(self.fc1(x)))
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return _TinyOutput(loss=loss, logits=logits)


class _TinyDataset(torch.utils.data.Dataset):
    def __init__(self, n: int = 32, seq_len: int = 8, vocab: int = 32, n_classes: int = 4):
        g = torch.Generator().manual_seed(0)
        self.input_ids = torch.randint(0, vocab, (n, seq_len), generator=g)
        self.labels = torch.randint(0, n_classes, (n,), generator=g)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {"input_ids": self.input_ids[i], "labels": self.labels[i]}


def _make_trainer(
    callback: AeroEvalCallback,
    *,
    max_steps: int = 5,
    tmp_path,
) -> Trainer:
    args = TrainingArguments(
        output_dir=str(tmp_path),
        per_device_train_batch_size=4,
        max_steps=max_steps,
        logging_strategy="no",
        save_strategy="no",
        report_to=[],
        disable_tqdm=True,
        learning_rate=1e-3,
    )
    return Trainer(
        model=_TinyHfModel(),
        args=args,
        train_dataset=_TinyDataset(),
        callbacks=[callback],
    )


# ---------------------------------------------------------------------------
# Unit tests on the callback's adapter / private logic (no Trainer needed)
# ---------------------------------------------------------------------------


def test_hf_dict_adapter_forwards_kwargs():
    inner = _TinyHfModel()
    adapter = _HfDictAdapter(inner)
    batch = {
        "input_ids": torch.randint(0, 32, (2, 8)),
        "labels": torch.randint(0, 4, (2,)),
    }
    out = adapter(batch)
    assert hasattr(out, "loss")
    assert out.loss.ndim == 0


def test_hf_dict_adapter_named_modules_delegates():
    inner = _TinyHfModel()
    adapter = _HfDictAdapter(inner)
    inner_names = {n for n, _ in inner.named_modules()}
    adapter_names = {n for n, _ in adapter.named_modules()}
    # Adapter exposes the inner model's modules verbatim.
    assert inner_names == adapter_names


def test_config_defaults():
    cfg = AeroEvalConfig()
    assert cfg.check_every_n_steps == 50
    assert cfg.run_overfit_preflight is False
    assert cfg.run_dead_neuron_preflight is True
    assert cfg.fail_fast is False


def test_results_starts_empty():
    cb = AeroEvalCallback()
    assert cb.results == []


# ---------------------------------------------------------------------------
# End-to-end: real Trainer.train() with the callback attached
# ---------------------------------------------------------------------------


def test_callback_runs_with_real_trainer(tmp_path):
    """Smoke test: Trainer.train() completes without errors when callback attached."""
    cb = AeroEvalCallback(
        config=AeroEvalConfig(
            check_every_n_steps=2,
            run_dead_neuron_preflight=True,
            run_overfit_preflight=False,
        )
    )
    trainer = _make_trainer(cb, max_steps=5, tmp_path=tmp_path)
    trainer.train()

    # Streaming checks fired at step 2 and 4 -> 2 NaN/Inf + 2 param-update = 4
    # plus 1 dead-neuron preflight = 5 results minimum.
    assert len(cb.results) >= 3
    names = {r.name for r in cb.results}
    assert "nan_inf" in names
    # Dead-neuron may fail on a randomly-initialized tiny model + tiny vocab;
    # we don't assert pass/fail — just that the check ran.
    assert "dead_neurons" in names


def test_callback_with_streaming_disabled(tmp_path):
    cb = AeroEvalCallback(
        config=AeroEvalConfig(
            check_every_n_steps=0,  # streaming off
            run_dead_neuron_preflight=False,
        )
    )
    trainer = _make_trainer(cb, max_steps=3, tmp_path=tmp_path)
    trainer.train()
    assert cb.results == []


def test_fail_fast_raises_on_nan(tmp_path):
    """Inject a NaN into params mid-training; fail_fast should raise."""
    cb = AeroEvalCallback(
        config=AeroEvalConfig(
            check_every_n_steps=1,
            run_dead_neuron_preflight=False,
            fail_fast=True,
        )
    )
    trainer = _make_trainer(cb, max_steps=5, tmp_path=tmp_path)

    # Corrupt the first parameter so the very first NaN check fails.
    with torch.no_grad():
        first_param = next(trainer.model.parameters())
        first_param[0, 0] = float("nan")

    with pytest.raises(AeroEvalCheckError) as exc_info:
        trainer.train()
    assert exc_info.value.result.name == "nan_inf"


def test_stop_on_failure_halts_training(tmp_path):
    """Without fail_fast but with stop_on_failure, training should halt early."""
    cb = AeroEvalCallback(
        config=AeroEvalConfig(
            check_every_n_steps=1,
            run_dead_neuron_preflight=False,
            fail_fast=False,
            stop_on_failure=True,
        )
    )
    trainer = _make_trainer(cb, max_steps=10, tmp_path=tmp_path)

    with torch.no_grad():
        first_param = next(trainer.model.parameters())
        first_param[0, 0] = float("nan")

    trainer.train()
    # global_step should be much less than max_steps because we halted.
    assert trainer.state.global_step < 10
    # And we should have at least one failed nan_inf result.
    assert any(r.name == "nan_inf" and not r.passed for r in cb.results)


def test_peek_first_batch_does_not_skip_samples(tmp_path):
    """The peek inside on_train_begin must not consume training samples.

    We assert that the Trainer still sees all dataset samples by counting how
    many steps it actually runs against a fresh dataset with known size.
    """
    cb = AeroEvalCallback(
        config=AeroEvalConfig(
            check_every_n_steps=0,
            run_dead_neuron_preflight=True,  # forces a peek
        )
    )
    # 32-sample dataset, batch_size=4 -> exactly 8 steps per epoch.
    args = TrainingArguments(
        output_dir=str(tmp_path),
        per_device_train_batch_size=4,
        num_train_epochs=1,
        logging_strategy="no",
        save_strategy="no",
        report_to=[],
        disable_tqdm=True,
        learning_rate=1e-3,
    )
    trainer = Trainer(
        model=_TinyHfModel(),
        args=args,
        train_dataset=_TinyDataset(n=32),
        callbacks=[cb],
    )
    trainer.train()
    # 32 / 4 = 8 expected steps; if the peek had consumed one, we'd see 7.
    assert trainer.state.global_step == 8


def test_distributed_skip_when_not_main_process():
    """When ``state.is_world_process_zero`` is False, pre-flight is skipped."""
    cb = AeroEvalCallback(
        config=AeroEvalConfig(run_dead_neuron_preflight=True, check_every_n_steps=0)
    )
    state = TrainerState()
    state.is_world_process_zero = False
    control = TrainerControl()

    # Pass dummy non-None model + dataloader to get past the early return.
    cb.on_train_begin(
        args=None,
        state=state,
        control=control,
        model=_TinyHfModel(),
        train_dataloader=[{"input_ids": torch.zeros(1, 1, dtype=torch.long)}],
    )
    assert cb.results == []
