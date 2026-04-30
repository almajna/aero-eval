"""Fixtures for the false-positive regression suite.

Loads tiny known-good HuggingFace models and prepares a real training scenario
against which all aero-eval checks must pass. If the model download fails (no
network, HF Hub down), the affected tests skip rather than fail — flaky
infrastructure should not redden the test suite.
"""

from __future__ import annotations

import pytest

transformers = pytest.importorskip("transformers")
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer  # noqa: E402

# Tiny known-good models. Pinned versions would be ideal but transformers
# loads by name; we accept that "current latest" is the version under test.
KNOWN_GOOD_MODELS = {
    "bert-tiny": {
        # Google's distilled BERT — 4.4M params, 2 layers, hidden=128.
        # We avoid prajjwal1/bert-tiny because its older config.json lacks
        # a model_type key and recent transformers won't auto-load it.
        "id": "google/bert_uncased_L-2_H-128_A-2",
        "loader": "encoder",
    },
    "tinystories-1m": {
        "id": "roneneldan/TinyStories-1M",
        "loader": "causal",
    },
    "tinystories-3m": {
        "id": "roneneldan/TinyStories-3M",
        "loader": "causal",
    },
}


def _load(model_id: str, loader_kind: str):
    """Load a tokenizer + model. Skip the test on download failures."""
    try:
        tok = AutoTokenizer.from_pretrained(model_id)
        if loader_kind == "encoder":
            model = AutoModel.from_pretrained(model_id)
        elif loader_kind == "causal":
            model = AutoModelForCausalLM.from_pretrained(model_id)
        else:
            raise ValueError(f"unknown loader kind: {loader_kind}")
    except Exception as e:  # OSError, ConnectionError, HF-specific errors, ...
        pytest.skip(f"could not load {model_id} (network/hub issue): {e}")
    return tok, model


@pytest.fixture(scope="module")
def bert_tiny():
    tok, model = _load(KNOWN_GOOD_MODELS["bert-tiny"]["id"], "encoder")
    return tok, model


@pytest.fixture(scope="module")
def tinystories_1m():
    tok, model = _load(KNOWN_GOOD_MODELS["tinystories-1m"]["id"], "causal")
    return tok, model


@pytest.fixture(scope="module")
def tinystories_3m():
    tok, model = _load(KNOWN_GOOD_MODELS["tinystories-3m"]["id"], "causal")
    return tok, model


# --------------------------------------------------------------------------- #
# Tiny dataset for real training scenarios. We keep this tiny on purpose:
# the goal is "real training, not synthetic noise" — not "expensive training".
# --------------------------------------------------------------------------- #

# A handful of canonical sentences. Deterministic, no external download.
_CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "All happy families are alike; each unhappy family is unhappy in its own way.",
    "It was the best of times, it was the worst of times.",
    "In a hole in the ground there lived a hobbit.",
    "Call me Ishmael.",
    "It is a truth universally acknowledged that a single man in possession of a good fortune must be in want of a wife.",
    "The past is a foreign country; they do things differently there.",
    "Two roads diverged in a wood, and I took the one less traveled by.",
    "To be or not to be, that is the question.",
    "I think, therefore I am.",
    "The only thing we have to fear is fear itself.",
    "Stay hungry, stay foolish.",
    "Float like a butterfly, sting like a bee.",
    "Ask not what your country can do for you, ask what you can do for your country.",
    "The journey of a thousand miles begins with a single step.",
    "Whereof one cannot speak, thereof one must be silent.",
]


def make_causal_batch(tokenizer, n: int = 8, seq_len: int = 32) -> dict:
    """Build a causal-LM batch from the canonical corpus.

    Returns a dict with ``input_ids`` and ``labels`` (labels = input_ids for
    next-token prediction). Pads to ``seq_len`` with the EOS token.
    """
    sentences = _CORPUS[:n]
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    enc = tokenizer(
        sentences,
        padding="max_length",
        truncation=True,
        max_length=seq_len,
        return_tensors="pt",
    )
    labels = enc["input_ids"].clone()
    return {
        "input_ids": enc["input_ids"],
        "labels": labels,
        "attention_mask": enc["attention_mask"],
    }


def make_mlm_batch(tokenizer, n: int = 8, seq_len: int = 32) -> dict:
    """Build a masked-LM-style batch for the encoder check.

    BERT-tiny's vanilla forward returns hidden states; we'll wrap loss
    computation in the test where needed. This fixture just gives us inputs.
    """
    sentences = _CORPUS[:n]
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "[PAD]"
    enc = tokenizer(
        sentences,
        padding="max_length",
        truncation=True,
        max_length=seq_len,
        return_tensors="pt",
    )
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
