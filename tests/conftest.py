"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import os
import random

import numpy as np
import pytest
import torch

# Determinism env vars must be set before any torch op.
# These are belt-and-braces; per-test fixtures pin seeds explicitly.
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


@pytest.fixture(autouse=True)
def _pin_seeds():
    """Pin all RNGs before every test. FPR suite depends on this."""
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    yield


@pytest.fixture
def device() -> torch.device:
    """CPU device for unit tests; GPU tests use the `gpu` marker explicitly."""
    return torch.device("cpu")
