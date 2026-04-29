"""Tests for the baseline-study module assembler.

The assembler is the layer between raw LLM output and an executable Python
module. It is regression-prone (whitespace, fences, indent levels) and is
the only piece of the baseline study that's worth covering with unit tests
— the rest is API I/O and subprocess orchestration.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# studies/ is not a package; import by path.
_BASELINE_DIR = Path(__file__).resolve().parents[2] / "studies" / "baseline"
sys.path.insert(0, str(_BASELINE_DIR))

from generate import assemble_module  # type: ignore[import-not-found]  # noqa: E402


def _exec_assembled(body: str):
    """Assemble a body, exec it, return the resulting CustomAttention class."""
    src = assemble_module(body)
    ns: dict = {}
    exec(src, ns)
    return ns["CustomAttention"]


def test_flush_left_body_assembles():
    body = "bsz, seq, _ = q.shape\nreturn q + k + v"
    cls = _exec_assembled(body)
    assert cls.__name__ == "CustomAttention"


def test_four_space_indent_normalizes():
    body = "    bsz, seq, _ = q.shape\n    return q + k + v"
    cls = _exec_assembled(body)
    assert cls.__name__ == "CustomAttention"


def test_eight_space_indent_normalizes():
    body = "        bsz, seq, _ = q.shape\n        return q + k + v"
    cls = _exec_assembled(body)
    assert cls.__name__ == "CustomAttention"


def test_fenced_code_block_stripped():
    body = "```python\nbsz, seq, _ = q.shape\nreturn q + k + v\n```"
    cls = _exec_assembled(body)
    assert cls.__name__ == "CustomAttention"


def test_fenced_and_indented_combined():
    body = "```python\n    bsz, seq, _ = q.shape\n    return q + k + v\n```"
    cls = _exec_assembled(body)
    assert cls.__name__ == "CustomAttention"


def test_empty_body_yields_pass_stub():
    src = assemble_module("")
    ns: dict = {}
    exec(src, ns)
    # The stub is a literal `pass`, so instantiation works but forward
    # returns None — that's fine, the *evaluator* will categorize it as
    # ``shape_contract`` later.
    assert "CustomAttention" in ns


def test_whitespace_only_yields_pass_stub():
    src = assemble_module("   \n\n   \n")
    ns: dict = {}
    exec(src, ns)
    assert "CustomAttention" in ns


@pytest.mark.parametrize(
    "leading_chars",
    ["", "    ", "  ", "\t"],
)
def test_various_leading_indents_assemble(leading_chars):
    body = f"{leading_chars}bsz, seq, _ = q.shape\n{leading_chars}return q"
    src = assemble_module(body)
    ns: dict = {}
    # We don't strictly require this to *exec* cleanly for tab indent (Python
    # does not mix tabs and spaces well), but assemble_module must not crash.
    if "\t" not in leading_chars:
        exec(src, ns)
        assert "CustomAttention" in ns
