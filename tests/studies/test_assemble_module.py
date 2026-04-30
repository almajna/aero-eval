"""Tests for the shared module assembler.

The assembler is the layer between raw LLM output and an executable Python
module. It is regression-prone (whitespace, fences, indent levels) and is
the only piece of the studies that's worth covering with unit tests.

These tests target the shared ``studies/_assembler.py`` module and explicitly
include the Opus 7-then-8-space pattern that broke the original
``textwrap.dedent``-based assembler in production.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

# studies/ is not a package; import by path.
_STUDIES_DIR = Path(__file__).resolve().parents[2] / "studies"
sys.path.insert(0, str(_STUDIES_DIR))

from _assembler import (  # type: ignore[import-not-found]  # noqa: E402
    assemble_module,
    robust_dedent,
)

# Minimal fake header that we wrap bodies inside. The body indent target is
# 8 spaces, which matches both real headers (CustomAttention, GQA).
_FAKE_HEADER = "def forward(self, q, k, v, mask=None):\n"


def _assembled_parses(body: str) -> bool:
    """Wrap body via assembler, attempt to ast.parse. True iff parse succeeds."""
    src = assemble_module(_FAKE_HEADER, body)
    try:
        ast.parse(src)
    except SyntaxError:
        return False
    return True


# --------------------------------------------------------------------------- #
# robust_dedent unit tests
# --------------------------------------------------------------------------- #


def test_robust_dedent_uniform_indent():
    text = "    a = 1\n    b = 2"
    out = robust_dedent(text)
    assert out == "a = 1\nb = 2"


def test_robust_dedent_flush_left_unchanged():
    text = "a = 1\nb = 2"
    assert robust_dedent(text) == text


def test_robust_dedent_under_indented_first_line():
    """Opus pattern: 7 spaces, then 8, 8, 8."""
    text = (
        "       bsz = q.shape[0]\n"
        "        a = self.q_proj(q)\n"
        "        b = self.k_proj(k)\n"
        "        return a + b"
    )
    out = robust_dedent(text)
    expected = "bsz = q.shape[0]\na = self.q_proj(q)\nb = self.k_proj(k)\nreturn a + b"
    assert out == expected


def test_robust_dedent_two_line_under_indent():
    """Edge: just two lines with mismatched indent."""
    text = "       a = 1\n        b = 2"
    out = robust_dedent(text)
    # No multi-occurrence indent; falls back to max=8. 7 < 8 so first line clamps to 0.
    assert out == "a = 1\nb = 2"


def test_robust_dedent_preserves_nested_blocks():
    text = "    if x:\n        y = 1\n    else:\n        y = 2"
    out = robust_dedent(text)
    expected = "if x:\n    y = 1\nelse:\n    y = 2"
    assert out == expected


# --------------------------------------------------------------------------- #
# assemble_module integration tests — every case must parse cleanly when
# wrapped inside _FAKE_HEADER.
# --------------------------------------------------------------------------- #


def test_flush_left_body_assembles():
    body = "bsz, seq, _ = q.shape\nreturn q + k + v"
    assert _assembled_parses(body)


def test_four_space_indent_normalizes():
    body = "    bsz, seq, _ = q.shape\n    return q + k + v"
    assert _assembled_parses(body)


def test_eight_space_indent_normalizes():
    body = "        bsz, seq, _ = q.shape\n        return q + k + v"
    assert _assembled_parses(body)


def test_fenced_code_block_stripped():
    body = "```python\nbsz, seq, _ = q.shape\nreturn q + k + v\n```"
    assert _assembled_parses(body)


def test_fenced_and_indented_combined():
    body = "```python\n    bsz, seq, _ = q.shape\n    return q + k + v\n```"
    assert _assembled_parses(body)


def test_opus_under_indented_first_line_regression():
    """Regression: real Opus 4.7 GQA output with 7-space first line.

    This pattern broke the original textwrap.dedent assembler and caused 32
    of 41 Opus samples to be misclassified as syntax errors before the fix.
    Captured here verbatim from studies/gqa/raw/claude-opus-4-7/001.txt.
    """
    body = (
        "       bsz, seq, _ = q.shape\n"
        "        q_h = self.q_proj(q).view(bsz, seq, self.n_heads, self.head_dim).transpose(1, 2)\n"
        "        k_h = self.k_proj(k).view(bsz, seq, self.n_kv_heads, self.head_dim).transpose(1, 2)\n"
        "        v_h = self.v_proj(v).view(bsz, seq, self.n_kv_heads, self.head_dim).transpose(1, 2)\n"
        "        k_h = k_h.repeat_interleave(self.n_rep, dim=1)\n"
        "        v_h = v_h.repeat_interleave(self.n_rep, dim=1)\n"
        "        scores = torch.matmul(q_h, k_h.transpose(-2, -1)) / (self.head_dim ** 0.5)\n"
        "        if mask is not None:\n"
        '            scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))\n'
        "        attn = F.softmax(scores, dim=-1)\n"
        "        out = torch.matmul(attn, v_h).transpose(1, 2).contiguous().view(bsz, seq, self.d_model)\n"
        "        return self.out_proj(out)"
    )
    assert _assembled_parses(body)


def test_deeply_nested_blocks_preserved():
    body = (
        "        for i in range(3):\n"
        "            if i:\n"
        "                x = i\n"
        "            else:\n"
        "                x = 0\n"
        "        return x"
    )
    assert _assembled_parses(body)


def test_empty_body_yields_pass_stub():
    src = assemble_module(_FAKE_HEADER, "")
    assert "        pass" in src
    # Should still parse cleanly.
    ast.parse(src)


def test_whitespace_only_yields_pass_stub():
    src = assemble_module(_FAKE_HEADER, "   \n\n   \n")
    assert "pass" in src


@pytest.mark.parametrize(
    "leading_chars",
    ["", "    ", "  ", "        "],
)
def test_various_leading_indents_assemble(leading_chars):
    body = f"{leading_chars}bsz = q.shape[0]\n{leading_chars}return q"
    assert _assembled_parses(body)
