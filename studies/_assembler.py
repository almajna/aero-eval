"""Shared utilities for stitching LLM-generated forward-method bodies into
runnable .py modules.

Used by both ``studies/baseline/generate.py`` and ``studies/gqa/generate.py``.
Lives at ``studies/_assembler.py`` (underscored to mark non-public).
"""

from __future__ import annotations

import textwrap
from collections import Counter

__all__ = ["assemble_module", "robust_dedent", "strip_fences"]


def robust_dedent(text: str) -> str:
    """Dedent ``text``, robust to a single under-indented first line.

    Real LLM output occasionally produces a first line with one fewer leading
    space than the rest of the body — for example, Opus often emits
    ``       bsz = q.shape\n        a = ...`` (7 spaces, then 8). Plain
    ``textwrap.dedent`` treats the 7 as the common prefix, leaving the rest
    of the body indented by 1 space and producing a ``SyntaxError`` when the
    body is wrapped inside a class method.

    The base indent is selected as:

    1. The smallest indent that appears on at least two non-blank lines.
       Singleton outliers (the under-indented first-line case) are ignored.
    2. If no indent repeats (very short bodies, e.g. a single ``return q``),
       fall back to the largest indent present. Under-indented outliers are
       typically *smaller* than intended, so picking the larger sample is a
       safer estimate than the smaller.

    Lines indented less than the base are clamped to flush-left rather than
    producing negative indent.
    """
    lines = text.splitlines()
    indents = [len(ln) - len(ln.lstrip(" ")) for ln in lines if ln.strip()]
    if not indents:
        return text

    counts = Counter(indents)
    multi = [ind for ind, n in counts.items() if n >= 2]
    base = min(multi) if multi else max(indents)

    out: list[str] = []
    for ln in lines:
        if not ln.strip():
            out.append(ln)
            continue
        leading = len(ln) - len(ln.lstrip(" "))
        if leading >= base:
            out.append(ln[base:])
        else:
            out.append(ln.lstrip(" "))
    return "\n".join(out)


def strip_fences(body: str) -> str:
    """Remove a leading/trailing markdown code fence if present."""
    if not body.lstrip().startswith("```"):
        return body
    lines = body.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and lines[0].lstrip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines)


def assemble_module(class_header: str, forward_body: str) -> str:
    """Stitch ``forward_body`` into ``class_header`` to produce a runnable module.

    Args:
        class_header: A string ending with the line ``    ) -> Tensor:`` and
            a trailing newline. The forward-body will be appended after this,
            indented to 8 spaces so it sits inside the method.
        forward_body: Raw output from the LLM. May be flush-left, indented at
            any uniform level, fenced in markdown, or have an under-indented
            first line.
    """
    body = forward_body.rstrip()

    body_lines = body.splitlines()
    while body_lines and not body_lines[0].strip():
        body_lines.pop(0)
    body = "\n".join(body_lines)

    body = strip_fences(body)

    if not body.strip():
        return class_header + "        pass\n"

    dedented = robust_dedent(body)
    indented = textwrap.indent(dedented, " " * 8)
    return class_header + indented + "\n"
