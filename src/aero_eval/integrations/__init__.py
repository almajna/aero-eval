"""Integration adapters for popular training frameworks.

Each integration is gated behind an optional install extra so that ``aero-eval``
does not pull heavy dependencies for users who only want the bare check API.

Example:
    >>> from aero_eval.integrations.hf_trainer import AeroEvalCallback  # doctest: +SKIP
"""

from __future__ import annotations

__all__: list[str] = []
