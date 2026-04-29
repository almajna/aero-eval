"""Canonical result type for all aero-eval checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CheckResult:
    """Result of running a single check.

    Attributes:
        name: Short identifier of the check (e.g. ``"nan_inf"``).
        passed: ``True`` if the model behaved correctly under this check.
        score: A scalar in ``[0, 1]`` summarizing severity. Higher = healthier.
            Checks that are inherently binary (passed/failed) report ``1.0`` or
            ``0.0`` here. Continuous checks (e.g. dead-neuron ratio) report a
            graded score.
        message: One-line human-readable summary suitable for logs.
        evidence: Free-form structured payload — layer names that triggered,
            tensor stats, step indices, etc. Used by the dashboard and by
            downstream auto-patching loops. Keep it JSON-serializable.
    """

    name: str
    passed: bool
    score: float
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(
                f"CheckResult.score must be in [0, 1], got {self.score!r} for {self.name!r}"
            )

    def __bool__(self) -> bool:
        """Allow ``if result: ...`` shorthand for the pass/fail bit."""
        return self.passed
