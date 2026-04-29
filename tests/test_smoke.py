"""Smoke test: package imports cleanly and exposes a version string."""

from __future__ import annotations

import aero_eval


def test_version_is_set():
    assert isinstance(aero_eval.__version__, str)
    assert len(aero_eval.__version__) > 0


def test_version_in_all():
    assert "__version__" in aero_eval.__all__
