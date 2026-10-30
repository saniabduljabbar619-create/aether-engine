"""Optional dependencies in tests.

On a developer machine a missing optional piece (PySide6, OpenCV, the face models) skips the tests that need it.
In CI (AETHER_REQUIRE_ALL=1) it must fail loudly instead: a green check that silently skipped half the
suite would be claiming coverage that no longer exists.
"""
import importlib
import os

import pytest

REQUIRE_ALL = os.environ.get("AETHER_REQUIRE_ALL") == "1"


def optional_module(name: str):
    """The module, or a skip when it is missing (an ImportError when everything is required)."""
    if REQUIRE_ALL:
        return importlib.import_module(name)
    return pytest.importorskip(name)


def skip_unless(available: bool, reason: str):
    return pytest.mark.skipif(not available and not REQUIRE_ALL, reason=reason)
