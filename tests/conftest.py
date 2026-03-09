"""Test configuration: prevent tests from loading real ML backends by default."""
from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _no_real_ml_backends():
    """Disable whisperx and stable_whisper in all tests by default.

    Tests that want to exercise a specific backend can override by explicitly
    patching _resolve_backend_order or mocking the relevant *_align function.
    """
    with (
        patch("songviz.lyrics._whisperx_available", return_value=False),
        patch("songviz.lyrics._stable_whisper_available", return_value=False),
    ):
        yield
