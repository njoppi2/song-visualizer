"""Tests for DrumSep integration (features + stems)."""
from __future__ import annotations

import numpy as np
import pytest

from songviz.features import drums_band_energy_3, drums_band_energy_3_from_components
from songviz.stems import _audio_separator_available, _DRUMSEP_COMPONENTS


def _make_component_audio(sr: int = 22050, duration_s: float = 1.0) -> dict[str, np.ndarray]:
    """Synthetic drum component signals for testing."""
    n = int(sr * duration_s)
    t = np.arange(n, dtype=np.float32) / sr
    return {
        "kick": 0.8 * np.sin(2 * np.pi * 80 * t).astype(np.float32),
        "snare": 0.5 * np.sin(2 * np.pi * 500 * t).astype(np.float32),
        "toms": 0.3 * np.sin(2 * np.pi * 300 * t).astype(np.float32),
        "hh": 0.4 * np.sin(2 * np.pi * 8000 * t).astype(np.float32),
        "ride": 0.2 * np.sin(2 * np.pi * 6000 * t).astype(np.float32),
        "crash": 0.1 * np.sin(2 * np.pi * 10000 * t).astype(np.float32),
    }


def test_drumsep_energy_shape_matches_heuristic() -> None:
    """Component-based energy has the same (N, 3) shape as the heuristic."""
    sr = 22050
    comps = _make_component_audio(sr=sr)
    # Mix all components for heuristic
    n = min(len(v) for v in comps.values())
    mix = sum(v[:n] for v in comps.values())

    heuristic = drums_band_energy_3(mix, sr)
    from_comps = drums_band_energy_3_from_components(comps, sr)

    assert heuristic.shape[1] == 3
    assert from_comps.shape[1] == 3
    assert heuristic.shape[0] == from_comps.shape[0]


def test_drumsep_energy_normalized_0_1() -> None:
    """Output values are in [0, 1] range."""
    comps = _make_component_audio()
    energy = drums_band_energy_3_from_components(comps, 22050)
    assert energy.min() >= 0.0
    assert energy.max() <= 1.0


def test_drumsep_energy_kick_dominant_in_band_0() -> None:
    """When only kick has signal, band 0 should be active and others near zero."""
    sr = 22050
    n = int(sr * 0.5)
    comps = {
        "kick": 0.8 * np.sin(2 * np.pi * 80 * np.arange(n, dtype=np.float32) / sr).astype(np.float32),
        "snare": np.zeros(n, dtype=np.float32),
        "toms": np.zeros(n, dtype=np.float32),
        "hh": np.zeros(n, dtype=np.float32),
        "ride": np.zeros(n, dtype=np.float32),
        "crash": np.zeros(n, dtype=np.float32),
    }
    energy = drums_band_energy_3_from_components(comps, sr)
    assert energy[:, 0].mean() > 0.1
    assert energy[:, 1].mean() < 1e-6
    assert energy[:, 2].mean() < 1e-6


def test_drumsep_energy_empty_input() -> None:
    """Empty components should produce empty output."""
    comps = {k: np.zeros(0, dtype=np.float32) for k in ("kick", "snare", "toms", "hh", "ride", "crash")}
    energy = drums_band_energy_3_from_components(comps, 22050)
    assert energy.shape == (0, 3)


def test_drumsep_components_constant() -> None:
    """The expected component names should be stable."""
    assert set(_DRUMSEP_COMPONENTS) == {"kick", "snare", "toms", "hh", "ride", "crash"}


def test_audio_separator_available_returns_bool() -> None:
    """_audio_separator_available should return a bool without raising."""
    result = _audio_separator_available()
    assert isinstance(result, bool)
