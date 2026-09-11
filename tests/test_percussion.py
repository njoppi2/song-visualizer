"""Synthetic timing tests for the off-grid component attack sidecar."""
from __future__ import annotations

import numpy as np

from songviz.percussion import extract_component_attacks


SR = 16_000


def _decaying_hit(y: np.ndarray, t: float, *, amplitude: float = 0.8, decay_s: float = 0.06) -> None:
    """Add a bipolar, oscillatory drum-like attack at a known non-grid time."""
    start = int(round(t * SR))
    n = min(len(y) - start, int(round(decay_s * SR)))
    if n <= 0:
        return
    samples = np.arange(n, dtype=np.float32)
    # The carrier crosses zero and alternates polarity, unlike a DC test pulse.
    carrier = np.sin(2 * np.pi * 210 * samples / SR)
    y[start : start + n] += amplitude * carrier * np.exp(-samples / (SR * 0.012))


def _component(times: list[float], duration_s: float = 1.5, **kwargs: float) -> np.ndarray:
    y = np.zeros(int(duration_s * SR), dtype=np.float32)
    for time in times:
        _decaying_hit(y, time, **kwargs)
    return y


def _all_hits(result: dict[str, object]) -> list[dict[str, object]]:
    return list(result["hits"]) + list(result["faint_hits"])  # type: ignore[arg-type]


def _times_for(result: dict[str, object], component: str) -> list[float]:
    return [float(hit["t"]) for hit in _all_hits(result) if hit["component"] == component]


def test_syncopated_attacks_keep_offgrid_times_within_20ms() -> None:
    expected = [0.137, 0.418, 0.863, 1.191]
    result = extract_component_attacks({"kick": _component(expected)}, SR)
    detected = _times_for(result, "kick")

    assert len(detected) == len(expected)
    for time in expected:
        assert any(abs(found - time) <= 0.020 for found in detected), (time, detected)
    # This guards against an accidental beat-template or fixed-grid stage.
    assert any(abs(found - round(found * 4) / 4) > 0.03 for found in detected)


def test_simultaneous_components_remain_distinct_hits() -> None:
    result = extract_component_attacks(
        {"kick": _component([0.317]), "snare": _component([0.317])}, SR,
    )
    hits = result["hits"]

    assert {hit["component"] for hit in hits} == {"kick", "snare"}
    assert all(abs(float(hit["t"]) - 0.317) <= 0.020 for hit in hits)


def test_silence_and_unknown_components_produce_no_attacks() -> None:
    result = extract_component_attacks(
        {"kick": np.zeros(SR, dtype=np.float32), "cowbell": _component([0.4])}, SR,
    )

    assert result["source"] == "component_attacks_v1"
    assert result["hits"] == []
    assert result["faint_hits"] == []
    assert result["parameters"]["component_thresholds"]["kick"]["hits"] == 0


def test_long_decay_does_not_repeat_and_quiet_later_attack_survives() -> None:
    y = _component([0.221], duration_s=1.3, amplitude=0.8, decay_s=0.35)
    _decaying_hit(y, 0.872, amplitude=0.08, decay_s=0.08)
    result = extract_component_attacks({"crash": y}, SR)
    detected = _times_for(result, "crash")

    assert len(detected) == 2, detected
    assert any(abs(time - 0.221) <= 0.020 for time in detected)
    assert any(abs(time - 0.872) <= 0.020 for time in detected)


def test_schema_reports_detector_measurements_not_grid_data() -> None:
    result = extract_component_attacks({"hh": _component([0.103, 0.177])}, SR)

    assert set(result) == {"source", "hits", "faint_hits", "parameters"}
    assert result["parameters"]["hop_seconds"] <= 0.0041
    assert "refractory_seconds" in result["parameters"]
    for hit in _all_hits(result):
        assert set(hit) == {"t", "component", "velocity", "strength", "peak_rms"}
        assert isinstance(hit["strength"], float)
        assert isinstance(hit["peak_rms"], float)
        threshold = result["parameters"]["component_thresholds"][hit["component"]]["faint_strength_threshold"]
        # strength remains the raw RMS-rise evidence; thresholding is explicit
        # in parameters instead of hidden by any beat/grid-derived rule.
        assert hit["strength"] + 1e-7 >= threshold
        assert 0.0 <= hit["velocity"] <= 1.0


def test_background_noise_is_not_promoted_to_prominent_hits() -> None:
    rng = np.random.default_rng(19)
    noise = rng.normal(0.0, 0.003, SR * 2).astype(np.float32)
    result = extract_component_attacks({"snare": noise}, SR)

    assert result["hits"] == []
