"""Tests for songviz.sonify — minimal sonifier for reduced.json."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from songviz.sonify import SR, diagnose_reduced, sonify_reduced, sonify_reduced_layers


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), dtype="float32")
    return data, sr


# ── 1. Empty reduced → valid short WAV, essentially silent ──


def test_empty_reduced_produces_short_wav(tmp_path: Path) -> None:
    reduced: dict = {"drums": {"hits": []}, "vocals": {"notes": []}, "bass": {"notes": []}}
    out = tmp_path / "empty.wav"
    sonify_reduced(reduced, out)
    data, sr = _read_wav(out)
    assert sr == SR
    # ~0.5 s minimum + tail → at least 0.5 s
    assert len(data) / sr >= 0.5
    # Essentially silent (peak < small threshold)
    assert float(np.max(np.abs(data))) < 1e-3


# ── 2. Single drum hit → non-silent region near that time ──


def test_single_drum_hit_produces_spike(tmp_path: Path) -> None:
    reduced = {
        "drums": {"hits": [{"t": 0.5, "component": "kick", "velocity": 0.8}]},
        "vocals": {"notes": []},
        "bass": {"notes": []},
    }
    out = tmp_path / "kick.wav"
    sonify_reduced(reduced, out)
    data, _ = _read_wav(out)
    # Region around 0.5 s should be non-silent
    start = int(0.48 * SR)
    end = int(0.58 * SR)
    region = data[start:end]
    assert float(np.max(np.abs(region))) > 0.1


# ── 3. Single vocal note → FFT peak near 440 Hz ──


def test_single_vocal_note_frequency(tmp_path: Path) -> None:
    reduced = {
        "drums": {"hits": []},
        "vocals": {
            "notes": [{"onset_s": 0.0, "offset_s": 1.0, "midi": 69.0, "velocity": 0.8}]
        },
        "bass": {"notes": []},
    }
    out = tmp_path / "vocal.wav"
    sonify_reduced(reduced, out)
    data, sr = _read_wav(out)
    # FFT of the vocal portion
    spectrum = np.abs(np.fft.rfft(data[: int(sr * 1.0)]))
    freqs = np.fft.rfftfreq(int(sr * 1.0), d=1.0 / sr)
    peak_freq = freqs[np.argmax(spectrum)]
    # Should be within ~5 Hz of 440
    assert abs(peak_freq - 440.0) < 5.0


# ── 4. All layers stacked → samples within [-1, 1] ──


def test_output_in_valid_range(tmp_path: Path) -> None:
    reduced = {
        "drums": {
            "hits": [
                {"t": 0.0, "component": "kick", "velocity": 1.0},
                {"t": 0.0, "component": "snare", "velocity": 1.0},
                {"t": 0.0, "component": "crash", "velocity": 1.0},
            ]
        },
        "vocals": {
            "notes": [{"onset_s": 0.0, "offset_s": 1.0, "midi": 69.0, "velocity": 1.0}]
        },
        "bass": {
            "notes": [{"onset_s": 0.0, "offset_s": 1.0, "midi": 45.0, "velocity": 1.0}]
        },
    }
    out = tmp_path / "stacked.wav"
    sonify_reduced(reduced, out)
    data, _ = _read_wav(out)
    assert float(np.max(np.abs(data))) <= 1.0


# ── 5. Duration matches latest event ──


def test_duration_matches_latest_event(tmp_path: Path) -> None:
    reduced = {
        "drums": {"hits": []},
        "vocals": {
            "notes": [{"onset_s": 1.0, "offset_s": 2.5, "midi": 60.0, "velocity": 0.5}]
        },
        "bass": {"notes": []},
    }
    out = tmp_path / "dur.wav"
    sonify_reduced(reduced, out)
    data, sr = _read_wav(out)
    dur = len(data) / sr
    assert dur >= 2.5
    assert dur < 3.5


# ── 6. Roundtrip file written ──


def test_roundtrip_file_written(tmp_path: Path) -> None:
    reduced = {
        "drums": {"hits": [{"t": 0.1, "component": "snare", "velocity": 0.6}]},
        "vocals": {"notes": []},
        "bass": {"notes": []},
    }
    sub = tmp_path / "analysis"
    out = sub / "reduced.wav"
    sonify_reduced(reduced, out)
    assert out.exists()
    # Valid WAV: soundfile can open it
    info = sf.info(str(out))
    assert info.samplerate == SR
    assert info.frames > 0


# ── 7. Bass triangle wave has harmonics ──


def test_bass_triangle_wave_has_harmonics(tmp_path: Path) -> None:
    # A2 = MIDI 45 → 110 Hz; 3rd harmonic at ~330 Hz
    reduced = {
        "drums": {"hits": []},
        "vocals": {"notes": []},
        "bass": {
            "notes": [{"onset_s": 0.0, "offset_s": 1.0, "midi": 45.0, "velocity": 0.8}]
        },
    }
    out = tmp_path / "bass.wav"
    sonify_reduced(reduced, out)
    data, sr = _read_wav(out)
    spectrum = np.abs(np.fft.rfft(data[: int(sr * 1.0)]))
    freqs = np.fft.rfftfreq(int(sr * 1.0), d=1.0 / sr)
    # Find energy near 110 Hz (fundamental) and 330 Hz (3rd harmonic)
    fund_mask = (freqs > 105) & (freqs < 115)
    h3_mask = (freqs > 325) & (freqs < 340)
    fund_energy = float(np.max(spectrum[fund_mask]))
    h3_energy = float(np.max(spectrum[h3_mask]))
    # 3rd harmonic should have meaningful energy (>1% of fundamental)
    assert h3_energy > fund_energy * 0.01


# ── 8. All drum components accepted ──


def test_all_drum_components_accepted(tmp_path: Path) -> None:
    hits = [
        {"t": 0.1, "component": "kick", "velocity": 0.7},
        {"t": 0.3, "component": "snare", "velocity": 0.7},
        {"t": 0.5, "component": "toms", "velocity": 0.7},
        {"t": 0.7, "component": "hh", "velocity": 0.7},
        {"t": 0.9, "component": "ride", "velocity": 0.7},
        {"t": 1.1, "component": "crash", "velocity": 0.7},
    ]
    reduced = {
        "drums": {"hits": hits},
        "vocals": {"notes": []},
        "bass": {"notes": []},
    }
    out = tmp_path / "all_drums.wav"
    sonify_reduced(reduced, out)
    data, _ = _read_wav(out)
    # Non-silent output
    assert float(np.max(np.abs(data))) > 0.1


# ── 9. diagnose_reduced: healthy input → no warnings ──


def test_diagnose_healthy() -> None:
    reduced = {
        "drums": {
            "source": "drumsep",
            "hits": [{"t": i * 0.25, "component": "kick", "velocity": 0.5} for i in range(40)],
        },
        "vocals": {
            "source": "basic_pitch",
            "notes": [
                {"onset_s": i * 0.5, "offset_s": i * 0.5 + 0.4, "midi": 65.0, "velocity": 0.6}
                for i in range(30)
            ],
        },
        "bass": {
            "source": "pyin",
            "notes": [
                {"onset_s": i * 0.5, "offset_s": i * 0.5 + 0.35, "midi": 40.0, "velocity": 0.5}
                for i in range(20)
            ],
        },
    }
    diag = diagnose_reduced(reduced)
    assert diag["warnings"] == []
    assert diag["drums"]["event_count"] == 40
    assert diag["vocals"]["event_count"] == 30
    assert diag["bass"]["event_count"] == 20
    assert diag["vocals"]["coverage_pct"] > 20.0
    assert diag["bass"]["coverage_pct"] > 15.0


# ── 10. diagnose_reduced: sparse vocals → warning ──


def test_diagnose_sparse_vocals() -> None:
    reduced = {
        "drums": {
            "source": "drumsep",
            "hits": [{"t": i * 0.25, "component": "kick", "velocity": 0.5} for i in range(40)],
        },
        "vocals": {
            "source": "basic_pitch",
            "notes": [
                {"onset_s": 0.0, "offset_s": 0.3, "midi": 65.0, "velocity": 0.6},
            ],
        },
        "bass": {
            "source": "pyin",
            "notes": [
                {"onset_s": i * 0.5, "offset_s": i * 0.5 + 0.35, "midi": 40.0, "velocity": 0.5}
                for i in range(20)
            ],
        },
    }
    diag = diagnose_reduced(reduced)
    assert "sparse_vocals" in diag["warnings"]


# ── 11. diagnose_reduced: drums dominate + basic_pitch failed ──


def test_diagnose_drums_dominate_and_basic_pitch_failed() -> None:
    reduced = {
        "drums": {
            "source": "drumsep",
            "hits": [{"t": i * 0.05, "component": "kick", "velocity": 0.9} for i in range(200)],
        },
        "vocals": {"source": "pyin", "notes": []},
        "bass": {"source": "pyin", "notes": []},
    }
    diag = diagnose_reduced(reduced)
    assert "drums_dominate" in diag["warnings"]
    assert "basic_pitch_unavailable_or_failed" in diag["warnings"]
    assert "sparse_vocals" in diag["warnings"]


# ── 12. sonify_reduced_layers writes all debug WAVs ──


def test_sonify_reduced_layers_writes_files(tmp_path: Path) -> None:
    reduced = {
        "drums": {"hits": [{"t": 0.1, "component": "kick", "velocity": 0.7}]},
        "vocals": {
            "notes": [{"onset_s": 0.0, "offset_s": 0.5, "midi": 65.0, "velocity": 0.6}]
        },
        "bass": {
            "notes": [{"onset_s": 0.0, "offset_s": 0.5, "midi": 40.0, "velocity": 0.5}]
        },
    }
    paths = sonify_reduced_layers(reduced, tmp_path)
    assert set(paths.keys()) == {"drums_only", "vocals_only", "bass_only", "vocals_plus_bass", "bass_only_up1oct"}
    for name, p in paths.items():
        assert p.exists(), f"{name} not written"
        info = sf.info(str(p))
        assert info.frames > 0


# ── 13. basic-pitch ONNX model path exists when installed ──


@pytest.mark.skipif(
    not pytest.importorskip("basic_pitch", reason="basic-pitch not installed"),
    reason="basic-pitch not installed",
)
def test_basic_pitch_onnx_model_exists() -> None:
    """When basic-pitch is installed, the ONNX model file should be present."""
    import basic_pitch as _bp

    onnx_path = Path(_bp.__file__).parent / "saved_models" / "icassp_2022" / "nmp.onnx"
    assert onnx_path.exists(), f"ONNX model not found at {onnx_path}"
