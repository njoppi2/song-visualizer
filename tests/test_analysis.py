from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from songviz.analyze import analyze_file


def _write_test_wav(path: Path, *, sr: int = 22050, duration_s: float = 1.25) -> None:
    n = int(sr * duration_s)
    t = np.arange(n, dtype=np.float32) / sr

    # Simple tone + a few deterministic "clicks" to create onsets.
    y = 0.08 * np.sin(2 * np.pi * 440.0 * t)
    for click_t in (0.20, 0.45, 0.70, 0.95, 1.10):
        i = int(click_t * sr)
        y[i : i + 8] += 0.9

    sf.write(path, y, sr)


def test_analyze_required_keys_and_lengths(tmp_path: Path) -> None:
    wav = tmp_path / "test.wav"
    _write_test_wav(wav)

    a = analyze_file(wav)
    for k in ("beats", "envelopes", "meta"):
        assert k in a

    assert "song_id" in a["meta"]
    assert "duration_s" in a["meta"]
    assert "sample_rate" in a["meta"]

    assert "tempo_bpm" in a["beats"]
    assert "beat_times_s" in a["beats"]

    env = a["envelopes"]
    for k in ("hop_s", "times_s", "loudness", "onset_strength"):
        assert k in env

    n = len(env["times_s"])
    assert n == len(env["loudness"])
    assert n == len(env["onset_strength"])
    assert n > 0

    onset = np.asarray(env["onset_strength"], dtype=np.float32)
    assert float(onset.min()) >= -1e-6
    assert float(onset.max()) <= 1.0 + 1e-6


