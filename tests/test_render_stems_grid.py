from __future__ import annotations

import numpy as np

from songviz.render import RenderConfig, StemGridVisualizer


def _fake_analysis(*, beat_times: list[float]) -> dict:
    times = np.linspace(0.0, 1.0, 32, dtype=np.float32)
    loud = np.clip(np.sin(times * 3.0) * 0.5 + 0.5, 0.0, 1.0)
    onset = np.clip(np.cos(times * 5.0) * 0.5 + 0.5, 0.0, 1.0)
    return {
        "beats": {"tempo_bpm": 120.0, "beat_times_s": beat_times},
        "envelopes": {
            "hop_s": 512 / 22050,
            "times_s": times.astype(float).tolist(),
            "loudness": loud.astype(float).tolist(),
            "onset_strength": onset.astype(float).tolist(),
        },
        "meta": {"duration_s": 1.0, "sample_rate": 22050, "created_at": "1970-01-01T00:00:00+00:00"},
    }


def test_stems_grid_frame_is_deterministic() -> None:
    cfg = RenderConfig(width=320, height=240, fps=30, seed=123)
    stems = {
        "drums": _fake_analysis(beat_times=[0.2, 0.4, 0.6, 0.8]),
        "bass": _fake_analysis(beat_times=[]),
        "vocals": _fake_analysis(beat_times=[]),
        "other": _fake_analysis(beat_times=[]),
    }

    v1 = StemGridVisualizer(stems, cfg)
    v2 = StemGridVisualizer(stems, cfg)

    b1 = v1.frame_rgb24(0.5)
    b2 = v2.frame_rgb24(0.5)

    assert len(b1) == cfg.width * cfg.height * 3
    assert b1 == b2

