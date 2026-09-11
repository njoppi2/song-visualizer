"""Check the source clock and unquantized event playback used for human review."""
import numpy as np
from PIL import Image
import pytest
import soundfile as sf

from experiments import build_review as review
from songviz.render import RenderConfig


def test_audio_excerpt_keeps_source_samples_and_stereo(tmp_path):
    samples = np.arange(4000, dtype=np.float32).reshape(2000, 2) / 4000
    path = tmp_path / "source.wav"
    sf.write(path, samples, 1000, subtype="FLOAT")
    cut, sr = review.cut_audio(path, 0.123, 0.987)
    assert sr == 1000
    np.testing.assert_array_equal(cut, samples[123:987])
    with pytest.raises(ValueError, match="outside"):
        review.cut_audio(path, 1, 3)


def test_stereo_rms_does_not_cancel_opposite_phase_channels():
    samples = np.tile([0.5, -0.5], (100, 1))
    times, values = review.rms_curve(samples, 1000, 130)
    np.testing.assert_allclose(values, 0.5)
    np.testing.assert_allclose(times, [130, 130.05])


def test_drum_audition_keeps_off_grid_time_and_tail_at_excerpt_boundary():
    reduced = {"drums": {"hits": [{"t": 10.123, "component": "kick", "velocity": 1}]}}
    audio = review.drum_clicks(reduced, 10, 11, sr=1000)
    assert not np.any(audio[:123])
    assert np.any(audio[124:163])
    assert not np.any(audio[163:])
    tail = review.drum_clicks(reduced, 10.13, 10.2, sr=1000)
    assert np.any(tail[:20])


def test_visuals_use_absolute_song_time_while_video_starts_at_zero(monkeypatch):
    seen = []
    class FakeVisualizer:
        def __init__(self, analysis, cfg):
            self.cfg = cfg
        def frame_rgb24(self, t):
            seen.append(t)
            return Image.new("RGB", (self.cfg.width, self.cfg.height)).tobytes()
    monkeypatch.setattr(review, "Visualizer", FakeVisualizer)
    cfg = RenderConfig(width=640, height=540)
    viz = review.ReviewVisualizer({}, {}, cfg, 130, 148, {})
    assert len(viz.frame_rgb24(0)) == 640 * 540 * 3
    viz.frame_rgb24(1.25)
    assert seen == [130, 131.25]
