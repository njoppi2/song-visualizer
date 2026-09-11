from __future__ import annotations

import numpy as np
import pytest

from songviz.structure_grid import prepare_beat_grid


SR = 1000
HOP = 10
N_FRAMES = 201
Y = np.zeros(SR * 2, dtype=np.float32)
EXTERNAL_BEATS = [0.137, 0.537, 0.937, 1.337, 1.737]


def _external_grid(beats: list[float] = EXTERNAL_BEATS) -> dict:
    return prepare_beat_grid(
        Y,
        SR,
        hop_length=HOP,
        n_frames=N_FRAMES,
        beat_times_s=beats,
        source="test-provided-beats",
    )


def test_external_beats_preserve_requested_times_and_record_quantized_endpoints() -> None:
    grid = _external_grid()

    assert grid["source"] == "test-provided-beats"
    assert grid["requested_times_s"] == EXTERNAL_BEATS
    assert grid["frame_indices"] == [0, 13, 53, 93, 133, 173, N_FRAMES - 1]
    assert grid["effective_times_s"] == [0.0, 0.13, 0.53, 0.93, 1.33, 1.73, 2.0]
    assert grid["fallback"] is None
    assert grid["endpoint_policy"]


def test_requested_hash_changes_with_exact_external_request() -> None:
    first = _external_grid()
    changed = _external_grid([0.138, 0.537, 0.937, 1.337, 1.737])

    assert isinstance(first["requested_sha256"], str)
    assert len(first["requested_sha256"]) == 64
    assert first["requested_sha256"] != changed["requested_sha256"]
    # A sub-frame request change is still recorded in the requested hash even
    # when the feature-grid quantization leaves the effective grid unchanged.
    assert first["effective_sha256"] == changed["effective_sha256"]


def test_external_beats_never_invoke_tracker(monkeypatch: pytest.MonkeyPatch) -> None:
    import songviz.structure_grid as structure_grid

    def tracker_must_not_run(*args: object, **kwargs: object) -> object:
        raise AssertionError("external beat times must bypass the tracker")

    monkeypatch.setattr(structure_grid.librosa.beat, "beat_track", tracker_must_not_run)

    grid = _external_grid()

    assert grid["requested_times_s"] == EXTERNAL_BEATS
    assert grid["fallback"] is None


def test_short_internal_tracker_grid_reports_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    import songviz.structure_grid as structure_grid

    monkeypatch.setattr(
        structure_grid.librosa.beat,
        "beat_track",
        lambda **kwargs: (120.0, np.array([14, 54, 94], dtype=int)),
    )

    grid = prepare_beat_grid(Y, SR, hop_length=HOP, n_frames=N_FRAMES)

    assert grid["fallback"] == "short_tracker_grid"
    assert grid["frame_indices"][0] == 0
    assert grid["frame_indices"][-1] == N_FRAMES - 1


def test_story_preserves_provided_grid_without_tracking(monkeypatch: pytest.MonkeyPatch) -> None:
    from songviz.story import compute_story
    import songviz.structure_grid as structure_grid

    def tracker_must_not_run(*args: object, **kwargs: object) -> object:
        raise AssertionError("provided story beats must bypass the tracker")

    monkeypatch.setattr(structure_grid.librosa.beat, "beat_track", tracker_must_not_run)
    story_sr = 22050
    t = np.arange(story_sr * 2, dtype=np.float32) / story_sr
    y = (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    story = compute_story(
        y,
        story_sr,
        hop_length=512,
        frame_length=1024,
        beat_times_s=EXTERNAL_BEATS,
        beat_grid_source="story-fixture",
    )

    assert story["meta"]["beat_grid"]["requested_times_s"] == EXTERNAL_BEATS
    assert story["meta"]["beat_grid"]["source"] == "story-fixture"


def test_story_section_fallback_preserves_supplied_grid_without_tracking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import songviz.story as story_module
    import songviz.structure_grid as structure_grid

    def tracker_must_not_run(*args: object, **kwargs: object) -> object:
        raise AssertionError("provided story beats must bypass the tracker")

    def forced_beat_sync_failure(*args: object, **kwargs: object) -> object:
        raise RuntimeError("forced beat-sync failure")

    monkeypatch.setattr(structure_grid.librosa.beat, "beat_track", tracker_must_not_run)
    monkeypatch.setattr(story_module, "_beat_sync_features", forced_beat_sync_failure)
    story_sr = 22050
    t = np.arange(story_sr * 2, dtype=np.float32) / story_sr
    y = (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    story = story_module.compute_story(
        y,
        story_sr,
        hop_length=512,
        frame_length=1024,
        beat_times_s=EXTERNAL_BEATS,
        beat_grid_source="story-fixture",
    )

    assert story["meta"]["beat_grid"]["requested_times_s"] == EXTERNAL_BEATS
    assert story["meta"]["beat_grid"]["source"] == "story-fixture"
    assert story["meta"]["section_method"] == "fallback"
    assert story["meta"]["section_error"] == "RuntimeError: forced beat-sync failure"


def test_story_malformed_supplied_grid_raises_before_section_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from songviz.story import compute_story
    import songviz.structure_grid as structure_grid

    def tracker_must_not_run(*args: object, **kwargs: object) -> object:
        raise AssertionError("malformed supplied beats must not invoke the tracker")

    monkeypatch.setattr(structure_grid.librosa.beat, "beat_track", tracker_must_not_run)
    story_sr = 22050
    y = np.zeros(story_sr * 2, dtype=np.float32)

    with pytest.raises(ValueError):
        compute_story(
            y,
            story_sr,
            hop_length=512,
            frame_length=1024,
            beat_times_s=[0.537, 0.137],
            beat_grid_source="story-fixture",
        )


@pytest.mark.parametrize(
    ("beat_times_s", "n_frames"),
    [
        ([0.137], N_FRAMES),
        ([0.137, 0.137], N_FRAMES),
        ([0.137, float("nan")], N_FRAMES),
        ([0.537, 0.137], N_FRAMES),
        ([-0.01, 0.137], N_FRAMES),
        ([0.137, 2.0], N_FRAMES),
        ([0.137, 0.139], N_FRAMES),
        ([[0.137, 0.537]], N_FRAMES),
        ([0.137, 1.237], 100),
    ],
    ids=[
        "fewer_than_two",
        "duplicate",
        "nan",
        "unsorted",
        "negative",
        "out_of_duration",
        "frame_collision",
        "not_1d",
        "outside_frame_coverage",
    ],
)
def test_invalid_external_beats_raise_without_tracker_fallback(
    monkeypatch: pytest.MonkeyPatch, beat_times_s: object, n_frames: int
) -> None:
    import songviz.structure_grid as structure_grid

    def tracker_must_not_run(*args: object, **kwargs: object) -> object:
        raise AssertionError("invalid external beat times must raise before tracker fallback")

    monkeypatch.setattr(structure_grid.librosa.beat, "beat_track", tracker_must_not_run)

    with pytest.raises(ValueError):
        prepare_beat_grid(
            Y,
            SR,
            hop_length=HOP,
            n_frames=n_frames,
            beat_times_s=beat_times_s,
        )
