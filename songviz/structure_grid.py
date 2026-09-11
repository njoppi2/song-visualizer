"""Explicit, inspectable beat grids for structural feature synchronization."""
from __future__ import annotations

import hashlib
import json

import librosa
import numpy as np


def grid_hash(times: list[float]) -> str:
    """Hash canonical JSON seconds, not platform-specific array bytes."""
    return hashlib.sha256(json.dumps(times, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def prepare_beat_grid(y, sr, *, hop_length, n_frames, beat_times_s=None, source=None):
    if sr <= 0 or hop_length <= 0 or n_frames < 2 or len(y) == 0:
        raise ValueError("Grid needs positive audio/frame coverage")
    fallback = None
    explicit = beat_times_s is not None
    if explicit:
        requested = np.asarray(beat_times_s, dtype=float)
        if requested.ndim != 1 or requested.size < 2 or not np.all(np.isfinite(requested)) or np.any(np.diff(requested) <= 0) or requested[0] < 0 or requested[-1] >= len(y) / sr:
            raise ValueError("Explicit beats must be finite, increasing and within audio duration")
        frames = librosa.time_to_frames(requested, sr=sr, hop_length=hop_length)
        if np.any(np.diff(frames) <= 0):
            raise ValueError("Explicit beats collide at the feature frame resolution")
        if frames[0] < 0 or frames[-1] >= n_frames:
            raise ValueError("Explicit beats exceed feature frame coverage")
        grid_source = source or "provided"
    else:
        grid_source = "librosa.beat.beat_track"
        try:
            _, frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
            requested = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
            if len(frames) < 8:
                fallback = "short_tracker_grid"
        except Exception:
            frames = np.array([], dtype=int)
            requested = np.array([], dtype=float)
            fallback = "tracker_error"
        if fallback:
            frames = np.arange(0, n_frames * hop_length, max(1, sr // 2)) // hop_length
    if not isinstance(grid_source, str) or not grid_source:
        raise ValueError("Grid source must be a nonempty string")
    # Synchronization includes endpoint frames, which are not asserted downbeats.
    effective_frames = librosa.util.fix_frames(frames, x_min=0, x_max=n_frames-1)
    effective_times = librosa.frames_to_time(effective_frames, sr=sr, hop_length=hop_length).astype(float).tolist()
    raw = requested.astype(float).tolist()
    return {"schema_version": 1, "source": grid_source,
            "requested_times_s": raw, "requested_sha256": grid_hash(raw),
            "effective_times_s": effective_times, "effective_sha256": grid_hash(effective_times),
            "frame_indices": effective_frames.astype(int).tolist(), "sample_rate": int(sr),
            "hop_length": int(hop_length), "fallback": fallback,
            "endpoint_policy": "floor seconds to feature frames; insert frames 0 and n_frames-1; endpoints are not downbeat claims",
            "explicit": explicit}
