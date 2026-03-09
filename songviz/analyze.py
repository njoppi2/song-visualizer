from __future__ import annotations

from pathlib import Path
from typing import Any

import librosa
import numpy as np

from .ingest import song_id_for_path, _utc_now_iso, _normalize_01
from .story import compute_story


def analyze_audio(
    y: np.ndarray,
    sr: int,
    *,
    hop_length: int = 512,
    frame_length: int = 2048,
) -> dict[str, Any]:
    if y.ndim != 1:
        raise ValueError(f"Expected mono audio (1D array), got shape={y.shape}")

    duration_s = float(len(y) / sr) if sr > 0 else 0.0

    tempo_bpm, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    beat_times_s = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)

    loudness = librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
    )[0]
    onset_strength = librosa.onset.onset_strength(
        y=y,
        sr=sr,
        hop_length=hop_length,
    )

    n = int(min(len(loudness), len(onset_strength)))
    loudness = loudness[:n]
    onset_strength = onset_strength[:n]

    times_s = librosa.frames_to_time(np.arange(n), sr=sr, hop_length=hop_length)
    hop_s = float(hop_length / sr)

    tempo_bpm_f = float(np.asarray(tempo_bpm).reshape(-1)[0]) if np.isfinite(tempo_bpm).all() else 0.0

    return {
        "beats": {
            "tempo_bpm": tempo_bpm_f,
            "beat_times_s": beat_times_s.astype(float).tolist(),
        },
        "envelopes": {
            "hop_s": hop_s,
            "times_s": times_s.astype(float).tolist(),
            "loudness": _normalize_01(loudness).astype(float).tolist(),
            "onset_strength": _normalize_01(onset_strength).astype(float).tolist(),
        },
        "meta": {
            "duration_s": duration_s,
            "sample_rate": int(sr),
            "created_at": _utc_now_iso(),
        },
    }


def analyze_file(
    audio_path: str | Path,
    *,
    target_sr: int = 22050,
    mono: bool = True,
    hop_length: int = 512,
    frame_length: int = 2048,
) -> dict[str, Any]:
    p = Path(audio_path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    if not p.is_file():
        raise ValueError(f"Not a file: {p}")

    song_id = song_id_for_path(p)

    y, sr = librosa.load(p, sr=target_sr, mono=mono)
    y = np.asarray(y, dtype=np.float32)

    analysis = analyze_audio(y, sr, hop_length=hop_length, frame_length=frame_length)
    # Optional "story" signals (segmentation + tension). Kept lightweight so v0 stays fast.
    analysis["story"] = compute_story(y, sr, hop_length=hop_length, frame_length=frame_length)
    analysis["meta"]["song_id"] = song_id
    return analysis


