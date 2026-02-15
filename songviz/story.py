from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import librosa
import numpy as np


def _normalize_01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    mn = float(np.nanmin(x))
    mx = float(np.nanmax(x))
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def _smooth_1d(x: np.ndarray, win: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    win = int(max(1, win))
    if win <= 1:
        return x
    k = np.ones((win,), dtype=np.float32) / float(win)
    return np.convolve(x, k, mode="same").astype(np.float32)


def _merge_short_segments(bounds_s: list[float], *, min_len_s: float, duration_s: float) -> list[float]:
    # bounds_s includes 0 and duration, strictly increasing.
    if not bounds_s:
        return [0.0, float(duration_s)]
    b = [float(x) for x in bounds_s]
    b[0] = 0.0
    b[-1] = float(duration_s)
    out = [b[0]]
    for i in range(1, len(b) - 1):
        prev = out[-1]
        cur = b[i]
        nxt = b[i + 1]
        if (cur - prev) < min_len_s or (nxt - cur) < min_len_s:
            # Skip this boundary: merge with neighbor.
            continue
        out.append(cur)
    out.append(b[-1])
    # Ensure strictly increasing and unique-ish.
    out2: list[float] = []
    for x in out:
        if not out2 or x > out2[-1] + 1e-3:
            out2.append(x)
    if out2[-1] < duration_s:
        out2[-1] = float(duration_s)
    return out2


def _labels_for_n(n: int) -> list[str]:
    # A, B, C ... Z, AA, AB ...
    out: list[str] = []
    i = 0
    while len(out) < n:
        x = i
        s = ""
        while True:
            s = chr(ord("A") + (x % 26)) + s
            x = (x // 26) - 1
            if x < 0:
                break
        out.append(s)
        i += 1
    return out


def compute_story(
    y: np.ndarray,
    sr: int,
    *,
    hop_length: int = 512,
    frame_length: int = 2048,
) -> dict[str, Any]:
    """
    Heuristic "song story" signals:
    - sections: coarse segmentation (A/B/C...) from timbre features (MFCC).
    - tension: a smooth energy/brightness curve (0..1), good for buildup/drop dynamics.

    This is intentionally lightweight and fully deterministic.
    """
    if y.ndim != 1:
        raise ValueError(f"Expected mono audio (1D array), got shape={y.shape}")
    if sr <= 0:
        raise ValueError(f"Invalid sample_rate={sr}")

    duration_s = float(len(y) / sr)

    # --- Per-frame features (aligned to our analysis hop) ---
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length, center=True)[0]
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]

    n = int(min(rms.size, onset.size, centroid.size))
    rms = rms[:n]
    onset = onset[:n]
    centroid = centroid[:n]
    times_s = librosa.frames_to_time(np.arange(n), sr=sr, hop_length=hop_length).astype(np.float32)

    rms01 = _normalize_01(rms)
    onset01 = _normalize_01(onset)
    cent01 = _normalize_01(centroid)

    tension = 0.48 * (rms01**0.8) + 0.34 * (onset01**0.7) + 0.18 * (cent01**0.9)
    win = max(5, int(round(0.35 / max(1e-6, (hop_length / sr)))))
    tension = _normalize_01(_smooth_1d(tension, win=win))

    # --- Coarse segmentation (timbre-based) ---
    # MFCC tends to separate verses/choruses reasonably well, even without harmonic analysis.
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=hop_length, n_fft=frame_length)
    # Standardize per coefficient to reduce scale dominance.
    mu = np.mean(mfcc, axis=1, keepdims=True)
    sig = np.std(mfcc, axis=1, keepdims=True) + 1e-6
    mfcc_z = (mfcc - mu) / sig

    # Choose a small number of clusters based on duration (coarse "chapters").
    # Typical pop songs: ~4-10 sections.
    k = int(np.clip(round(duration_s / 25.0), 4, 10))
    try:
        seg_labels = librosa.segment.agglomerative(mfcc_z, k=k)
    except Exception:
        seg_labels = np.zeros((mfcc_z.shape[1],), dtype=int)

    # Boundaries are where labels change.
    change = np.flatnonzero(np.diff(seg_labels) != 0) + 1
    bounds_frames = np.concatenate([[0], change.astype(int), [int(seg_labels.size)]])
    bounds_s = librosa.frames_to_time(bounds_frames, sr=sr, hop_length=hop_length).astype(float).tolist()
    bounds_s = _merge_short_segments(bounds_s, min_len_s=7.0, duration_s=duration_s)

    sections: list[dict[str, Any]] = []
    labels = _labels_for_n(max(1, len(bounds_s) - 1))
    for i in range(len(bounds_s) - 1):
        s0 = float(bounds_s[i])
        s1 = float(bounds_s[i + 1])
        sections.append(
            {
                "label": labels[i],
                "start_s": s0,
                "end_s": s1,
            }
        )

    # Events: simple "drop" detection via sharp tension decrease.
    dt = float(hop_length / sr)
    d = np.diff(tension, prepend=float(tension[0])) / max(dt, 1e-6)
    drops = np.flatnonzero(d < -0.65)
    drop_times = times_s[drops].astype(float).tolist()[:25]  # cap spam

    return {
        "sections": sections,
        "tension": {
            "hop_s": float(hop_length / sr),
            "times_s": times_s.astype(float).tolist(),
            "value": tension.astype(float).tolist(),
        },
        "events": {
            "drop_times_s": drop_times,
        },
        "meta": {
            "duration_s": float(duration_s),
            "sample_rate": int(sr),
            "features": ["mfcc_20", "rms", "onset_strength", "spectral_centroid"],
        },
    }
