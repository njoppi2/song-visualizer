from __future__ import annotations

from typing import Final

import numpy as np
import librosa


_DEFAULT_FMIN_HZ: Final[float] = float(librosa.note_to_hz("C2"))
_DEFAULT_FMAX_HZ: Final[float] = float(librosa.note_to_hz("C7"))


def vocals_pitch_hz(
    y: np.ndarray,
    sr: int,
    *,
    hop_length: int = 512,
    frame_length: int = 2048,
    fmin_hz: float = _DEFAULT_FMIN_HZ,
    fmax_hz: float = _DEFAULT_FMAX_HZ,
) -> np.ndarray:
    """
    Pitch track (Hz) for vocals visualization.

    We use YIN (fast) + an RMS-based gate to mark unvoiced frames as NaN.
    """
    if y.ndim != 1:
        raise ValueError(f"Expected mono audio (1D array), got shape={y.shape}")
    if sr <= 0:
        return np.zeros((0,), dtype=np.float32)

    f0 = librosa.yin(
        y=y,
        fmin=float(fmin_hz),
        fmax=float(fmax_hz),
        sr=int(sr),
        frame_length=int(frame_length),
        hop_length=int(hop_length),
    ).astype(np.float32, copy=False)

    rms = librosa.feature.rms(
        y=y,
        frame_length=int(frame_length),
        hop_length=int(hop_length),
        center=True,
    )[0].astype(np.float32, copy=False)

    n = int(min(f0.size, rms.size))
    if n <= 0:
        return np.zeros((0,), dtype=np.float32)
    f0 = f0[:n]
    rms = rms[:n]

    # Gate based on relative energy to avoid pitch scribbles in silence.
    thr = float(np.percentile(rms, 25)) * 0.8
    thr = max(thr, float(rms.max()) * 0.08, 1e-6)
    voiced = rms >= thr

    out = np.where(voiced, f0, np.nan).astype(np.float32)
    return out


def other_chroma_12(
    y: np.ndarray,
    sr: int,
    *,
    hop_length: int = 512,
    n_fft: int = 2048,
) -> np.ndarray:
    """
    Chroma feature (N, 12) for harmonic-ish visualization.

    Output is per-frame normalized to [0, 1] (by max per frame).
    """
    if y.ndim != 1:
        raise ValueError(f"Expected mono audio (1D array), got shape={y.shape}")
    if sr <= 0:
        return np.zeros((0, 12), dtype=np.float32)

    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=int(sr),
        hop_length=int(hop_length),
        n_fft=int(n_fft),
    ).astype(np.float32, copy=False)  # (12, n)

    if chroma.size == 0:
        return np.zeros((0, 12), dtype=np.float32)

    chroma = chroma.T  # (n, 12)
    denom = np.max(chroma, axis=1, keepdims=True)
    chroma = np.divide(chroma, denom, out=np.zeros_like(chroma), where=denom > 1e-6)
    return chroma.astype(np.float32, copy=False)

