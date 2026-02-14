from __future__ import annotations

from typing import Final

import numpy as np
import librosa


_DEFAULT_FMIN_HZ: Final[float] = float(librosa.note_to_hz("C2"))
_DEFAULT_FMAX_HZ: Final[float] = float(librosa.note_to_hz("C7"))
_BASS_FMIN_HZ: Final[float] = 30.0
_BASS_FMAX_HZ: Final[float] = 400.0


def _nanmedian_smooth(x: np.ndarray, *, win: int) -> np.ndarray:
    """
    NaN-aware running median smoothing.

    Small utility to stabilize pitch tracks without adding scipy as a dependency.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    if win <= 1:
        return x.astype(np.float32, copy=True)
    win = int(win)
    if win % 2 == 0:
        win += 1
    r = win // 2
    out = np.empty_like(x, dtype=np.float32)
    for i in range(x.size):
        lo = max(0, i - r)
        hi = min(x.size, i + r + 1)
        v = np.nanmedian(x[lo:hi])
        out[i] = (0.0 if not np.isfinite(v) else float(v))
    return out


def _hz_to_midi(hz: np.ndarray) -> np.ndarray:
    hz = np.asarray(hz, dtype=np.float32)
    return (69.0 + 12.0 * np.log2(np.maximum(hz, 1e-6) / 440.0)).astype(np.float32)


def _midi_to_hz(midi: np.ndarray) -> np.ndarray:
    midi = np.asarray(midi, dtype=np.float32)
    return (440.0 * (2.0 ** ((midi - 69.0) / 12.0))).astype(np.float32)


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

    We want "notes", not sibilance/noise scribbles:
    - extract harmonic component to suppress consonants/fricatives
    - use pYIN (voicing probability) + RMS gate
    - smooth + quantize to semitones for readability
    """
    if y.ndim != 1:
        raise ValueError(f"Expected mono audio (1D array), got shape={y.shape}")
    if sr <= 0:
        return np.zeros((0,), dtype=np.float32)

    # Harmonic component makes pitch tracking much more stable for vocals stems.
    yh = librosa.effects.harmonic(y=y, margin=8.0).astype(np.float32, copy=False)

    f0, voiced_flag, voiced_prob = librosa.pyin(
        y=yh,
        fmin=float(fmin_hz),
        fmax=float(fmax_hz),
        sr=int(sr),
        frame_length=int(frame_length),
        hop_length=int(hop_length),
    )
    f0 = np.asarray(f0, dtype=np.float32)
    voiced_flag = np.asarray(voiced_flag, dtype=bool)
    voiced_prob = np.asarray(voiced_prob, dtype=np.float32)

    rms = librosa.feature.rms(
        y=yh,
        frame_length=int(frame_length),
        hop_length=int(hop_length),
        center=True,
    )[0].astype(np.float32, copy=False)

    n = int(min(f0.size, rms.size))
    if n <= 0:
        return np.zeros((0,), dtype=np.float32)
    f0 = f0[:n]
    rms = rms[:n]
    voiced_flag = voiced_flag[:n]
    voiced_prob = voiced_prob[:n]

    # Gate based on relative energy to avoid pitch scribbles in silence.
    thr = float(np.percentile(rms, 25)) * 0.8
    thr = max(thr, float(rms.max()) * 0.08, 1e-6)
    voiced = (rms >= thr) & voiced_flag & (voiced_prob >= 0.80)

    f0v = np.where(voiced, f0, np.nan).astype(np.float32)

    # Smooth in MIDI space, then quantize to semitones.
    midi = np.where(np.isfinite(f0v), _hz_to_midi(f0v), np.nan).astype(np.float32)
    midi = _nanmedian_smooth(midi, win=7)
    midi_q = np.round(midi).astype(np.float32)
    f0_q = np.where(np.isfinite(midi_q), _midi_to_hz(midi_q), np.nan).astype(np.float32)
    return f0_q


def bass_pitch_hz(
    y: np.ndarray,
    sr: int,
    *,
    hop_length: int = 512,
    frame_length: int = 2048,
    fmin_hz: float = _BASS_FMIN_HZ,
    fmax_hz: float = _BASS_FMAX_HZ,
) -> np.ndarray:
    """
    Pitch track (Hz) for bass visualization.

    Same method as vocals (YIN + energy gate), but with a bass-appropriate range.
    """
    if y.ndim != 1:
        raise ValueError(f"Expected mono audio (1D array), got shape={y.shape}")
    if sr <= 0:
        return np.zeros((0,), dtype=np.float32)

    yh = librosa.effects.harmonic(y=y, margin=6.0).astype(np.float32, copy=False)
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y=yh,
        fmin=float(fmin_hz),
        fmax=float(fmax_hz),
        sr=int(sr),
        frame_length=int(frame_length),
        hop_length=int(hop_length),
    )
    f0 = np.asarray(f0, dtype=np.float32)
    voiced_flag = np.asarray(voiced_flag, dtype=bool)
    voiced_prob = np.asarray(voiced_prob, dtype=np.float32)

    rms = librosa.feature.rms(
        y=yh,
        frame_length=int(frame_length),
        hop_length=int(hop_length),
        center=True,
    )[0].astype(np.float32, copy=False)

    n = int(min(f0.size, rms.size))
    if n <= 0:
        return np.zeros((0,), dtype=np.float32)
    f0 = f0[:n]
    rms = rms[:n]
    voiced_flag = voiced_flag[:n]
    voiced_prob = voiced_prob[:n]

    thr = float(np.percentile(rms, 25)) * 0.8
    thr = max(thr, float(rms.max()) * 0.08, 1e-6)
    voiced = (rms >= thr) & voiced_flag & (voiced_prob >= 0.75)

    f0v = np.where(voiced, f0, np.nan).astype(np.float32)
    midi = np.where(np.isfinite(f0v), _hz_to_midi(f0v), np.nan).astype(np.float32)
    midi = _nanmedian_smooth(midi, win=7)
    midi_q = np.round(midi).astype(np.float32)
    f0_q = np.where(np.isfinite(midi_q), _midi_to_hz(midi_q), np.nan).astype(np.float32)
    return f0_q


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


def drums_band_energy_3(
    y: np.ndarray,
    sr: int,
    *,
    hop_length: int = 512,
    n_fft: int = 2048,
) -> np.ndarray:
    """
    3-band energy proxy for drums: (N, 3) => [kick, snare, hats].

    This is not true drum classification, but it's a decent heuristic:
    - kick: 20-150 Hz
    - snare: 150-2500 Hz
    - hats: 2500+ Hz
    """
    if y.ndim != 1:
        raise ValueError(f"Expected mono audio (1D array), got shape={y.shape}")
    if sr <= 0:
        return np.zeros((0, 3), dtype=np.float32)

    S = np.abs(librosa.stft(y=y, n_fft=int(n_fft), hop_length=int(hop_length))) ** 2  # (f, n)
    if S.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    freqs = librosa.fft_frequencies(sr=int(sr), n_fft=int(n_fft)).astype(np.float32)

    def _band(lo: float, hi: float | None) -> np.ndarray:
        if hi is None:
            m = freqs >= float(lo)
        else:
            m = (freqs >= float(lo)) & (freqs < float(hi))
        if not np.any(m):
            return np.zeros((S.shape[1],), dtype=np.float32)
        return np.sum(S[m, :], axis=0).astype(np.float32, copy=False)

    kick = _band(20.0, 150.0)
    snare = _band(150.0, 2500.0)
    hats = _band(2500.0, None)

    X = np.stack([kick, snare, hats], axis=1)  # (n, 3)

    # Per-band normalize to [0,1] with a robust percentile cap (avoid one huge hit flattening everything).
    out = np.zeros_like(X, dtype=np.float32)
    for i in range(3):
        v = X[:, i]
        cap = float(np.percentile(v, 99)) if v.size else 0.0
        cap = max(cap, float(v.max()) if v.size else 0.0, 1e-9)
        out[:, i] = np.clip(v / cap, 0.0, 1.0)
    return out
