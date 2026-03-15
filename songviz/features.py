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
        sl = x[lo:hi]
        if not np.isfinite(sl).any():
            out[i] = np.nan
            continue
        v = np.nanmedian(sl)
        out[i] = (0.0 if not np.isfinite(v) else float(v))
    return out


def _hz_to_midi(hz: np.ndarray) -> np.ndarray:
    hz = np.asarray(hz, dtype=np.float32)
    return (69.0 + 12.0 * np.log2(np.maximum(hz, 1e-6) / 440.0)).astype(np.float32)


def _midi_to_hz(midi: np.ndarray) -> np.ndarray:
    midi = np.asarray(midi, dtype=np.float32)
    return (440.0 * (2.0 ** ((midi - 69.0) / 12.0))).astype(np.float32)


def _pyin_pitch_hz(
    y: np.ndarray,
    sr: int,
    *,
    fmin_hz: float,
    fmax_hz: float,
    harmonic_margin: float,
    voiced_prob_thr: float,
    hop_length: int,
    frame_length: int,
) -> np.ndarray:
    if y.ndim != 1:
        raise ValueError(f"Expected mono audio (1D array), got shape={y.shape}")
    if sr <= 0:
        return np.zeros((0,), dtype=np.float32)

    yh = librosa.effects.harmonic(y=y, margin=harmonic_margin).astype(np.float32, copy=False)

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
    voiced = (rms >= thr) & voiced_flag & (voiced_prob >= voiced_prob_thr)

    f0v = np.where(voiced, f0, np.nan).astype(np.float32)
    midi = np.where(np.isfinite(f0v), _hz_to_midi(f0v), np.nan).astype(np.float32)
    midi = _nanmedian_smooth(midi, win=7)
    midi_q = np.round(midi).astype(np.float32)
    return np.where(np.isfinite(midi_q), _midi_to_hz(midi_q), np.nan).astype(np.float32)


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
    return _pyin_pitch_hz(
        y, sr,
        fmin_hz=fmin_hz, fmax_hz=fmax_hz,
        harmonic_margin=8.0, voiced_prob_thr=0.80,
        hop_length=hop_length, frame_length=frame_length,
    )


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

    Same method as vocals (pYIN + energy gate), but with a bass-appropriate range.
    """
    return _pyin_pitch_hz(
        y, sr,
        fmin_hz=fmin_hz, fmax_hz=fmax_hz,
        harmonic_margin=6.0, voiced_prob_thr=0.75,
        hop_length=hop_length, frame_length=frame_length,
    )


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


def drums_band_energy_3_from_components(
    components: dict[str, np.ndarray],
    sr: int,
    *,
    hop_length: int = 512,
) -> np.ndarray:
    """
    3-band energy from DrumSep component stems: (N, 3) => [kick, snare, hats].

    Maps the 6 DrumSep components to our 3-band scheme:
    - kick: kick
    - snare: snare + toms
    - hats: hh + ride + crash

    Same output shape and normalization as drums_band_energy_3() — drop-in replacement.
    """
    # Check for empty input early
    if all(v.size == 0 for v in components.values()):
        return np.zeros((0, 3), dtype=np.float32)

    def _rms(y: np.ndarray) -> np.ndarray:
        if y.size == 0:
            return np.zeros((0,), dtype=np.float32)
        return librosa.feature.rms(y=y, hop_length=int(hop_length))[0].astype(np.float32, copy=False)

    band_rms: list[np.ndarray] = []
    for group in [("kick",), ("snare", "toms"), ("hh", "ride", "crash")]:
        parts = [_rms(components[k]) for k in group if k in components and components[k].size > 0]
        if not parts:
            band_rms.append(np.zeros((0,), dtype=np.float32))
            continue
        n = min(p.size for p in parts)
        combined = sum(p[:n] for p in parts)
        band_rms.append(combined)

    if not band_rms or any(b.size == 0 for b in band_rms):
        return np.zeros((0, 3), dtype=np.float32)

    n = min(b.size for b in band_rms)
    out = np.zeros((n, 3), dtype=np.float32)
    for i, b in enumerate(band_rms):
        v = b[:n]
        cap = float(np.percentile(v, 99)) if v.size else 0.0
        cap = max(cap, float(v.max()) if v.size else 0.0, 1e-9)
        out[:, i] = np.clip(v / cap, 0.0, 1.0)
    return out


def vocals_note_events_basic_pitch(
    audio_path: str,
    *,
    onset_threshold: float = 0.55,
    frame_threshold: float = 0.30,
    minimum_note_length_ms: float = 120.0,
    minimum_frequency: float = 60.0,
    maximum_frequency: float = 1100.0,
) -> list[dict[str, float]]:
    """
    Extract note events from (isolated) vocals using Spotify's Basic Pitch.

    Returns a list of events: {"start_s","end_s","midi","velocity"}.
    This is intentionally a small, renderer-friendly schema.
    """
    try:
        from basic_pitch.inference import predict  # type: ignore
    except Exception as e:  # pragma: no cover (optional dependency)
        raise RuntimeError(
            "basic_pitch is not installed. Install it to enable vocal note events:\n"
            "  pip install -e '.[notes]'\n"
            "or:\n"
            "  python3 -m pip install basic-pitch"
        ) from e

    _, _, note_events = predict(
        audio_path,
        onset_threshold=float(onset_threshold),
        frame_threshold=float(frame_threshold),
        minimum_note_length=float(minimum_note_length_ms),
        minimum_frequency=float(minimum_frequency),
        maximum_frequency=float(maximum_frequency),
        melodia_trick=True,
    )

    events: list[dict[str, float]] = []

    # Basic Pitch has returned different shapes in the wild; normalize defensively.
    if isinstance(note_events, list):
        for it in note_events:
            if isinstance(it, dict):
                s = float(it.get("start_time_s", it.get("start_s", 0.0)))
                e = float(it.get("end_time_s", it.get("end_s", s)))
                m = it.get("pitch_midi", it.get("midi", it.get("pitch", np.nan)))
                v = it.get("velocity", 0.0)
                if not np.isfinite(m):
                    continue
                events.append({"start_s": s, "end_s": e, "midi": float(m), "velocity": float(v)})
            else:
                # unknown element type
                continue
        return events

    arr = np.asarray(note_events)
    if arr.size == 0:
        return events

    # Structured array with named fields.
    if arr.dtype.fields:
        f = arr.dtype.fields
        for row in arr:
            s = float(row["start_time_s"] if "start_time_s" in f else row["start_s"])
            e = float(row["end_time_s"] if "end_time_s" in f else row["end_s"])
            if "pitch_midi" in f:
                m = float(row["pitch_midi"])
            elif "midi" in f:
                m = float(row["midi"])
            else:
                m = float(row["pitch"])
            vel = float(row["velocity"]) if "velocity" in f else 0.0
            events.append({"start_s": s, "end_s": e, "midi": m, "velocity": vel})
        return events

    # Fallback: assume columns [start, end, midi, velocity] or [start, end, pitch, velocity].
    if arr.ndim == 2 and arr.shape[1] >= 3:
        for row in arr:
            s = float(row[0])
            e = float(row[1])
            m = float(row[2])
            vel = float(row[3]) if arr.shape[1] >= 4 else 0.0
            if not np.isfinite(m):
                continue
            events.append({"start_s": s, "end_s": e, "midi": m, "velocity": vel})
    return events
