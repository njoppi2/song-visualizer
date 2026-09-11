"""Independent, off-grid attack extraction for separated percussion components.

This is deliberately a sidecar to :mod:`songviz.reduction`: it does not use a
beat grid, infer a repeating pattern, or move attacks onto musical divisions.
It is intended for inspecting the raw timing candidates supplied by a source
separator, whose component bleed and missed/merged instruments remain limits of
the input audio.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


_COMPONENTS = frozenset({"kick", "snare", "hh", "toms", "ride", "crash"})

# These guard against a single decaying hit being emitted repeatedly.  They are
# intentionally short enough to retain rolls and fast hi-hat patterns.
_REFRACTORY_SECONDS: dict[str, float] = {
    "kick": 0.070,
    "snare": 0.070,
    "hh": 0.030,
    "toms": 0.070,
    "ride": 0.045,
    "crash": 0.180,
}

_SILENCE_RMS = 1e-7
_SMOOTHING_SECONDS = 0.008
_LOOKBACK_SECONDS = 0.060
_BACKTRACK_FRACTION = 0.15

# A faint candidate must clear a local noise gate.  A displayed hit also has
# to be material relative to this component's whole-excerpt envelope.  The
# latter is intentionally conservative: faint candidates remain available for
# inspection but are not silently promoted to drum hits.
_FAINT_MAD_MULTIPLIER = 4.0
_FAINT_PEAK_RATIO = 0.005
_PROMINENT_MAD_MULTIPLIER = 10.0
_PROMINENT_PEAK_RATIO = 0.200


def _small_hop_rms(y: np.ndarray, hop_length: int) -> np.ndarray:
    """Return non-centred RMS frames, so a frame time is its leading edge."""
    n_frames = max(1, int(np.ceil(len(y) / hop_length)))
    padded = np.pad(y, (0, n_frames * hop_length - len(y)))
    frames = padded.reshape(n_frames, hop_length).astype(np.float64, copy=False)
    return np.sqrt(np.mean(frames * frames, axis=1))


def _component_hits(
    component: str,
    y: np.ndarray,
    sr: int,
    hop_length: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float | int]]:
    """Detect envelope peaks, backtracked to a preceding local rise."""
    raw_env = _small_hop_rms(y, hop_length)
    max_rms = float(np.max(raw_env))
    if max_rms < _SILENCE_RMS:
        return [], [], {
            "frames": int(len(raw_env)),
            "max_rms": max_rms,
            "faint_strength_threshold": 0.0,
            "prominent_strength_threshold": 0.0,
            "hits": 0,
            "faint_hits": 0,
        }

    sigma_frames = max(1.0, _SMOOTHING_SECONDS * sr / hop_length)
    env = gaussian_filter1d(raw_env, sigma=sigma_frames, mode="nearest")
    median_env = float(np.median(env))
    mad_env = float(np.median(np.abs(env - median_env)))
    faint_strength_threshold = max(
        _FAINT_MAD_MULTIPLIER * mad_env,
        max_rms * _FAINT_PEAK_RATIO,
        _SILENCE_RMS,
    )
    prominent_strength_threshold = max(
        _PROMINENT_MAD_MULTIPLIER * mad_env,
        max_rms * _PROMINENT_PEAK_RATIO,
        _SILENCE_RMS,
    )
    refractory_frames = max(
        1, int(np.ceil(_REFRACTORY_SECONDS[component] * sr / hop_length))
    )
    lookback_frames = max(1, int(np.ceil(_LOOKBACK_SECONDS * sr / hop_length)))

    # Prominence is measured in a short local window.  This rejects
    # frame-to-frame RMS wiggle and prevents a slow tail from being repeatedly
    # emitted as attacks; it is not a tempo or beat-grid criterion.
    peak_frames, _ = find_peaks(
        env,
        prominence=faint_strength_threshold,
        distance=refractory_frames,
        wlen=max(3, 2 * lookback_frames + 1),
    )
    candidates: list[tuple[int, float, float]] = []
    for peak_frame in peak_frames:
        start = max(0, int(peak_frame) - lookback_frames)
        local_env = env[start : int(peak_frame) + 1]
        trough_frame = start + int(np.argmin(local_env))
        strength = float(env[peak_frame] - env[trough_frame])
        if strength < faint_strength_threshold:
            continue
        # Timestamp the upward crossing rather than the later envelope peak.
        crossing = env[trough_frame] + _BACKTRACK_FRACTION * strength
        between = np.flatnonzero(env[trough_frame : int(peak_frame) + 1] >= crossing)
        attack_frame = trough_frame + int(between[0]) if len(between) else int(peak_frame)
        candidates.append((attack_frame, strength, float(raw_env[peak_frame])))

    if not candidates:
        return [], [], {
            "frames": int(len(env)),
            "max_rms": max_rms,
            "faint_strength_threshold": float(faint_strength_threshold),
            "prominent_strength_threshold": float(prominent_strength_threshold),
            "hits": 0,
            "faint_hits": 0,
        }

    # Velocity is relative only to retained candidates in this component.  The
    # raw ``strength`` and ``peak_rms`` fields are detector measurements, not
    # calibrated confidence or cross-component loudness measurements.
    velocity_cap = float(np.percentile([candidate[2] for candidate in candidates], 95))
    if velocity_cap < _SILENCE_RMS:
        velocity_cap = max_rms

    hits: list[dict[str, Any]] = []
    faint_hits: list[dict[str, Any]] = []
    for frame, strength, peak_rms in candidates:
        hit = {
            "t": round(float(frame * hop_length / sr), 6),
            "component": component,
            "velocity": round(float(np.clip(peak_rms / velocity_cap, 0.0, 1.0)), 4),
            "strength": round(strength, 8),
            "peak_rms": round(peak_rms, 8),
        }
        if strength >= prominent_strength_threshold:
            hits.append(hit)
        else:
            faint_hits.append(hit)
    return hits, faint_hits, {
        "frames": int(len(env)),
        "max_rms": float(max_rms),
        "faint_strength_threshold": float(faint_strength_threshold),
        "prominent_strength_threshold": float(prominent_strength_threshold),
        "hits": int(len(hits)),
        "faint_hits": int(len(faint_hits)),
    }


def extract_component_attacks(
    components: dict[str, np.ndarray], sr: int,
) -> dict[str, Any]:
    """Extract independent percussion attacks from cached component waveforms.

    Only ``kick``, ``snare``, ``hh``, ``toms``, ``ride``, and ``crash`` are
    considered.  Every accepted component is analysed independently, so two
    simultaneous attacks remain two hits.  Timestamps are leading edges of
    non-centred, small-hop RMS frames and are never quantized to beats.

    ``hits`` is the conservative, prominent set for a first review.
    ``faint_hits`` retains separately gated weaker candidates.  ``strength``
    is the raw smoothed-envelope rise from a preceding local trough and
    ``peak_rms`` is the unsmoothed RMS at its envelope peak.  Neither is a
    confidence claim.  ``velocity`` is capped and normalized within each
    component's retained candidates.
    """
    if not isinstance(sr, (int, np.integer)) or int(sr) <= 0:
        raise ValueError("sr must be a positive integer")

    sr = int(sr)
    # About 4 ms provides enough temporal resolution for visual comparison
    # while retaining a stable envelope at common cached-audio sample rates.
    hop_length = max(16, min(256, int(round(sr * 0.004))))
    all_hits: list[dict[str, Any]] = []
    all_faint_hits: list[dict[str, Any]] = []
    component_parameters: dict[str, dict[str, float | int]] = {}

    for component, waveform in components.items():
        if component not in _COMPONENTS:
            continue
        y = np.asarray(waveform, dtype=np.float32)
        if y.ndim != 1:
            raise ValueError(f"component {component!r} must be a mono 1-D waveform")
        # A separator should not produce non-finite samples, but making them
        # silent avoids a single bad sample poisoning the complete envelope.
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        hits, faint_hits, details = _component_hits(component, y, sr, hop_length)
        all_hits.extend(hits)
        all_faint_hits.extend(faint_hits)
        component_parameters[component] = details

    all_hits.sort(key=lambda hit: (hit["t"], hit["component"]))
    all_faint_hits.sort(key=lambda hit: (hit["t"], hit["component"]))
    return {
        "source": "component_attacks_v1",
        "hits": all_hits,
        "faint_hits": all_faint_hits,
        "parameters": {
            "hop_length": hop_length,
            "hop_seconds": hop_length / sr,
            "envelope": "non-centred RMS; timestamp is frame leading edge",
            "onset": "smoothed-envelope prominence, backtracked to local rise",
            "silence_rms": _SILENCE_RMS,
            "smoothing_seconds": _SMOOTHING_SECONDS,
            "lookback_seconds": _LOOKBACK_SECONDS,
            "backtrack_fraction": _BACKTRACK_FRACTION,
            "faint_mad_multiplier": _FAINT_MAD_MULTIPLIER,
            "faint_peak_ratio": _FAINT_PEAK_RATIO,
            "prominent_mad_multiplier": _PROMINENT_MAD_MULTIPLIER,
            "prominent_peak_ratio": _PROMINENT_PEAK_RATIO,
            "refractory_seconds": dict(_REFRACTORY_SECONDS),
            "component_thresholds": component_parameters,
        },
    }
