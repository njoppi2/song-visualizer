"""Experimental regular-pulse inference for recordings with a steady backbeat.

This deliberately does not replace the general beat tracker. It reports the
constant-tempo assumption and measured fit; a fit score is not musical certainty.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


def prominent_attacks(y: np.ndarray, sr: int) -> np.ndarray:
    """Find strong transient starts for a timing anchor, using 2ms RMS windows.

    Use a separated snare-like component. The 40%-peak prominence selects its
    main attacks; quiet ghost notes are intentionally unsuitable as grid anchors.
    Backtrack each peak within 80ms to a 15% local-amplitude rise.
    """
    if sr <= 0 or y.ndim not in (1, 2) or not np.all(np.isfinite(y)):
        raise ValueError("Expected finite mono/stereo samples and a positive sample rate")
    if y.ndim == 1:
        y = y[:, None]
    hop = max(1, round(sr * 0.002))
    count = len(y) // hop
    if count < 3:
        return np.array([], dtype=float)
    energy = np.sqrt(np.mean(y[:count * hop].reshape(count, hop, -1).astype(float) ** 2, axis=(1, 2)))
    if energy.max() < 1e-7:
        return np.array([], dtype=float)
    dt = hop / sr
    peaks, _ = find_peaks(energy, prominence=energy.max() * 0.4, distance=max(1, round(0.15 / dt)))
    starts = []
    for peak in peaks:
        lo = max(0, peak - round(0.08 / dt))
        segment = energy[lo:peak + 1]
        threshold = segment.min() + 0.15 * (energy[peak] - segment.min())
        below = np.flatnonzero(segment <= threshold)
        starts.append((lo + int(below[-1]) + 1 if len(below) else peak) * dt)
    return np.asarray(starts)


def fit_regular_pulse(anchors: np.ndarray, *, duration_s: float,
                      fit_start_s: float, fit_end_s: float,
                      pulses_per_anchor: int = 2,
                      active_start_s: float = 0.0) -> dict:
    """Fit a constant grid to repeated anchors, tolerating missing/extra events.

    Integer subdivisions are a caller-supplied metrical hypothesis, not inferred
    downbeats. Fit on one interval and retain residuals outside it for inspection.
    Reject weak evidence instead of creating an apparently confident tempo.
    """
    anchors = np.asarray(anchors, dtype=float)
    if anchors.ndim != 1 or not np.all(np.isfinite(anchors)) or np.any(np.diff(anchors) <= 0):
        raise ValueError("Anchors must be finite and strictly increasing")
    if not (0 <= fit_start_s < fit_end_s <= duration_s) or pulses_per_anchor not in (1, 2, 4):
        raise ValueError("Invalid fitting interval or subdivision")
    if not 0 <= active_start_s < duration_s:
        raise ValueError("Invalid active start")
    train = anchors[(anchors >= fit_start_s) & (anchors < fit_end_s)]
    if len(train) < 12 or np.ptp(train) < 10:
        return {"status": "insufficient_evidence", "beat_times_s": []}
    intervals = np.diff(train)
    edges = np.arange(0.3, 1.51, 0.01)
    hist, _ = np.histogram(intervals, bins=edges)
    if hist.max() < 4:
        return {"status": "insufficient_repetition", "beat_times_s": []}
    mode = int(np.argmax(hist))
    nearby = intervals[(intervals >= edges[mode] - .01) & (intervals < edges[mode + 1] + .01)]
    period = float(np.median(nearby))
    support = [np.sum(np.abs((train - phase + period / 2) % period - period / 2) < .03) for phase in train]
    phase = float(train[np.argmax(support)])
    for _ in range(6):
        indices = np.round((train - phase) / period)
        residual = train - (phase + indices * period)
        good = np.abs(residual) < .04
        if np.count_nonzero(good) < 12 or np.ptp(indices[good]) < 8:
            return {"status": "insufficient_consensus", "beat_times_s": []}
        period, phase = (float(x) for x in np.polyfit(indices[good], train[good], 1))
    residuals = anchors - (phase + np.round((anchors - phase) / period) * period)
    mask = (anchors >= fit_start_s) & (anchors < fit_end_s)
    consensus = float(np.mean(np.abs(residuals[mask]) < .03))
    if consensus < .75:
        return {"status": "non_regular_evidence", "consensus_30ms": consensus, "beat_times_s": []}
    beat_period = period / pulses_per_anchor
    first = int(np.ceil((active_start_s - phase) / beat_period))
    last = int(np.ceil((duration_s - phase) / beat_period))
    beats = phase + np.arange(first, last) * beat_period
    outside = np.abs(residuals[~mask])
    return {
        "status": "candidate", "source": "regular_prominent_attack_fit_v1",
        "assumption": "constant tempo, snare-like anchors, no downbeat claim",
        "tempo_bpm": 60 / beat_period, "anchor_period_s": period,
        "phase_s": phase % beat_period, "pulses_per_anchor": pulses_per_anchor,
        "fit_interval_s": [fit_start_s, fit_end_s], "fit_anchor_count": len(train),
        "consensus_30ms": consensus,
        "fit_median_residual_ms": float(np.median(np.abs(residuals[mask])) * 1000),
        "outside_fit_median_residual_ms": float(np.median(outside) * 1000) if outside.size else None,
        "outside_fit_p95_residual_ms": float(np.quantile(outside, .95) * 1000) if outside.size else None,
        "anchors_s": anchors.tolist(), "anchor_residuals_ms": (residuals * 1000).tolist(),
        "beat_times_s": beats.tolist(),
    }
