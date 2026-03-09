from __future__ import annotations

from typing import Any

import librosa
import numpy as np

from .ingest import _normalize_01


def _smooth_1d(x: np.ndarray, win: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    win = int(max(1, win))
    if win <= 1:
        return x
    k = np.ones((win,), dtype=np.float32) / float(win)
    return np.convolve(x, k, mode="same").astype(np.float32)


def _tension_valley_boundaries(
    tension: np.ndarray,
    times_s: np.ndarray,
    *,
    min_len_s: float,
    duration_s: float,
) -> list[float]:
    """Find section boundaries at the deepest valleys in a smoothed tension curve.

    Used as a fallback when MFCC agglomerative clustering degenerates
    (e.g. for songs with very consistent timbre like Do I Wanna Know).
    """
    n_per_sec = max(1, int(tension.size / max(1e-6, duration_s)))
    heavy_win = max(5, int(12.0 * n_per_sec))
    smooth = _smooth_1d(np.asarray(tension, dtype=np.float32), win=heavy_win)

    # Find all derivative zero-crossings (neg→pos = valley).
    deriv = np.diff(smooth, prepend=smooth[0])
    sign = np.sign(deriv)

    # Score each valley by its depth: average of surrounding peaks minus valley.
    peak_search = max(1, int(min_len_s * n_per_sec))
    valleys: list[tuple[int, float]] = []  # (index, depth)
    for i in range(1, len(sign)):
        if sign[i - 1] < 0 and sign[i] >= 0:
            left_peak = float(smooth[max(0, i - peak_search) : i].max())
            right_peak = float(smooth[i : min(len(smooth), i + peak_search)].max())
            depth = (left_peak + right_peak) / 2.0 - float(smooth[i])
            valleys.append((i, depth))

    # Greedily select the deepest valleys with at least min_len_s separation
    # (non-maximum suppression), targeting ~1 boundary per 40 s.
    target_k = max(2, int(round(duration_s / 55.0)))
    valleys.sort(key=lambda v: v[1], reverse=True)

    selected: list[float] = []
    for idx, _depth in valleys:
        valley_val = float(smooth[idx])
        back_start = max(0, idx - peak_search)
        forward_end = min(len(smooth), idx + peak_search)
        left_energy = float(smooth[back_start:idx].mean()) if idx > back_start else valley_val
        right_energy = float(smooth[idx:forward_end].mean()) if forward_end > idx else valley_val

        if right_energy >= left_energy:
            # LOW→HIGH transition: shift to rising edge (scan forward).
            peak_val = float(smooth[idx:forward_end].max())
            threshold = valley_val + 0.4 * (peak_val - valley_val)
            edge_idx = idx
            for j in range(idx, forward_end):
                if float(smooth[j]) >= threshold:
                    edge_idx = j
                    break
        else:
            # HIGH→LOW transition: shift to falling edge (scan backward).
            peak_val = float(smooth[back_start:idx].max())
            threshold = valley_val + 0.4 * (peak_val - valley_val)
            edge_idx = idx
            for j in range(idx, back_start - 1, -1):
                if float(smooth[j]) >= threshold:
                    edge_idx = j
                    break

        t = float(times_s[min(edge_idx, len(times_s) - 1)])
        if t < min_len_s or (duration_s - t) < min_len_s:
            continue
        if all(abs(t - s) >= min_len_s for s in selected):
            selected.append(t)
        if len(selected) >= target_k:
            break

    selected.sort()
    return [0.0] + selected + [duration_s]


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
            continue
        out.append(cur)
    out.append(b[-1])
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


def _assign_motif_labels(
    section_means: list[np.ndarray],
    *,
    threshold: float = 0.82,
) -> list[str]:
    """Cluster sections by cosine similarity and assign recurring letter labels.

    Sections that sound alike (cos_sim >= threshold) get the same letter so the
    renderer can use the same palette for recurring song parts (verse/chorus/bridge).
    """
    n = len(section_means)
    if n == 0:
        return []

    norms = [float(np.linalg.norm(m)) for m in section_means]
    normed = [m / (nrm + 1e-8) for m, nrm in zip(section_means, norms)]

    motif_id = [-1] * n
    next_id = 0
    for i in range(n):
        if motif_id[i] >= 0:
            continue
        motif_id[i] = next_id
        for j in range(i + 1, n):
            if motif_id[j] >= 0:
                continue
            sim = float(np.dot(normed[i], normed[j]))
            if sim >= threshold:
                motif_id[j] = next_id
        next_id += 1

    n_unique = max(motif_id) + 1
    unique_labels = _labels_for_n(n_unique)
    return [unique_labels[mid] for mid in motif_id]


def _merge_same_label_sections(
    sections: list[dict[str, Any]],
    *,
    max_merged_len_s: float = 120.0,
) -> list[dict[str, Any]]:
    """Merge consecutive sections that share the same motif label, but only
    when the combined length stays below max_merged_len_s.  This removes
    micro-duplicate pairs (A-A → A) without collapsing long same-label
    runs (verse–chorus alternation) into a single unreadable block.
    """
    if not sections:
        return sections
    merged = [sections[0].copy()]
    for sec in sections[1:]:
        prev = merged[-1]
        would_be_len = sec["end_s"] - prev["start_s"]
        if sec["label"] == prev["label"] and would_be_len <= max_merged_len_s:
            merged[-1]["end_s"] = sec["end_s"]
        else:
            merged.append(sec.copy())
    return merged


def compute_story(
    y: np.ndarray,
    sr: int,
    *,
    hop_length: int = 512,
    frame_length: int = 2048,
) -> dict[str, Any]:
    """
    Heuristic "song story" signals:
    - sections: coarse segmentation with motif-aware labels (same letter = same sound).
    - tension: a smooth energy/brightness curve (0..1), good for buildup/drop dynamics.
    - events: drop_times_s (sharp tension drops) and buildups (rising tension windows).

    This is intentionally lightweight and fully deterministic.
    """
    if y.ndim != 1:
        raise ValueError(f"Expected mono audio (1D array), got shape={y.shape}")
    if sr <= 0:
        raise ValueError(f"Invalid sample_rate={sr}")

    duration_s = float(len(y) / sr)

    # --- Per-frame features ---
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

    raw_tension = 0.48 * (rms01**0.8) + 0.34 * (onset01**0.7) + 0.18 * (cent01**0.9)
    win = max(5, int(round(0.35 / max(1e-6, (hop_length / sr)))))
    raw_tension_smoothed = _smooth_1d(raw_tension, win=win)
    tension = _normalize_01(raw_tension_smoothed)

    # --- Coarse segmentation (timbre + energy via MFCC + RMS/onset) ---
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=hop_length, n_fft=frame_length)
    mu = np.mean(mfcc, axis=1, keepdims=True)
    sig = np.std(mfcc, axis=1, keepdims=True) + 1e-6
    mfcc_z = (mfcc - mu) / sig

    # Augment MFCC with energy features so the clustering can distinguish
    # sections by loudness/onset, not just timbre.  Songs with consistent
    # timbre (e.g. same guitar tone throughout) would otherwise collapse
    # into a single segment.
    # Align frame counts (MFCC may differ from RMS/onset by a few frames).
    n_seg_frames = min(mfcc_z.shape[1], len(rms01), len(onset01))
    mfcc_z_seg = mfcc_z[:, :n_seg_frames]
    # Smooth energy features with ~2s window to avoid per-beat noise
    _seg_smooth_win = min(max(3, int(2.0 / max(1e-6, hop_length / sr))), n_seg_frames)
    rms_seg = _smooth_1d(rms01[:n_seg_frames], win=_seg_smooth_win)[np.newaxis, :]
    onset_seg = _smooth_1d(onset01[:n_seg_frames], win=_seg_smooth_win)[np.newaxis, :]
    seg_features = np.vstack([mfcc_z_seg, rms_seg * 3.0, onset_seg * 2.0])

    k = int(np.clip(round(duration_s / 25.0), 4, 10))
    try:
        seg_labels = librosa.segment.agglomerative(seg_features, k=k)
    except Exception:
        seg_labels = np.zeros((n_seg_frames,), dtype=int)

    change = np.flatnonzero(np.diff(seg_labels) != 0) + 1
    bounds_frames = np.concatenate([[0], change.astype(int), [int(seg_labels.size)]])
    bounds_s = librosa.frames_to_time(bounds_frames, sr=sr, hop_length=hop_length).astype(float).tolist()
    bounds_s = _merge_short_segments(bounds_s, min_len_s=15.0, duration_s=duration_s)

    # Fallback: if MFCC clustering degenerates (≤2 sections for songs >60 s),
    # use tension valleys to find structural boundaries from the energy contour.
    if len(bounds_s) <= 3 and duration_s > 60.0:
        bounds_s = _tension_valley_boundaries(
            raw_tension_smoothed, times_s,
            min_len_s=15.0, duration_s=duration_s,
        )

    # --- Motif-aware labels: compute mean MFCC + energy per section ---
    # Augmenting with energy means sections at different loudness levels
    # (verse vs chorus) get different labels even when timbre is identical.
    section_means: list[np.ndarray] = []
    hop_s = float(hop_length / sr)
    for i in range(len(bounds_s) - 1):
        s0_f = int(round(bounds_s[i] / hop_s))
        s1_f = int(round(bounds_s[i + 1] / hop_s))
        s0_f = max(0, min(s0_f, mfcc_z.shape[1]))
        s1_f = max(s0_f + 1, min(s1_f, mfcc_z.shape[1]))
        mfcc_mean = mfcc_z[:, s0_f:s1_f].mean(axis=1)
        # Clamp to RMS/onset array bounds.
        e0 = max(0, min(s0_f, n - 1))
        e1 = max(e0 + 1, min(s1_f, n))
        rms_mean = float(rms01[e0:e1].mean())
        onset_mean = float(onset01[e0:e1].mean())
        section_means.append(np.concatenate([mfcc_mean, [rms_mean * 5.0, onset_mean * 3.0]]))

    motif_labels = _assign_motif_labels(section_means)
    if len(motif_labels) < len(bounds_s) - 1:
        motif_labels = _labels_for_n(max(1, len(bounds_s) - 1))

    sections: list[dict[str, Any]] = []
    for i in range(len(bounds_s) - 1):
        sections.append(
            {
                "label": motif_labels[i],
                "start_s": float(bounds_s[i]),
                "end_s": float(bounds_s[i + 1]),
            }
        )
    sections = _merge_same_label_sections(sections)

    # --- Drop detection: sharp tension decrease ---
    _DROP_MIN_PRETENSION = 0.60
    _DROP_PRETENSION_WINDOW_S = 2.0
    # Absolute loudness gate: skip drop detection on very quiet signals where
    # spectral centroid leakage causes spurious tension variation.  Typical
    # songs have peak rms >> 0.05; a 0.01-amplitude test sine has rms ≈ 0.007.
    _DROP_MIN_ABSOLUTE_RMS = 0.05

    dt = float(hop_length / sr)
    d = np.diff(tension, prepend=float(tension[0])) / max(dt, 1e-6)

    if float(rms.max()) >= _DROP_MIN_ABSOLUTE_RMS:
        drops_raw = np.flatnonzero(d < -0.65)
        drops: list[int] = []
        for idx in drops_raw:
            lb = max(0, int(np.searchsorted(times_s, float(times_s[idx]) - _DROP_PRETENSION_WINDOW_S)))
            if float(raw_tension_smoothed[lb : idx + 1].max()) >= _DROP_MIN_PRETENSION:
                drops.append(int(idx))
        drop_times = times_s[np.asarray(drops, dtype=int)].astype(float).tolist()[:25] if drops else []
    else:
        drop_times = []

    # --- Buildup detection: for each drop, find the sustained rise before it ---
    _BUILDUP_LOOKBACK_S = 20.0
    _BUILDUP_RISE_THR = 0.30  # d(tension)/dt threshold (units: tension/second)

    buildups: list[dict[str, Any]] = []
    for drop_t in drop_times:
        drop_idx = max(0, int(np.searchsorted(times_s, drop_t, side="left")) - 1)
        lookback_idx = max(0, int(np.searchsorted(times_s, drop_t - _BUILDUP_LOOKBACK_S)))
        if drop_idx <= lookback_idx + 1:
            continue

        window_d = d[lookback_idx:drop_idx + 1]
        rising = np.flatnonzero(window_d > _BUILDUP_RISE_THR)
        if rising.size == 0:
            continue

        buildup_start_idx = lookback_idx + int(rising[0])
        buildups.append(
            {
                "buildup_start_s": float(times_s[buildup_start_idx]),
                "buildup_peak_s": float(times_s[drop_idx]),
                "drop_time_s": float(drop_t),
            }
        )

    return {
        "sections": sections,
        "tension": {
            "hop_s": float(hop_s),
            "times_s": times_s.astype(float).tolist(),
            "value": tension.astype(float).tolist(),
        },
        "events": {
            "drop_times_s": drop_times,
            "buildups": buildups,
        },
        "meta": {
            "duration_s": float(duration_s),
            "sample_rate": int(sr),
            "features": ["mfcc_20", "rms", "onset_strength", "spectral_centroid"],
        },
    }
