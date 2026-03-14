from __future__ import annotations

from typing import Any

import librosa
import numpy as np
import scipy.ndimage
import scipy.stats

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
    target_k: int | None = None,
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
    # (non-maximum suppression).
    if target_k is None:
        target_k = max(2, int(round(duration_s / 55.0)))
    valleys.sort(key=lambda v: v[1], reverse=True)

    # Require minimum absolute depth to reject noise-level valleys
    # (e.g. constant-amplitude signals with near-zero tension variation).
    _min_depth = 0.05
    selected: list[float] = []
    for idx, _depth in valleys:
        if _depth < _min_depth:
            break  # remaining valleys are even shallower
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
    """Merge boundaries that are too close, keeping one representative per cluster.

    When multiple detectors (SSM, energy) place boundaries within *min_len_s*
    of each other, the old per-boundary forward-gap check dropped all of them.
    This version clusters nearby boundaries first, then picks the single best
    representative per cluster — the one that maximizes
    ``min(gap_to_prev_kept, gap_to_next_cluster_start)``.
    """
    if not bounds_s:
        return [0.0, float(duration_s)]
    b = sorted(set(float(x) for x in bounds_s))
    b[0] = 0.0
    b[-1] = float(duration_s)

    internal = [x for x in b if 0.0 < x < duration_s]
    if not internal:
        return [0.0, float(duration_s)]

    # Phase 1: group internal boundaries into clusters.
    # Consecutive boundaries with gap < min_len_s belong to the same cluster.
    clusters: list[list[float]] = [[internal[0]]]
    for i in range(1, len(internal)):
        if internal[i] - clusters[-1][-1] < min_len_s:
            clusters[-1].append(internal[i])
        else:
            clusters.append([internal[i]])

    # Phase 2: pick one representative per cluster.
    out = [0.0]
    for ci, cluster in enumerate(clusters):
        prev = out[-1]
        # Conservative next reference: closest possible boundary after this
        # cluster (start of next cluster, or duration).
        next_ref = clusters[ci + 1][0] if ci + 1 < len(clusters) else duration_s

        best: float | None = None
        best_min_gap = -1.0
        for c in cluster:
            gap_back = c - prev
            gap_fwd = next_ref - c
            min_gap = min(gap_back, gap_fwd)
            if min_gap > best_min_gap:
                best, best_min_gap = c, min_gap

        if best is not None and best_min_gap >= min_len_s:
            out.append(best)

    out.append(float(duration_s))
    return out


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


# ---------------------------------------------------------------------------
# Role-based section labeling
# ---------------------------------------------------------------------------

_ROLES = ("intro", "build", "payoff", "valley", "contrast", "outro")


def _clamp01(x: float) -> float:
    """Clamp *x* to the ``[0, 1]`` interval."""
    return max(0.0, min(1.0, x))


def _section_sim(
    mean_a: np.ndarray, mean_b: np.ndarray,
    energy_a: np.ndarray, energy_b: np.ndarray,
) -> float:
    """Per-block blended similarity for section_means vectors.

    Decomposes the 34-dim vector into MFCC [0:20], chroma [20:32].
    Computes cosine on MFCC/chroma blocks separately,
    L1 distance on min-max normalized energy, then blends.
    """
    a_mfcc, b_mfcc = mean_a[:20], mean_b[:20]
    a_chroma, b_chroma = mean_a[20:32], mean_b[20:32]

    def _cos01(x: np.ndarray, y: np.ndarray) -> float:
        nx = np.linalg.norm(x)
        ny = np.linalg.norm(y)
        if nx < 1e-8 or ny < 1e-8:
            return 0.5
        return (float(np.dot(x, y)) / (nx * ny) + 1.0) / 2.0

    sim_mfcc = _cos01(a_mfcc, b_mfcc)
    sim_chroma = _cos01(a_chroma, b_chroma)

    energy_dist = float(np.abs(energy_a - energy_b).sum())
    sim_energy = max(0.0, 1.0 - energy_dist / 2.0)

    blended = 0.40 * sim_mfcc + 0.20 * sim_chroma + 0.40 * sim_energy
    return max(0.0, min(1.0, blended))


def _compute_section_features(
    bounds_s: list[float],
    *,
    rms01: np.ndarray,
    onset01: np.ndarray,
    cent01: np.ndarray,
    times_s: np.ndarray,
    beat_times: np.ndarray | None,
    section_means: list[np.ndarray],
    duration_s: float,
    hop_s: float,
) -> list[dict[str, float]]:
    """Compute per-section feature vectors for role assignment.

    Returns one dict per section with 11 features, normalized within the
    song for cross-section comparison.
    """
    n_sections = len(bounds_s) - 1
    if n_sections <= 0:
        return []

    n = len(rms01)
    features: list[dict[str, float]] = []

    # --- Local features (raw) ---
    for i in range(n_sections):
        s0, s1 = bounds_s[i], bounds_s[i + 1]
        f0 = max(0, min(int(round(s0 / hop_s)), n - 1))
        f1 = max(f0 + 1, min(int(round(s1 / hop_s)), n))

        rms_sec = rms01[f0:f1]
        onset_sec = onset01[f0:f1]
        cent_sec = cent01[f0:f1]

        mean_rms = float(rms_sec.mean()) if rms_sec.size else 0.0
        onset_density = float(onset_sec.mean()) if onset_sec.size else 0.0
        spectral_centroid_mean = float(cent_sec.mean()) if cent_sec.size else 0.0

        fc = rms_sec.size
        if fc >= 4:
            # Smooth with a ~0.5s window to suppress transient spikes
            win_slope = max(3, int(round(0.5 / max(hop_s, 1e-8))))
            win_slope = min(win_slope, fc)
            if win_slope >= 3:
                kernel = np.ones(win_slope) / win_slope
                rms_smooth = np.convolve(rms_sec, kernel, mode="same")
            else:
                rms_smooth = rms_sec
            quarter = max(1, fc // 4)
            first_q = float(rms_smooth[:quarter].mean())
            last_q = float(rms_smooth[-quarter:].mean())
            raw_delta = last_q - first_q
            rms_slope = raw_delta if abs(raw_delta) > 0.02 else 0.0
        else:
            rms_slope = 0.0

        rms_variance = float(rms_sec.var()) if rms_sec.size else 0.0
        song_position = (s0 + s1) / 2.0 / max(duration_s, 1e-8)

        if beat_times is not None:
            duration_beats = float(
                int(np.searchsorted(beat_times, s1))
                - int(np.searchsorted(beat_times, s0))
            )
        else:
            duration_beats = (s1 - s0) / 0.5

        features.append({
            "mean_rms": mean_rms,
            "onset_density": onset_density,
            "spectral_centroid_mean": spectral_centroid_mean,
            "rms_slope": rms_slope,
            "rms_variance": rms_variance,
            "song_position": song_position,
            "duration_beats": duration_beats,
        })

    if n_sections == 1:
        features[0]["relative_intensity_rank"] = 0.5
    else:
        rms_vals = np.array([f["mean_rms"] for f in features])
        ranks = scipy.stats.rankdata(rms_vals, method="average")
        for i in range(n_sections):
            features[i]["relative_intensity_rank"] = float(
                (ranks[i] - 1) / (n_sections - 1)
            )

    # --- Post-processing: within-song min-max normalization ---
    for key in ("mean_rms", "onset_density", "spectral_centroid_mean", "rms_variance"):
        vals = [f[key] for f in features]
        lo, hi = min(vals), max(vals)
        rng = hi - lo + 1e-8
        for f in features:
            f[key] = (f[key] - lo) / rng

    # rms_slope: center on 0.5 (flat = 0.5, rising > 0.5, falling < 0.5)
    slopes = [f["rms_slope"] for f in features]
    max_abs = max((abs(s) for s in slopes), default=1e-8)
    max_abs = max(max_abs, 1e-8)
    for f in features:
        f["rms_slope"] = float(np.clip(0.5 + 0.5 * f["rms_slope"] / max_abs, 0.0, 1.0))

    # --- Relational features (after normalization so energy is in [0,1]) ---
    energy_vecs = [
        np.array([features[i]["mean_rms"], features[i]["onset_density"]])
        for i in range(n_sections)
    ]

    for i in range(n_sections):
        if n_sections < 3:
            features[i]["repetition_strength"] = 0.0
        else:
            sims = [
                _section_sim(section_means[i], section_means[j],
                             energy_vecs[i], energy_vecs[j])
                for j in range(n_sections)
                if abs(i - j) > 1
            ]
            features[i]["repetition_strength"] = max(sims) if sims else 0.0

        features[i]["novelty_to_prev"] = (
            1.0 if i == 0
            else 1.0 - _section_sim(section_means[i], section_means[i - 1],
                                     energy_vecs[i], energy_vecs[i - 1])
        )
        features[i]["novelty_to_next"] = (
            1.0 if i == n_sections - 1
            else 1.0 - _section_sim(section_means[i], section_means[i + 1],
                                     energy_vecs[i], energy_vecs[i + 1])
        )

    return features


def _assign_roles(features: list[dict[str, float]]) -> list[dict[str, Any]]:
    """Score each section for every role and assign the highest-scoring eligible one."""
    n = len(features)
    results: list[dict[str, Any]] = []

    all_rs = [f["repetition_strength"] for f in features]
    rs_min, rs_max = min(all_rs), max(all_rs)

    for i, f in enumerate(features):
        sp = f["song_position"]
        rir = f["relative_intensity_rank"]
        rs = f["repetition_strength"]
        sl = f["rms_slope"]
        od = f["onset_density"]
        mr = f["mean_rms"]
        scm = f["spectral_centroid_mean"]
        rv = f["rms_variance"]
        ntp = f["novelty_to_prev"]
        ntn = f["novelty_to_next"]

        scores: dict[str, float] = {}

        # intro (constraint: song_position < 0.25)
        if sp < 0.25:
            scores["intro"] = (
                0.40 * (1 - sp)
                + 0.25 * (1 - rir)
                + 0.20 * (1 - rs)
                + 0.15 * sl
            )
        else:
            scores["intro"] = 0.0

        # build — sl used directly (0=decline, 0.5=flat, 1=rise)
        scores["build"] = (
            0.45 * sl
            + 0.30 * (1 - rir)
            + 0.15 * od
            + 0.10 * (1 - rv)
        )
        if i + 1 < n:
            gap = features[i + 1]["relative_intensity_rank"] - rir
            if gap > 0.15:
                scores["build"] += 0.12 * gap

        # payoff (no repetition requirement)
        scores["payoff"] = (
            0.50 * rir
            + 0.20 * mr
            + 0.15 * od
            + 0.15 * scm
        )

        # valley — penalize song-start and strongly rising sections
        valley_base = (
            0.50 * (1 - rir)
            + 0.25 * (1 - od)
            + 0.15 * (1 - mr)
            + 0.10 * (1 - scm)
        )
        # Position penalty: valleys shouldn't open or close the song.
        # Full penalty (0.5x) at sp=0; no penalty from sp=0.2 onward.
        # Mirror for outro zone: full penalty at sp=1.
        pos_pen = min(1.0, sp / 0.2) if sp < 0.2 else min(1.0, (1.0 - sp) / 0.2)
        pos_pen = 0.5 + 0.5 * pos_pen  # range [0.5, 1.0]
        # Slope penalty: a strongly rising section is a build, not a valley.
        # sl=0.5 is flat (no penalty), sl=1.0 is max rise (0.6x).
        slope_pen = 1.0 - 0.4 * max(0.0, sl - 0.5) / 0.5 if sl > 0.5 else 1.0
        scores["valley"] = valley_base * pos_pen * slope_pen

        # contrast (relative novelty within song)
        relative_novelty = (rs_max - rs) / (rs_max - rs_min + 1e-8)
        scores["contrast"] = (
            0.35 * relative_novelty
            + 0.30 * ntp
            + 0.25 * ntn
            + 0.10 * rv
        )

        # outro (constraint: song_position > 0.75)
        if sp > 0.75:
            scores["outro"] = (
                0.40 * sp
                + 0.30 * (1 - sl)
                + 0.20 * (1 - rir)
                + 0.10 * (1 - od)
            )
        else:
            scores["outro"] = 0.0

        sorted_roles = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_role, best_score = sorted_roles[0]
        _second_role, second_score = sorted_roles[1]

        confidence = best_score / (best_score + second_score + 1e-8)

        results.append({
            "role": best_role,
            "confidence": confidence,
            "scores": scores,
            "second_best_role": _second_role,
        })

    return results


def _revise_roles_globally(
    sections: list[dict[str, Any]],
    features: list[dict[str, float]],
    role_assignments: list[dict[str, Any]],
    section_means: list[np.ndarray],
) -> None:
    """Apply global sequence constraints — mutates *sections* in place."""
    n = len(sections)
    if n == 0:
        return

    def _downgrade(idx: int, exclude: set[str] | None = None) -> None:
        ra = role_assignments[idx]
        exclude = exclude or set()
        candidates = sorted(
            ((r, s) for r, s in ra["scores"].items() if r not in exclude and s > 0),
            key=lambda x: x[1],
            reverse=True,
        )
        new_role = candidates[0][0] if candidates else "valley"
        sections[idx]["role"] = new_role
        ra["role"] = new_role

    # 1. Intro only first 1–2 sections.
    for i in range(2, n):
        if sections[i].get("role") == "intro":
            _downgrade(i, exclude={"intro"})

    # 2. Outro only last 1–2 sections.
    for i in range(0, max(0, n - 2)):
        if sections[i].get("role") == "outro":
            _downgrade(i, exclude={"outro"})

    # 3. Build without nearby payoff → downgrade.
    for i in range(n):
        if sections[i].get("role") != "build":
            continue
        has_payoff = any(
            sections[j].get("role") == "payoff"
            for j in range(i + 1, min(i + 3, n))
        )
        if not has_payoff:
            _downgrade(i, exclude={"build"})

    # 4. Repeated-section consistency — removed.
    # MFCC+chroma cosine similarity is too high in pop/rock (most pairs ≥ 0.82),
    # causing cascading that collapses all sections into a single role.
    # Role-based labeling (_assign_role_based_labels) already clusters similar
    # sections within the same role for consistent lettering.

    # 5. At least one payoff (if 3+ sections).
    if n >= 3 and not any(s.get("role") == "payoff" for s in sections):
        best_idx, best_rank = -1, -1.0
        for i in range(n):
            if sections[i].get("role") in ("intro", "outro"):
                continue
            rank = features[i]["relative_intensity_rank"] if i < len(features) else 0.0
            if rank > best_rank:
                best_rank = rank
                best_idx = i
        if best_idx >= 0:
            sections[best_idx]["role"] = "payoff"
            role_assignments[best_idx]["role"] = "payoff"

    # 6. Consecutive identical roles — soft preference only (no forced reassignment).

    # Propagate updated confidence.
    for i in range(n):
        sections[i]["confidence"] = role_assignments[i]["confidence"]


def _resolve_visual_behavior(role: str, idx: int, sections: list[dict[str, Any]]) -> str:
    """Map a role + its position in the sequence to a visual behavior string."""
    if role == "intro":
        return "establish_world"
    if role == "build":
        return "build_tension"
    if role == "payoff":
        for j in range(idx):
            if sections[j].get("role") == "payoff":
                return "sustain_euphoria"
        return "release_payoff"
    if role == "valley":
        total = sections[-1].get("end_s", 1.0)
        mid = (sections[idx]["start_s"] + sections[idx]["end_s"]) / 2.0
        return "develop_motion" if mid / max(total, 1e-8) < 0.5 else "strip_down"
    if role == "contrast":
        return "contrast_reset"
    if role == "outro":
        return "close_out"
    return "develop_motion"


def _assign_role_based_labels(
    sections: list[dict[str, Any]],
    section_means: list[np.ndarray],
    *,
    sec_features: list[dict[str, float]] | None = None,
    similarity_threshold: float = 0.82,
) -> None:
    """Derive A/B/C labels from roles + acoustic similarity.  Mutates *sections*."""
    n = len(sections)
    if n == 0:
        return

    if sec_features is not None:
        energy_vecs = [
            np.array([sf["mean_rms"], sf["onset_density"]])
            for sf in sec_features
        ]
    else:
        energy_vecs = [np.array([0.5, 0.5])] * n

    # Group by role, preserving indices.
    role_groups: dict[str, list[int]] = {}
    for i, sec in enumerate(sections):
        role_groups.setdefault(sec.get("role", "unknown"), []).append(i)

    # Sub-cluster within each role by blended similarity.
    cluster_id = [-1] * n
    next_cluster = 0

    for role in sorted(role_groups, key=lambda r: role_groups[r][0]):
        for idx in role_groups[role]:
            if cluster_id[idx] >= 0:
                continue
            if idx >= len(section_means):
                cluster_id[idx] = next_cluster
                next_cluster += 1
                continue
            cluster_id[idx] = next_cluster
            for other in role_groups[role]:
                if cluster_id[other] >= 0 or other >= len(section_means):
                    continue
                if _section_sim(section_means[idx], section_means[other],
                                energy_vecs[idx], energy_vecs[other]) >= similarity_threshold:
                    cluster_id[other] = next_cluster
            next_cluster += 1

    # Assign letters in song-order of first occurrence.
    label_map: dict[int, str] = {}
    labels = _labels_for_n(next_cluster)
    label_idx = 0
    for i in range(n):
        cid = cluster_id[i]
        if cid not in label_map:
            label_map[cid] = labels[label_idx]
            label_idx += 1
        sections[i]["label"] = label_map[cid]


def _make_subsection(
    start_s: float,
    end_s: float,
    tension: np.ndarray,
    times_s: np.ndarray,
) -> dict[str, Any]:
    """Create a subsection dict with an energy descriptor."""
    mask = (times_s >= start_s) & (times_s < end_s)
    local = tension[mask]

    if local.size == 0:
        return {"start_s": start_s, "end_s": end_s, "energy": "mid"}

    mean_val = float(local.mean())
    # Check for rising/falling trend using first/last quarter.
    quarter = max(1, local.size // 4)
    start_mean = float(local[:quarter].mean())
    end_mean = float(local[-quarter:].mean())

    trend_diff = end_mean - start_mean
    if trend_diff > 0.12:
        energy = "rising"
    elif trend_diff < -0.12:
        energy = "falling"
    elif mean_val < 0.35:
        energy = "low"
    elif mean_val > 0.65:
        energy = "high"
    else:
        energy = "mid"

    return {"start_s": start_s, "end_s": end_s, "energy": energy}


def _detect_subsections(
    section: dict[str, Any],
    tension: np.ndarray,
    times_s: np.ndarray,
    *,
    min_subsection_len_s: float = 8.0,
) -> list[dict[str, Any]]:
    """Detect finer-grained subsections within a section using tension valleys.

    Returns a list of subsection dicts with start_s, end_s, and energy descriptor.
    The subsections always cover the full section range without gaps.
    """
    start_s = float(section["start_s"])
    end_s = float(section["end_s"])
    section_len = end_s - start_s

    # Too short to subdivide.
    if section_len < min_subsection_len_s * 2.5:
        return [_make_subsection(start_s, end_s, tension, times_s)]

    # Extract tension within this section's time range.
    mask = (times_s >= start_s) & (times_s < end_s)
    idx = np.flatnonzero(mask)
    if idx.size < 10:
        return [_make_subsection(start_s, end_s, tension, times_s)]

    local_tension = tension[idx]
    local_times = times_s[idx]

    # Lighter smoothing (4s window, vs 12s for sections).
    n_per_sec = max(1, int(idx.size / max(1e-6, section_len)))
    smooth_win = max(3, int(4.0 * n_per_sec))
    smooth = _smooth_1d(local_tension, win=smooth_win)

    # Find valleys (derivative zero-crossings: neg→pos).
    deriv = np.diff(smooth, prepend=smooth[0])
    sign = np.sign(deriv)

    peak_search = max(1, int(min_subsection_len_s * n_per_sec))
    valleys: list[tuple[int, float]] = []
    for i in range(1, len(sign)):
        if sign[i - 1] < 0 and sign[i] >= 0:
            left_peak = float(smooth[max(0, i - peak_search) : i].max())
            right_peak = float(smooth[i : min(len(smooth), i + peak_search)].max())
            depth = (left_peak + right_peak) / 2.0 - float(smooth[i])
            valleys.append((i, depth))

    if not valleys:
        return [_make_subsection(start_s, end_s, tension, times_s)]

    # Require minimum depth to prevent noise splits.
    section_range = float(smooth.max() - smooth.min())
    min_depth = section_range * 0.15
    valleys = [(vi, d) for vi, d in valleys if d >= min_depth]

    if not valleys:
        return [_make_subsection(start_s, end_s, tension, times_s)]

    # Target ~1 subsection boundary per 25s of section length.
    target_k = max(1, int(round(section_len / 25.0)))
    valleys.sort(key=lambda v: v[1], reverse=True)

    selected_times: list[float] = []
    for v_idx, _depth in valleys:
        t = float(local_times[min(v_idx, len(local_times) - 1)])
        if (t - start_s) < min_subsection_len_s or (end_s - t) < min_subsection_len_s:
            continue
        if all(abs(t - s) >= min_subsection_len_s for s in selected_times):
            selected_times.append(t)
        if len(selected_times) >= target_k:
            break

    if not selected_times:
        return [_make_subsection(start_s, end_s, tension, times_s)]

    selected_times.sort()
    bounds = [start_s] + selected_times + [end_s]

    subsections = []
    for i in range(len(bounds) - 1):
        subsections.append(_make_subsection(bounds[i], bounds[i + 1], tension, times_s))
    return subsections


def _merge_same_label_sections(
    sections: list[dict[str, Any]],
    *,
    max_merged_len_s: float = 120.0,
) -> list[dict[str, Any]]:
    """Merge consecutive sections that share the same role and low novelty.

    Backward-compatible: sections without a ``role`` key fall back to merging
    by matching ``label`` only (original behaviour).
    """
    if not sections:
        return sections
    merged = [sections[0].copy()]
    for sec in sections[1:]:
        prev = merged[-1]
        would_be_len = sec["end_s"] - prev["start_s"]
        has_role = "role" in prev and "role" in sec
        if has_role:
            can_merge = (
                prev["role"] == sec["role"]
                and prev.get("novelty_to_next", 1.0) < 0.3
            )
        else:
            can_merge = sec["label"] == prev["label"]
        if can_merge and would_be_len <= max_merged_len_s:
            merged[-1]["end_s"] = sec["end_s"]
            if "novelty_to_next" in sec:
                merged[-1]["novelty_to_next"] = sec["novelty_to_next"]
        else:
            merged.append(sec.copy())
    return merged


def _beat_sync_features(
    y: np.ndarray,
    sr: int,
    *,
    hop_length: int,
    n_frames: int,
    mfcc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Beat-synchronous chroma + MFCC features with time-delay embedding.

    Returns (C_sync_stacked, M_sync_stacked, C_sync_raw, M_sync_raw, beat_times, beat_duration_s).
    Falls back to uniform 2 Hz pseudo-beats if beat_track produces < 8 beats.
    """
    _tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    if len(beat_frames) < 8:
        pseudo_hop = sr // 2
        beat_frames = np.arange(0, n_frames * hop_length, pseudo_hop) // hop_length
        beat_frames = beat_frames.astype(int)

    beat_frames = librosa.util.fix_frames(beat_frames, x_min=0, x_max=n_frames - 1)

    C = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)

    # Trim/pad so sync doesn't exceed array bounds
    max_f = max(int(beat_frames[-1]) + 1, C.shape[1], mfcc.shape[1])
    if C.shape[1] < max_f:
        C = np.pad(C, ((0, 0), (0, max_f - C.shape[1])))
    if mfcc.shape[1] < max_f:
        mfcc_pad = np.pad(mfcc, ((0, 0), (0, max_f - mfcc.shape[1])))
    else:
        mfcc_pad = mfcc

    C_sync_raw = librosa.util.sync(C, beat_frames, aggregate=np.median)
    M_sync_raw = librosa.util.sync(mfcc_pad, beat_frames, aggregate=np.mean)

    C_sync = librosa.feature.stack_memory(C_sync_raw, n_steps=3, delay=1)
    M_sync = librosa.feature.stack_memory(M_sync_raw, n_steps=3, delay=1)

    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
    beat_duration_s = float(np.median(np.diff(beat_times))) if len(beat_times) > 1 else 0.5

    return C_sync, M_sync, C_sync_raw, M_sync_raw, beat_times, beat_duration_s


def _build_ssm(features: np.ndarray, *, n_path: int = 11) -> np.ndarray:
    """Build an affinity self-similarity matrix with path enhancement."""
    if float(np.std(features)) < 1e-6:
        return np.ones((features.shape[1], features.shape[1]), dtype=np.float64)
    R = librosa.segment.recurrence_matrix(
        features, mode="affinity", sym=True, self=True, full=True,
    )
    # If raw affinity is near-uniform the signal has no structural change;
    # path_enhance would amplify numerical noise into fake structure.
    if float(np.std(R)) < 0.05:
        return np.ones_like(R)
    R = librosa.segment.path_enhance(R, n=n_path)
    return R


def _checkerboard_novelty(R: np.ndarray, *, kernel_width: int) -> np.ndarray:
    """Checkerboard kernel convolution along SSM diagonal. Returns normalized 1D novelty curve."""
    kw = kernel_width
    n = R.shape[0]
    kernel = np.ones((2 * kw, 2 * kw), dtype=np.float64)
    kernel[:kw, kw:] = -1.0
    kernel[kw:, :kw] = -1.0

    novelty = np.zeros(n, dtype=np.float64)
    for i in range(kw, n - kw):
        block = R[i - kw : i + kw, i - kw : i + kw]
        novelty[i] = float(np.sum(block * kernel))

    novelty = np.clip(novelty, 0.0, None)
    max_val = float(novelty.max())
    if max_val > 1e-8:
        novelty /= max_val
    return novelty


def _novelty_boundaries(
    novelty: np.ndarray,
    beat_times: np.ndarray,
    duration_s: float,
    *,
    min_section_s: float = 12.0,
    beat_duration_s: float = 0.5,
) -> list[float]:
    """Peak-pick the novelty curve and return section boundary times."""
    novelty = scipy.ndimage.median_filter(novelty.astype(np.float64), size=3)

    n_beats = len(beat_times)
    pre_max = post_max = max(2, n_beats // 40)
    pre_avg = post_avg = max(4, n_beats // 20)
    wait = max(4, int(min_section_s / max(beat_duration_s, 1e-6)))
    delta = max(0.05, 0.5 * float(np.std(novelty)))

    peaks = librosa.util.peak_pick(
        novelty,
        pre_max=pre_max,
        post_max=post_max,
        pre_avg=pre_avg,
        post_avg=post_avg,
        delta=delta,
        wait=wait,
    )

    if len(peaks) == 0:
        return [0.0, duration_s]

    valid_peaks = peaks[peaks < len(beat_times)]
    if len(valid_peaks) == 0:
        return [0.0, duration_s]

    peak_times = beat_times[valid_peaks].tolist()
    return [0.0] + sorted(peak_times) + [duration_s]


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

    # --- Coarse segmentation via SSM + checkerboard novelty ---
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=hop_length, n_fft=frame_length)
    mu = np.mean(mfcc, axis=1, keepdims=True)
    sig = np.std(mfcc, axis=1, keepdims=True) + 1e-6
    mfcc_z = (mfcc - mu) / sig

    _ssm_ok = False
    C_sync_raw: np.ndarray | None = None
    M_sync_raw: np.ndarray | None = None
    beat_times: np.ndarray | None = None

    try:
        # Only run SSM if the signal has meaningful temporal variation.
        # A flat waveform (constant sine, silence) produces degenerate SSMs
        # where numerical noise gets amplified into fake structure.
        _has_structure = float(np.std(onset01)) > 0.08 or float(np.std(rms01)) > 0.08

        C_sync, M_sync, C_sync_raw, M_sync_raw, beat_times, beat_dur = _beat_sync_features(
            y, sr, hop_length=hop_length, n_frames=n, mfcc=mfcc,
        )
        n_beats = C_sync.shape[-1]

        if _has_structure:
            R_chroma = _build_ssm(C_sync)
            R_mfcc = _build_ssm(M_sync)

            kw = int(np.clip(n_beats // 20, 4, 32))
            nov_chroma = _normalize_01(_checkerboard_novelty(R_chroma, kernel_width=kw))
            nov_mfcc = _normalize_01(_checkerboard_novelty(R_mfcc, kernel_width=kw))
            novelty = 0.55 * nov_chroma + 0.45 * nov_mfcc

            bounds_ssm = _novelty_boundaries(
                novelty, beat_times, duration_s, beat_duration_s=beat_dur,
            )
        else:
            bounds_ssm = [0.0, duration_s]

        # Energy-based boundaries: the "obvious" ones visible in the waveform
        # (tension dips where onset/loudness drop). These are what a human sees
        # first when looking at overview.png.
        energy_target_k = max(3, int(round(duration_s / 35.0)))
        bounds_energy = _tension_valley_boundaries(
            raw_tension_smoothed, times_s,
            min_len_s=12.0, duration_s=duration_s,
            target_k=energy_target_k,
        )

        # Combine: union of harmonic (SSM) and energy (tension) boundaries
        internal = sorted(set(
            [b for b in bounds_ssm if 0 < b < duration_s]
            + [b for b in bounds_energy if 0 < b < duration_s]
        ))
        bounds_s = [0.0] + internal + [duration_s]
        bounds_s = _merge_short_segments(bounds_s, min_len_s=12.0, duration_s=duration_s)

        _ssm_ok = True

    except Exception:
        # Emergency fallback: agglomerative clustering (original approach)
        n_seg_frames = min(mfcc_z.shape[1], len(rms01), len(onset01))
        mfcc_z_seg = mfcc_z[:, :n_seg_frames]
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

        if len(bounds_s) <= 3 and duration_s > 60.0:
            bounds_s = _tension_valley_boundaries(
                raw_tension_smoothed, times_s, min_len_s=15.0, duration_s=duration_s,
            )

    # --- Motif-aware labels ---
    hop_s = float(hop_length / sr)
    section_means: list[np.ndarray] = []

    if _ssm_ok and C_sync_raw is not None and M_sync_raw is not None and beat_times is not None:
        # Beat-sync chroma + MFCC + energy for richer motif comparison.
        # Energy features ensure quiet (intro) and loud (chorus) sections
        # get different labels even when timbre/harmony is similar.
        for i in range(len(bounds_s) - 1):
            b0 = int(np.searchsorted(beat_times, bounds_s[i]))
            b1 = int(np.searchsorted(beat_times, bounds_s[i + 1]))
            b0 = min(b0, C_sync_raw.shape[1] - 1)
            b1 = max(b0 + 1, min(b1, C_sync_raw.shape[1]))
            chroma_mean = C_sync_raw[:, b0:b1].mean(axis=1)   # 12-dim
            mfcc_mean = M_sync_raw[:, b0:b1].mean(axis=1)     # 20-dim
            # Frame-level energy for this section
            s0_f = int(round(bounds_s[i] / hop_s))
            s1_f = int(round(bounds_s[i + 1] / hop_s))
            e0 = max(0, min(s0_f, n - 1))
            e1 = max(e0 + 1, min(s1_f, n))
            rms_mean = float(rms01[e0:e1].mean())
            onset_mean = float(onset01[e0:e1].mean())
            section_means.append(np.concatenate([
                mfcc_mean, chroma_mean * 2.0, [rms_mean * 5.0, onset_mean * 3.0],
            ]))
    else:
        # Fallback: frame-level MFCC + energy
        for i in range(len(bounds_s) - 1):
            s0_f = int(round(bounds_s[i] / hop_s))
            s1_f = int(round(bounds_s[i + 1] / hop_s))
            s0_f = max(0, min(s0_f, mfcc_z.shape[1]))
            s1_f = max(s0_f + 1, min(s1_f, mfcc_z.shape[1]))
            mfcc_mean = mfcc_z[:, s0_f:s1_f].mean(axis=1)
            e0 = max(0, min(s0_f, n - 1))
            e1 = max(e0 + 1, min(s1_f, n))
            rms_mean = float(rms01[e0:e1].mean())
            onset_mean = float(onset01[e0:e1].mean())
            section_means.append(np.concatenate([mfcc_mean, [rms_mean * 5.0, onset_mean * 3.0]]))

    # --- Role-based section labeling ---
    sections: list[dict[str, Any]] = []
    for i in range(len(bounds_s) - 1):
        sections.append({
            "start_s": float(bounds_s[i]),
            "end_s": float(bounds_s[i + 1]),
        })

    sec_features = _compute_section_features(
        bounds_s,
        rms01=rms01, onset01=onset01, cent01=cent01,
        times_s=times_s, beat_times=beat_times,
        section_means=section_means,
        duration_s=duration_s, hop_s=hop_s,
    )

    role_assignments = _assign_roles(sec_features)

    for i, sec in enumerate(sections):
        if i < len(role_assignments):
            sec["role"] = role_assignments[i]["role"]
            sec["confidence"] = role_assignments[i]["confidence"]
        if i < len(sec_features):
            sf = sec_features[i]
            sec["intensity"] = sf["mean_rms"]
            sec["repetition_strength"] = sf["repetition_strength"]
            sec["novelty_to_prev"] = sf["novelty_to_prev"]
            sec["novelty_to_next"] = sf["novelty_to_next"]
            sec["relative_intensity_rank"] = sf["relative_intensity_rank"]

    _revise_roles_globally(sections, sec_features, role_assignments, section_means)
    _assign_role_based_labels(sections, section_means, sec_features=sec_features)
    sections = _merge_same_label_sections(sections, max_merged_len_s=30.0)

    # --- Subsection detection within each section ---
    for sec in sections:
        sec["subsections"] = _detect_subsections(sec, tension, times_s)

    # --- Resolve visual behavior (after merge) ---
    for i, sec in enumerate(sections):
        if "role" in sec:
            sec["visual_behavior"] = _resolve_visual_behavior(sec["role"], i, sections)

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
