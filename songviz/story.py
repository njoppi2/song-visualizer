from __future__ import annotations

from typing import Any

import librosa
import numpy as np
import scipy.ndimage
import scipy.stats

from .ingest import _normalize_01
from .structure_grid import prepare_beat_grid


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
            intro_base = (
                0.40 * (1 - sp)
                + 0.25 * (1 - rir)
                + 0.20 * (1 - rs)
                + 0.15 * sl
            )
            # Bonus for first section that is quiet — strong signal it's an intro,
            # not a build.  A quiet opener (rir < 0.3) that comes before the main
            # content is almost always intro or valley; boost intro to win.
            if i == 0 and rir < 0.3:
                intro_base += 0.15 * (0.3 - rir) / 0.3
            scores["intro"] = intro_base
        else:
            scores["intro"] = 0.0

        # build — sl used directly (0=decline, 0.5=flat, 1=rise)
        build_base = (
            0.45 * sl
            + 0.30 * (1 - rir)
            + 0.15 * od
            + 0.10 * (1 - rv)
        )
        if i + 1 < n:
            gap = features[i + 1]["relative_intensity_rank"] - rir
            if gap > 0.15:
                build_base += 0.12 * gap
        # Duration penalty for build: a very short first section (<15s) is more
        # likely an intro than a build.  Sections are normalized so we proxy
        # duration via song_position × estimated_duration.
        # Use duration_beats feature (already computed, in raw beats).
        dur_beats = f.get("duration_beats", 32.0)
        if i == 0 and dur_beats < 24:
            # Scale linearly: 0 beats → 0.5x, 24 beats → 1.0x.
            # Intro sections are typically 4-6 bars (16-24 beats). A rising first
            # section shorter than 24 beats is more likely an intro than a build.
            build_base *= max(0.5, dur_beats / 24.0)
        # Quiet-island penalty: if this section is the absolute quietest in the
        # song (rir == 0), it can't be a "build" — true builds have baseline
        # energy that they're escalating from.  Only rir=0 is targeted so that
        # other low-energy sections (verse, rap verse) are unaffected.
        if rir < 0.01:  # catches only rank-0 (the single quietest section)
            build_base = 0.0
        scores["build"] = build_base

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
        # Use novelty_to_prev only when there actually IS a previous section.
        # For section 0 ntp is stored as 1.0 (sentinel), which would incorrectly
        # boost contrast for the opening section.
        effective_ntp = ntp if i > 0 else 0.0
        contrast_base = (
            0.35 * relative_novelty
            + 0.30 * effective_ntp
            + 0.25 * ntn
            + 0.10 * rv
        )
        # Energy-dip bonus: if this section's energy is noticeably lower than
        # both its neighbours, it's likely a break/bridge (contrast/valley).
        # This helps detect windmill/bridge sections that have rising internal
        # slope but are quieter than the surrounding choruses.
        if i > 0 and i < n - 1:
            prev_rms = features[i - 1]["mean_rms"]
            next_rms = features[i + 1]["mean_rms"]
            neighbor_avg = (prev_rms + next_rms) / 2.0
            energy_dip = max(0.0, neighbor_avg - mr)
            contrast_base += 0.20 * energy_dip
        # Position gate: a contrast section needs something before it to contrast
        # against.  Ramp from 0 at sp=0 to full weight at sp=0.10.
        if sp < 0.10:
            contrast_base *= sp / 0.10
        scores["contrast"] = contrast_base

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

    # Quiet-start detection: if the section's opening is significantly below its
    # mean tension, insert a split at the first upward crossing.  This catches
    # the "sparse opening before the groove rebuilds" pattern that tension-valley
    # detection misses because the derivative is monotonically rising at the start.
    quiet_frac = 0.20
    q_idx = max(1, int(quiet_frac * len(smooth)))
    q_mean = float(smooth[:q_idx].mean())
    sec_mean = float(smooth.mean())
    cross_level = sec_mean - 0.15 * section_range
    if q_mean < cross_level and section_len >= min_subsection_len_s * 2.0:
        for ci in range(q_idx, len(smooth)):
            if smooth[ci] >= sec_mean:
                t = float(local_times[min(ci, len(local_times) - 1)])
                if (t - start_s) >= min_subsection_len_s and (end_s - t) >= min_subsection_len_s:
                    if all(abs(t - s) >= min_subsection_len_s for s in selected_times):
                        selected_times.append(t)
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
    beat_frames_override: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Beat-synchronous chroma + MFCC features with time-delay embedding.

    Returns (C_sync_stacked, M_sync_stacked, C_sync_raw, M_sync_raw, beat_times, beat_duration_s).
    Falls back to uniform 2 Hz pseudo-beats if beat_track produces < 8 beats.
    """
    if beat_frames_override is None:
        grid = prepare_beat_grid(y, sr, hop_length=hop_length, n_frames=n_frames)
        beat_frames = np.asarray(grid["frame_indices"], dtype=int)
    else:
        beat_frames = np.asarray(beat_frames_override, dtype=int)

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


def _bar_phase_similarity_diagnostic(
    Q_beat: np.ndarray | None,
    rms_beat: np.ndarray | None,
    *,
    beats_per_bar: int = 4,
    parent_beats: int = 16,
    min_samples: int = 6,
    min_margin: float = 0.03,
) -> dict[str, Any]:
    """Weak internal fallback for bar phase from Spec Old 8-in-16 contrast.

    This heuristic is intentionally demoted: on real songs it can pick the
    wrong phase because phrase-similarity contrast is not the same signal as
    musical downbeat evidence. Prefer external downbeat trackers (Beat This or
    madmom) whenever available; keep this only as a diagnostic/fallback.
    """
    method = "internal_weak_similarity_fallback"
    if beats_per_bar <= 1 or parent_beats <= 1:
        return {
            "beats_per_bar": int(max(1, beats_per_bar)),
            "phase": 0,
            "estimated_phase": 0,
            "confidence": 0.0,
            "accepted": False,
            "scores": {},
            "sample_counts": {},
            "method": method,
            "warning": "weak internal fallback; prefer external downbeat tracker",
        }

    if Q_beat is None or rms_beat is None:
        return {
            "beats_per_bar": int(beats_per_bar),
            "phase": 0,
            "estimated_phase": 0,
            "confidence": 0.0,
            "accepted": False,
            "scores": {str(i): 0.0 for i in range(beats_per_bar)},
            "sample_counts": {str(i): 0 for i in range(beats_per_bar)},
            "method": method,
            "warning": "weak internal fallback; prefer external downbeat tracker",
        }

    Q = np.log1p(np.asarray(Q_beat, dtype=np.float64))
    rms = np.asarray(rms_beat, dtype=np.float64).ravel()
    if Q.ndim != 2 or Q.shape[1] < parent_beats * 2 or rms.size < parent_beats * 2:
        return {
            "beats_per_bar": int(beats_per_bar),
            "phase": 0,
            "estimated_phase": 0,
            "confidence": 0.0,
            "accepted": False,
            "scores": {str(i): 0.0 for i in range(beats_per_bar)},
            "sample_counts": {str(i): 0 for i in range(beats_per_bar)},
            "method": method,
            "warning": "weak internal fallback; prefer external downbeat tracker",
        }

    n_beats = min(int(Q.shape[1]), int(rms.size))
    Q = Q[:, :n_beats]
    rms = rms[:n_beats]
    audible_floor = float(rms.max()) * (10.0 ** (_PHRASE_AUDIBLE_FLOOR_DB / 20.0))
    audibility = np.clip((rms - audible_floor) / max(float(rms.max()) - audible_floor, 1e-12), 0.0, 1.0)
    scores: dict[int, float] = {}
    sample_counts: dict[int, int] = {}

    def spec_old_similarity(prv_s: int, cur_s: int, length: int) -> float:
        if length <= 0:
            return 1.0
        Q_prv = Q[:, prv_s:prv_s + length]
        Q_cur = Q[:, cur_s:cur_s + length]
        weights = np.maximum(
            audibility[prv_s:prv_s + length],
            audibility[cur_s:cur_s + length],
        )
        if float(weights.max()) < _PHRASE_COVERAGE_THRESHOLD:
            return 1.0
        return float(np.clip(_weighted_cosine_01(Q_cur, Q_prv, weights), 0.0, 1.0))

    for phase in range(beats_per_bar):
        sims: list[float] = []
        for prv_s, cur_s, length in _matched_half_block_pairs(n_beats, phase, parent_beats):
            sims.append(spec_old_similarity(prv_s, cur_s, length))
        sample_counts[phase] = len(sims)
        if len(sims) < min_samples:
            scores[phase] = 0.0
        else:
            arr = np.asarray(sims, dtype=np.float64)
            scores[phase] = float(np.percentile(arr, 80) - np.percentile(arr, 20))

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    estimated_phase = int(ranked[0][0]) if ranked else 0
    best = float(ranked[0][1]) if ranked else 0.0
    second = float(ranked[1][1]) if len(ranked) > 1 else 0.0
    confidence = max(0.0, best - second)
    accepted = sample_counts.get(estimated_phase, 0) >= min_samples and confidence >= min_margin
    phase = estimated_phase if accepted else 0

    return {
        "beats_per_bar": int(beats_per_bar),
        "phase": int(phase),
        "estimated_phase": int(estimated_phase),
        "confidence": float(confidence),
        "accepted": bool(accepted),
        "scores": {str(k): float(v) for k, v in scores.items()},
        "sample_counts": {str(k): int(v) for k, v in sample_counts.items()},
        "method": method,
        "warning": "weak internal fallback; prefer external downbeat tracker",
    }


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


_MAX_SECTION_S = 60.0  # Force-split sections longer than this


def _detect_intro_onset_boundary(
    tension: np.ndarray,
    times_s: np.ndarray,
    *,
    duration_s: float,
    quiet_window_s: float = 5.0,
    quiet_threshold: float = 0.25,
    active_threshold: float = 0.35,
    max_intro_fraction: float = 0.25,
) -> float | None:
    """Detect intro-end boundary: first time tension rises from a quiet start.

    If the song's first ``quiet_window_s`` seconds have mean normalized tension
    below ``quiet_threshold``, scan forward for the first frame where tension
    exceeds ``active_threshold``.  Returns that time, or None if the song does
    not start quietly or no crossing is found within the first
    ``max_intro_fraction`` of the song.

    ``tension`` must already be normalized to [0, 1].
    """
    if times_s.size < 10:
        return None

    quiet_mask = times_s < quiet_window_s
    if not quiet_mask.any():
        return None
    if float(tension[quiet_mask].mean()) >= quiet_threshold:
        return None  # Song doesn't start quiet — no intro onset to detect

    max_search_t = duration_s * max_intro_fraction
    search_mask = (times_s >= quiet_window_s) & (times_s < max_search_t)
    if not search_mask.any():
        return None

    search_tension = tension[search_mask]
    search_times = times_s[search_mask]

    crossing = np.flatnonzero(search_tension >= active_threshold)
    if crossing.size == 0:
        return None
    return float(search_times[crossing[0]])


def _force_split_long_sections(
    bounds_s: list[float],
    *,
    tension: np.ndarray,
    times_s: np.ndarray,
    max_section_s: float = _MAX_SECTION_S,
    min_section_s: float = 12.0,
) -> list[float]:
    """Insert a split at the deepest tension valley in any section > max_section_s.

    Iterates until no section exceeds the limit (or no valid valley is found).
    Uses the sliced segment tension so the smoothing window is correctly scaled.
    Falls back to a midpoint split if no tension valley is found.
    """
    out = list(bounds_s)
    changed = True
    while changed:
        changed = False
        new_bounds: list[float] = [out[0]]
        for i in range(len(out) - 1):
            seg_start = out[i]
            seg_end = out[i + 1]
            seg_len = seg_end - seg_start
            if seg_len > max_section_s:
                # Slice tension to the segment so smoothing window is correct.
                mask = (times_s >= seg_start) & (times_s < seg_end)
                seg_tension = tension[mask]
                seg_times = times_s[mask]

                split_t: float | None = None
                if seg_times.size >= 10:
                    splits = _tension_valley_boundaries(
                        seg_tension, seg_times,
                        min_len_s=min_section_s,
                        duration_s=float(seg_times[-1]),
                        target_k=1,
                    )
                    # Keep internal boundaries (excluding segment endpoints).
                    valid = [
                        t for t in splits
                        if seg_start + min_section_s < t < seg_end - min_section_s
                    ]
                    if valid:
                        mid = (seg_start + seg_end) / 2.0
                        split_t = min(valid, key=lambda t: abs(t - mid))

                # No valley found — leave the section unsplit.
                # A long section with no tension valley has uniform energy;
                # splitting at the midpoint would create a false boundary.
                if split_t is not None:
                    new_bounds.append(split_t)
                    changed = True
            new_bounds.append(seg_end)
        out = sorted(set(new_bounds))
    return out


def _score_and_filter_boundaries(
    bounds_ssm: list[float],
    bounds_energy: list[float],
    *,
    novelty: np.ndarray | None,
    beat_times: np.ndarray | None,
    duration_s: float,
    agreement_window_s: float = 5.0,
    ssm_prominence_thr: float = 0.50,
) -> list[float]:
    """Score boundaries by cross-detector agreement and filter weak singles.

    - A one-to-one SSM/energy pair within *agreement_window_s* is counted as
      "agreed" and contributes the SSM timestamp only.  SSM represents the
      acoustic structural change, while an energy valley is a delayed or
      advanced tension minimum; keeping the SSM time therefore makes one pair
      represent one transition.
    - SSM-only boundaries are kept only if the novelty curve peak at that
      boundary exceeds *ssm_prominence_thr*.
    - Energy-only boundaries retain the existing keep policy. An energy change
      is evidence to review, not proof of a section transition.

    Candidate pairs are greedily selected from smallest time difference to
    largest, with timestamps breaking ties.  This deterministic one-to-one
    matching deliberately avoids transitive clustering: an energy boundary
    cannot make multiple nearby SSM peaks all look agreed.

    Returns a sorted list of internal boundary times (excludes 0 and duration_s).
    """
    # Sanitizing here prevents repeated detector candidates and endpoints from
    # participating in matching or leaking into the returned internal bounds.
    ssm_internal = sorted({b for b in bounds_ssm if 0 < b < duration_s})
    energy_internal = sorted({b for b in bounds_energy if 0 < b < duration_s})

    # Pair each candidate at most once.  Sorting all eligible edges by distance
    # gives the closest available counterpart first; the timestamps make an
    # equal-distance choice reproducible.
    candidate_pairs = sorted(
        (abs(ssm_t - energy_t), ssm_t, energy_t)
        for ssm_t in ssm_internal
        for energy_t in energy_internal
        if abs(ssm_t - energy_t) <= agreement_window_s
    )
    matched_ssm: set[float] = set()
    matched_energy: set[float] = set()
    for _distance, ssm_t, energy_t in candidate_pairs:
        if ssm_t not in matched_ssm and energy_t not in matched_energy:
            matched_ssm.add(ssm_t)
            matched_energy.add(energy_t)

    # Build novelty lookup: SSM boundary → peak novelty within ±0.5s
    def _novelty_at(t: float) -> float:
        if novelty is None or beat_times is None:
            return 1.0  # no info → keep
        # Find the nearest beat frame to t
        idx = int(np.searchsorted(beat_times, t))
        # Search ±2 beat frames for peak novelty
        lo = max(0, idx - 2)
        hi = min(len(novelty) - 1, idx + 2)
        return float(novelty[lo : hi + 1].max())

    kept: set[float] = set()

    for t in ssm_internal:
        if t in matched_ssm:
            kept.add(t)  # agreed pair — SSM is the representative timestamp
        elif _novelty_at(t) >= ssm_prominence_thr:
            kept.add(t)  # strong SSM-only boundary

    for t in energy_internal:
        if t not in matched_energy:
            # Always keep energy-only boundaries — they're visible in the waveform.
            kept.add(t)

    # --- Intro-end rescue ---
    # Intro→verse boundaries are often feature-subtle: the SSM detects the
    # groove onset (drums entering, bass locking in) but novelty is below the
    # main threshold because the timbral change is gradual in beat-sync space.
    # If nothing survived in the first 25% of the song, rescue the strongest
    # SSM candidate there that clears a relaxed novelty floor of 0.20.
    early_cutoff = duration_s * 0.25
    if not any(t < early_cutoff for t in kept):
        early_candidates = [
            (t, _novelty_at(t))
            for t in ssm_internal
            if t < early_cutoff and _novelty_at(t) >= 0.20
        ]
        if early_candidates:
            best_t, _ = max(early_candidates, key=lambda x: x[1])
            kept.add(best_t)

    return sorted(kept)


_LAG_MAX_BEATS = 32
_NOVELTY_SCALES: dict[str, int] = {"short": 4, "medium": 16, "long": 32}


def _lag_matrix_novelty(
    features: np.ndarray,
    beat_times: np.ndarray,
    frame_times: np.ndarray,
    lag_max: int | None = None,
) -> np.ndarray:
    """Time-lag similarity matrix.

    L[lag-1, t] = cosine_similarity(features[:, t], features[:, t-lag]).

    High values (bright) = current beat strongly resembles the beat `lag` beats
    ago, which indicates a repeating riff. Low values (dark) = novelty/change.
    Rows run from lag=1 (top, small scale) to lag=lag_max (bottom, large scale).
    Cosines are clipped to [0, 1], never independently stretched by lag/song.
    Missing history is stored as zero for serialization; consumers must use
    _lag_history_mask to distinguish unavailable comparisons from dissimilarity.
    """
    if lag_max is None:
        lag_max = _LAG_MAX_BEATS
    F = np.asarray(features, dtype=np.float64)
    n_b = F.shape[1]
    norms = np.linalg.norm(F, axis=0)
    result = np.zeros((lag_max, n_b), dtype=np.float64)

    for lag in range(1, lag_max + 1):
        for t in range(lag, n_b):
            denom = norms[t] * norms[t - lag]
            if denom > 1e-12:
                result[lag - 1, t] = np.clip(np.dot(F[:, t], F[:, t - lag]) / denom, 0.0, 1.0)
            elif norms[t] <= 1e-6 and norms[t - lag] <= 1e-6:
                result[lag - 1, t] = 1.0  # unchanged empty features, not a new event

    out = np.zeros((lag_max, len(frame_times)), dtype=np.float32)
    for i in range(lag_max):
        out[i] = np.interp(frame_times, beat_times, result[i])
    return out


def _lag_history_mask(
    beat_times: np.ndarray, frame_times: np.ndarray, lag_max: int = _LAG_MAX_BEATS,
) -> np.ndarray:
    """Which lag comparisons have past context at each output timestamp."""
    mask = np.zeros((lag_max, len(frame_times)), dtype=bool)
    for lag in range(1, min(lag_max + 1, len(beat_times))):
        mask[lag - 1] = frame_times >= beat_times[lag]
    return mask


def _novelty_curves_from_lag(
    L_chroma: np.ndarray,
    L_mfcc: np.ndarray,
    scales: dict[str, int] | None = None,
    *,
    history_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Derive 1D novelty curves from time-lag similarity matrices.

    novelty(t, S) = 1 - max over lag in [1, S] of L[lag, t]

    "How unlike the current beat is from its closest match in the last S beats."
    Riffs that repeat within window S → low novelty. Genuinely new content with
    no past match → high novelty. Decays as the new pattern starts repeating.
    Missing history is unknown (neutral zero), not novelty. Production callers
    supply history_mask and serialize validity. No per-song min/max rescaling.
    """
    if scales is None:
        scales = _NOVELTY_SCALES
    L = 0.5 * np.asarray(L_chroma, dtype=np.float64) + 0.5 * np.asarray(L_mfcc, dtype=np.float64)
    out: dict[str, np.ndarray] = {}
    for name, S in scales.items():
        s_clip = min(S, L.shape[0])
        if history_mask is None:
            nov = 1.0 - L[:s_clip].max(axis=0)
        else:
            valid = history_mask[:s_clip]
            best = np.where(valid, L[:s_clip], -np.inf).max(axis=0)
            nov = np.where(valid.any(axis=0), 1.0 - best, 0.0)
        nov = np.clip(nov, 0.0, 1.0)
        out[name] = nov.astype(np.float32)
    return out


def _rep_strength(
    R_chroma: np.ndarray,
    R_mfcc: np.ndarray,
    beat_times: np.ndarray,
    frame_times: np.ndarray,
    *,
    min_lag_beats: int = 4,
) -> np.ndarray:
    """Per-beat mean SSM similarity to non-adjacent beats, interpolated to frame times."""
    R = 0.55 * np.asarray(R_chroma, dtype=np.float64) + 0.45 * np.asarray(R_mfcc, dtype=np.float64)
    n_b = R.shape[0]
    rep_beats = np.zeros(n_b)
    for i in range(n_b):
        mask = np.ones(n_b, dtype=bool)
        mask[max(0, i - min_lag_beats): min(n_b, i + min_lag_beats + 1)] = False
        if mask.sum() > 0:
            rep_beats[i] = R[i, mask].mean()
    mn, mx = rep_beats.min(), rep_beats.max()
    if mx - mn > 1e-6:
        rep_beats = (rep_beats - mn) / (mx - mn)
    return np.interp(frame_times, beat_times, rep_beats).astype(np.float32)


def _stem_novelties(
    stem_y: np.ndarray,
    sr: int,
    n_frames: int,
    hop_length: int,
    frame_length: int,
    beat_times: np.ndarray | None = None,
    frame_times: np.ndarray | None = None,
    stem_name: str = "",
    block_offsets_override: dict[int, int] | None = None,
    bar_grid_override: dict[str, Any] | None = None,
    alignment_source_stem: str | None = None,
) -> dict[str, Any]:
    """Compute energy + time-lag harmonic/timbre similarity for a single stem."""
    rms = librosa.feature.rms(y=stem_y, frame_length=frame_length, hop_length=hop_length, center=True)[0]
    energy = _normalize_01(rms[:n_frames]).astype(float).tolist()

    if frame_times is None:
        frame_times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_length)

    C = librosa.feature.chroma_cqt(y=stem_y, sr=sr, hop_length=hop_length)
    M = librosa.feature.mfcc(y=stem_y, sr=sr, n_mfcc=20, hop_length=hop_length, n_fft=frame_length)
    Q_mag = np.abs(librosa.cqt(y=stem_y, sr=sr, hop_length=hop_length,
                                n_bins=84, bins_per_octave=12,
                                fmin=librosa.note_to_hz("C1")))

    if beat_times is not None and len(beat_times) > 1:
        bf = librosa.time_to_frames(beat_times, sr=sr, hop_length=hop_length)
        C_sync = librosa.util.sync(C, bf, aggregate=np.median)
        M_sync = librosa.util.sync(M, bf, aggregate=np.mean)
        Q_sync = librosa.util.sync(Q_mag, bf, aggregate=np.mean)
        R_sync = librosa.util.sync(rms.reshape(1, -1), bf, aggregate=np.mean)[0]
        n_sync = min(
            C_sync.shape[1],
            M_sync.shape[1],
            Q_sync.shape[1],
            R_sync.shape[0],
            len(beat_times),
            len(bf),
        )
        C_sync = C_sync[:, :n_sync]
        M_sync = M_sync[:, :n_sync]
        Q_sync = Q_sync[:, :n_sync]
        R_sync = R_sync[:n_sync]
        _bt = beat_times[:n_sync]
        bf = bf[:n_sync]
    else:
        _bt = np.arange(0, float(frame_times[-1]) + 0.5, 0.5)
        bf = librosa.time_to_frames(_bt, sr=sr, hop_length=hop_length)
        C_sync = librosa.util.sync(C, bf, aggregate=np.median)
        M_sync = librosa.util.sync(M, bf, aggregate=np.mean)
        Q_sync = librosa.util.sync(Q_mag, bf, aggregate=np.mean)
        R_sync = librosa.util.sync(rms.reshape(1, -1), bf, aggregate=np.mean)[0]
        n_sync = min(C_sync.shape[1], M_sync.shape[1], Q_sync.shape[1], R_sync.shape[0], len(_bt), len(bf))
        C_sync = C_sync[:, :n_sync]
        M_sync = M_sync[:, :n_sync]
        Q_sync = Q_sync[:, :n_sync]
        R_sync = R_sync[:n_sync]
        _bt = _bt[:n_sync]
        bf = bf[:n_sync]

    h = _lag_matrix_novelty(C_sync, _bt, frame_times)
    t = _lag_matrix_novelty(M_sync[1:] if M_sync.shape[0] > 1 else M_sync, _bt, frame_times)
    history_mask = _lag_history_mask(_bt, frame_times)
    curves = _novelty_curves_from_lag(h, t, history_mask=history_mask)

    Q_sync = Q_sync[:, : C_sync.shape[1]]

    # Weak fallback only. This does not use true downbeat evidence and has been
    # observed to choose bad phases; external downbeat trackers should override
    # it when integrated.
    local_bar_grid = _bar_phase_similarity_diagnostic(Q_sync, R_sync)
    if block_offsets_override is None:
        stem_bar_grid = local_bar_grid
        block_offsets = _stem_block_offsets(
            R_sync,
            _STEM_COMPARISON_LAGS,
            bar_phase=int(stem_bar_grid.get("phase", 0)),
        )
    else:
        stem_bar_grid = dict(bar_grid_override or local_bar_grid)
        block_offsets = {
            int(lag): int(block_offsets_override.get(int(lag), 0))
            for lag in _STEM_COMPARISON_LAGS
        }
    median_beat_s = float(np.median(np.diff(_bt))) if len(_bt) > 1 else 0.5

    def span_end(beat_idx: int) -> float:
        if beat_idx < len(_bt):
            return float(_bt[beat_idx])
        return float(_bt[-1]) + max(0, beat_idx - len(_bt) + 1) * median_beat_s

    block_spans: dict[str, list[dict[str, float]]] = {}
    for lag in _STEM_COMPARISON_LAGS:
        offset = _block_offset_for_lag(block_offsets, lag)
        block_spans[f"{lag}b"] = [
            {"start_s": float(_bt[cur_s]), "end_s": span_end(cur_s + length)}
            for _prv_s, cur_s, length in _full_block_pairs(len(_bt), offset, lag)
        ]
        child = max(1, lag // 2)
        block_spans[f"{child}-in-{lag}"] = [
            {"start_s": float(_bt[cur_s]), "end_s": span_end(cur_s + length)}
            for _prv_s, cur_s, length in _matched_half_block_pairs(len(_bt), offset, lag)
        ]

    cqt = _cqt_similarity_curves(Q_sync, _bt, frame_times, block_offsets=block_offsets)
    phrase = _phrase_similarity_curves(Q_sync, rms, bf, _bt, frame_times, block_offsets=block_offsets)
    onset = _onset_similarity_curves(
        Q_mag,
        rms,
        bf,
        _bt,
        frame_times,
        sr=sr,
        hop_length=hop_length,
        block_offsets=block_offsets,
    )

    return {
        "energy": energy,
        "bar_grid": stem_bar_grid,
        "local_bar_grid": local_bar_grid,
        "alignment_source_stem": alignment_source_stem or stem_name,
        "block_offsets": {str(k): int(v) for k, v in block_offsets.items()},
        "block_spans": block_spans,
        "novelty_short": curves["short"].astype(float).tolist(),
        "novelty_medium": curves["medium"].astype(float).tolist(),
        "novelty_long": curves["long"].astype(float).tolist(),
        "novelty_history_valid": history_mask.any(axis=0).tolist(),
        **{k: v.astype(float).tolist() for k, v in cqt.items()},
        **{k: v.astype(float).tolist() for k, v in phrase.items()},
        **{k: v.astype(float).tolist() for k, v in onset.items()},
    }


_STEM_COMPARISON_LAGS = (8, 16, 32)
_CQT_LAGS = _STEM_COMPARISON_LAGS


def _first_stable_active_beat(rms_beat: np.ndarray, *, hold: int = 4) -> int:
    """Return the first beat where a stem is stably audible."""
    energy = _normalize_01(np.asarray(rms_beat, dtype=np.float64))
    if energy.size == 0 or float(energy.max()) < 1e-7:
        return 0

    threshold = 0.15
    hold = max(1, int(hold))
    for i in range(0, energy.size):
        window = energy[i : min(i + hold, energy.size)]
        if energy[i] >= threshold and window.size and float(np.mean(window >= threshold)) >= 0.5:
            return int(i)
    return 0


def _snap_anchor_to_bar_beat(
    anchor_beat: int,
    *,
    beats_per_bar: int = 4,
    bar_phase: int = 0,
    max_shift: int = 2,
) -> int:
    """Snap a stem entrance to a nearby assumed bar boundary."""
    if anchor_beat <= 0 or beats_per_bar <= 1 or max_shift < 0:
        return max(0, int(anchor_beat))

    anchor = int(anchor_beat)
    phase = int(bar_phase) % int(beats_per_bar)
    lower = phase + ((anchor - phase) // beats_per_bar) * beats_per_bar
    upper = lower + beats_per_bar
    candidates = [lower, upper]
    snapped = min(candidates, key=lambda c: (abs(c - anchor), c))
    if abs(snapped - anchor) <= max_shift:
        return int(snapped)
    return anchor


def _stem_block_anchor(rms_beat: np.ndarray, *, bar_phase: int = 0) -> int:
    """Choose the beat-index anchor used for all block sizes of one stem."""
    first_active = _first_stable_active_beat(rms_beat)
    return _snap_anchor_to_bar_beat(first_active, bar_phase=bar_phase)


def _stem_block_offset(rms_beat: np.ndarray, lag: int, *, bar_phase: int = 0) -> int:
    """Choose a per-lag comparison offset from the stem's beat-index anchor."""
    if lag <= 1:
        return 0
    return int(_stem_block_anchor(rms_beat, bar_phase=bar_phase) % lag)


def _stem_block_offsets(
    rms_beat: np.ndarray,
    lags: tuple[int, ...],
    *,
    bar_phase: int = 0,
) -> dict[int, int]:
    """Choose per-lag offsets from one stem-specific beat-index anchor."""
    if not lags:
        return {}
    anchor = _stem_block_anchor(rms_beat, bar_phase=bar_phase)
    return {
        int(lag): int(anchor % int(lag)) if int(lag) > 1 else 0
        for lag in lags
    }


def _block_offset_for_lag(block_offsets: dict[int, int] | None, lag: int) -> int:
    if not block_offsets:
        return 0
    return int(np.clip(block_offsets.get(lag, 0), 0, max(0, lag - 1)))


def _full_block_pairs(n_beats: int, offset: int, lag: int) -> list[tuple[int, int, int]]:
    """Return (previous_start, current_start, length) for normal adjacent blocks."""
    pairs: list[tuple[int, int, int]] = []
    n_chunks = max(0, (n_beats - offset) // lag)
    for i in range(1, n_chunks):
        cur_s = offset + i * lag
        prv_s = offset + (i - 1) * lag
        length = min(n_beats - cur_s, n_beats - prv_s, lag)
        if length > 0:
            pairs.append((prv_s, cur_s, length))
    return pairs


def _matched_half_block_pairs(n_beats: int, offset: int, lag: int) -> list[tuple[int, int, int]]:
    """Return half-size slices matched to the same position in the previous block."""
    sub_lag = max(1, lag // 2)
    pairs: list[tuple[int, int, int]] = []
    n_chunks = max(0, (n_beats - offset) // lag)
    for i in range(1, n_chunks):
        prev_parent = offset + (i - 1) * lag
        cur_parent = offset + i * lag
        for pos in range(0, lag, sub_lag):
            cur_s = cur_parent + pos
            prv_s = prev_parent + pos
            length = min(n_beats - cur_s, n_beats - prv_s, sub_lag)
            if length > 0:
                pairs.append((prv_s, cur_s, length))
    return pairs


def _cqt_similarity_curves(
    Q_beat: np.ndarray,
    beat_times: np.ndarray,
    frame_times: np.ndarray,
    lags: tuple[int, ...] = _CQT_LAGS,
    block_offsets: dict[int, int] | None = None,
) -> dict[str, np.ndarray]:
    """Block comparison using log-CQT cosine similarity (84 bins, no octave collapse).

    Unlike chroma, this preserves register: G2 and G3 map to different bins, so
    a bass-line change is visible even when pitch classes overlap. Unlike MFCC,
    it is not cepstrally smoothed, so timbral averaging doesn't mask pitch changes.
    """
    n_beats = min(Q_beat.shape[1], len(beat_times))
    Q = np.log1p(Q_beat[:, :n_beats])   # log-compress; all values ≥ 0

    out: dict[str, np.ndarray] = {}
    bt = beat_times[:n_beats]

    for lag in lags:
        sim_beat = np.ones(n_beats, dtype=np.float32)
        nov_beat = np.zeros(n_beats, dtype=np.float32)
        sim_half_beat = np.ones(n_beats, dtype=np.float32)
        nov_half_beat = np.zeros(n_beats, dtype=np.float32)
        offset = _block_offset_for_lag(block_offsets, lag)

        def score_pair(prv_s: int, cur_s: int, length: int) -> float:
            Q_c = Q[:, cur_s:cur_s + length]
            Q_p = Q[:, prv_s:prv_s + length]

            nc = np.linalg.norm(Q_c, axis=0) + 1e-8
            np_ = np.linalg.norm(Q_p, axis=0) + 1e-8
            s = np.clip(np.sum(Q_c * Q_p, axis=0) / (nc * np_), 0.0, 1.0)
            return float(s.mean())

        for prv_s, cur_s, L in _full_block_pairs(n_beats, offset, lag):
            mean_sim = score_pair(prv_s, cur_s, L)
            sim_beat[cur_s:cur_s + L] = mean_sim
            nov_beat[cur_s:cur_s + L] = 1.0 - mean_sim

        for prv_s, cur_s, L in _matched_half_block_pairs(n_beats, offset, lag):
            mean_sim = score_pair(prv_s, cur_s, L)
            sim_half_beat[cur_s:cur_s + L] = mean_sim
            nov_half_beat[cur_s:cur_s + L] = 1.0 - mean_sim

        out[f"cqt_sim_{lag}"] = np.interp(frame_times, bt, sim_beat).astype(np.float32)
        out[f"cqt_nov_{lag}"] = np.interp(frame_times, bt, nov_beat).astype(np.float32)
        out[f"cqt_half_sim_{lag}"] = np.interp(frame_times, bt, sim_half_beat).astype(np.float32)
        out[f"cqt_half_nov_{lag}"] = np.interp(frame_times, bt, nov_half_beat).astype(np.float32)

    return out


_PHRASE_LAGS = _STEM_COMPARISON_LAGS
_PHRASE_BINS_PER_BEAT = 2
_PHRASE_WEIGHTS = {
    "spectrum": 0.55,
    "attack": 0.25,
    "energy": 0.15,
    "peaks": 0.05,
}
_PHRASE_ENV_WEIGHTS = {
    "energy": 0.45,
    "attack": 0.25,
    "peaks": 0.15,
    "valleys": 0.15,
}
_PHRASE_AUDIBLE_FLOOR_DB = -36.0
_PHRASE_COVERAGE_THRESHOLD = 0.05
_PHRASE_SPEC_AUD_ALPHA     = 0.20  # weight given to audible-but-not-onset beats in phrase_spec_sim
_PHRASE_COVERAGE_POWER = 2.0
_PHRASE_ACCENT_SMOOTH_S = 0.10
_PHRASE_ACCENT_MIN_DIST_S = 0.10
_PHRASE_ACCENT_PROMINENCE = 0.10
_PHRASE_ACCENT_TOL_S = 0.45
_PHRASE_ACCENT_TIMING_SIGMA_S = 0.22
_PHRASE_ACCENT_CURVE_WEIGHT = 0.75
_PHRASE_ACCENT_CURVE_POINTS = 128
_PHRASE_AMP_SMOOTH_S = 0.08
_PHRASE_AMP_CURVE_POINTS = 128
_PHRASE_AMP_MIN_DIST_S = 0.15   # calibrated: ~20 bass note attacks per 11s (Do I Wanna Know 50-61s)
_PHRASE_AMP_WEIGHTS = {
    "shape": 0.45,
    "texture": 0.40,
    "pulse": 0.15,
}


def _cosine_01(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).ravel()
    bb = np.asarray(b, dtype=np.float64).ravel()
    n = min(aa.size, bb.size)
    if n == 0:
        return 1.0
    aa = aa[:n]
    bb = bb[:n]
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom < 1e-12:
        return 1.0 if float(np.linalg.norm(aa - bb)) < 1e-12 else 0.0
    return float(np.clip(np.dot(aa, bb) / denom, 0.0, 1.0))


def _weighted_cosine_01(a: np.ndarray, b: np.ndarray, weights: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    ww = np.asarray(weights, dtype=np.float64).ravel()
    n = min(aa.shape[-1], bb.shape[-1], ww.size)
    if n == 0:
        return 1.0
    ww = np.clip(ww[:n], 0.0, 1.0)
    if float(ww.sum()) < 1e-9:
        return 1.0
    aa = aa[..., :n] * np.sqrt(ww)
    bb = bb[..., :n] * np.sqrt(ww)
    return _cosine_01(aa, bb)


def _shape_similarity(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).ravel()
    bb = np.asarray(b, dtype=np.float64).ravel()
    n = min(aa.size, bb.size)
    if n == 0:
        return 1.0
    aa = aa[:n]
    bb = bb[:n]
    l1 = 1.0 - (float(np.abs(aa - bb).sum()) / (float(aa.sum() + bb.sum()) + 1e-8))
    return float(np.clip(0.7 * _cosine_01(aa, bb) + 0.3 * l1, 0.0, 1.0))


def _weighted_shape_similarity(a: np.ndarray, b: np.ndarray, weights: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).ravel()
    bb = np.asarray(b, dtype=np.float64).ravel()
    ww = np.asarray(weights, dtype=np.float64).ravel()
    n = min(aa.size, bb.size, ww.size)
    if n == 0:
        return 1.0
    aa = aa[:n]
    bb = bb[:n]
    ww = np.clip(ww[:n], 0.0, 1.0)
    if float(ww.sum()) < 1e-9:
        return 1.0

    dot = float(np.sum(ww * aa * bb))
    denom = float(np.sqrt(np.sum(ww * aa * aa)) * np.sqrt(np.sum(ww * bb * bb)))
    cosine = 1.0 if denom < 1e-12 else dot / denom
    l1 = 1.0 - (
        float(np.sum(ww * np.abs(aa - bb)))
        / (float(np.sum(ww * (aa + bb))) + 1e-8)
    )
    return float(np.clip(0.7 * cosine + 0.3 * l1, 0.0, 1.0))


def _binned_block_shape(values: np.ndarray, start: int, end: int, bins: int, *, reduce: str) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    start = int(np.clip(start, 0, vals.size))
    end = int(np.clip(end, start, vals.size))
    if end <= start or bins <= 0:
        return np.zeros(max(0, bins), dtype=np.float64)

    seg = vals[start:end]
    edges = np.linspace(0, len(seg), bins + 1, dtype=int)
    out = np.zeros(bins, dtype=np.float64)
    for i in range(bins):
        lo, hi = int(edges[i]), int(edges[i + 1])
        if hi <= lo:
            continue
        part = seg[lo:hi]
        out[i] = float(part.max() if reduce == "max" else part.mean())
    return out


def _binned_onset_strength(attack: np.ndarray, start: int, end: int, bins: int) -> np.ndarray:
    """Per-beat MAX of the frame-resolution attack envelope (half-wave-rectified d/dt log-RMS)."""
    vals = np.asarray(attack, dtype=np.float64)
    start = int(np.clip(start, 0, vals.size))
    end = int(np.clip(end, start, vals.size))
    if end <= start or bins <= 0:
        return np.zeros(max(0, bins), dtype=np.float64)
    seg = vals[start:end]
    edges = np.linspace(0, len(seg), bins + 1, dtype=int)
    out = np.zeros(bins, dtype=np.float64)
    for i in range(bins):
        lo, hi = int(edges[i]), int(edges[i + 1])
        if hi > lo:
            out[i] = float(seg[lo:hi].max())
    return out


def _binned_audibility(rms: np.ndarray, start: int, end: int, bins: int, floor: float) -> np.ndarray:
    vals = np.asarray(rms, dtype=np.float64)
    start = int(np.clip(start, 0, vals.size))
    end = int(np.clip(end, start, vals.size))
    if end <= start or bins <= 0:
        return np.zeros(max(0, bins), dtype=np.float64)

    seg = vals[start:end]
    scale = max(float(vals.max()) - floor, 1e-12)
    aud = np.clip((seg - floor) / scale, 0.0, 1.0)
    edges = np.linspace(0, len(aud), bins + 1, dtype=int)
    out = np.zeros(bins, dtype=np.float64)
    for i in range(bins):
        lo, hi = int(edges[i]), int(edges[i + 1])
        if hi > lo:
            out[i] = float(aud[lo:hi].mean())
    return out


def _macro_peak_count_similarity(a: np.ndarray, b: np.ndarray, weights: np.ndarray | None = None) -> float:
    from scipy.signal import find_peaks as _find_peaks

    def count(x: np.ndarray, w: np.ndarray | None) -> int:
        xx = _smooth_1d(np.asarray(x, dtype=np.float32), win=3)
        if w is not None:
            ww = np.asarray(w, dtype=np.float32)
            xx = xx[: ww.size] * ww[: xx.size]
        if xx.size == 0 or float(xx.max()) < 1e-9:
            return 0
        peaks, _ = _find_peaks(xx, height=0.2 * float(xx.max()), distance=2)
        if xx[0] >= 0.2 * float(xx.max()) and (xx.size == 1 or xx[0] >= xx[1]):
            peaks = np.unique(np.concatenate([np.array([0], dtype=int), peaks]))
        return int(peaks.size)

    ww = None if weights is None else np.asarray(weights, dtype=np.float64)
    ca = count(a, ww)
    cb = count(b, ww)
    mx = max(ca, cb)
    if mx == 0:
        return 1.0
    return 1.0 - (abs(ca - cb) / mx)


def _macro_valley_count_similarity(a: np.ndarray, b: np.ndarray, weights: np.ndarray | None = None) -> float:
    from scipy.signal import find_peaks as _find_peaks

    def count(x: np.ndarray, w: np.ndarray | None) -> int:
        xx = _smooth_1d(np.asarray(x, dtype=np.float32), win=3)
        if w is not None:
            ww = np.asarray(w, dtype=np.float32)
            xx = xx[: ww.size] * ww[: xx.size]
        if xx.size == 0 or float(xx.max() - xx.min()) < 1e-9:
            return 0
        inv = float(xx.max()) - xx
        if float(inv.max()) < 1e-9:
            return 0
        valleys, _ = _find_peaks(inv, height=0.2 * float(inv.max()), distance=2)
        if inv[0] >= 0.2 * float(inv.max()) and (inv.size == 1 or inv[0] >= inv[1]):
            valleys = np.unique(np.concatenate([np.array([0], dtype=int), valleys]))
        return int(valleys.size)

    ww = None if weights is None else np.asarray(weights, dtype=np.float64)
    ca = count(a, ww)
    cb = count(b, ww)
    mx = max(ca, cb)
    if mx == 0:
        return 1.0
    return 1.0 - (abs(ca - cb) / mx)


def _accent_peak_events(
    rms: np.ndarray,
    start: int,
    end: int,
    frame_s: float,
    *,
    smooth_s: float = _PHRASE_ACCENT_SMOOTH_S,
    min_dist_s: float = _PHRASE_ACCENT_MIN_DIST_S,
    prominence: float = _PHRASE_ACCENT_PROMINENCE,
) -> list[tuple[float, float]]:
    from scipy.signal import find_peaks as _find_peaks

    vals = np.asarray(rms, dtype=np.float64)
    start = int(np.clip(start, 0, vals.size))
    end = int(np.clip(end, start, vals.size))
    if end <= start or frame_s <= 0:
        return []

    seg = vals[start:end]
    span = float(seg.max() - seg.min()) if seg.size else 0.0
    if span < 1e-10:
        return []

    x = (seg - float(seg.min())) / (span + 1e-12)
    smooth_win = max(1, int(round(smooth_s / frame_s)))
    x = _smooth_1d(x.astype(np.float32), win=smooth_win).astype(np.float64)
    if float(x.max() - x.min()) < 1e-8:
        return []

    min_dist = max(1, int(round(min_dist_s / frame_s)))
    peaks, props = _find_peaks(x, distance=min_dist, prominence=prominence)
    if peaks.size == 0:
        return []

    prominences = np.asarray(props.get("prominences", np.zeros_like(peaks, dtype=float)), dtype=np.float64)
    duration_s = max((end - start - 1) * frame_s, frame_s)
    events: list[tuple[float, float]] = []
    for peak, prom in zip(peaks, prominences):
        rel_s = float(peak) * frame_s
        strength = float(np.clip(0.5 * x[int(peak)] + 0.5 * prom, 0.0, 1.0))
        events.append((rel_s / duration_s, strength))
    return events


def _accent_envelope_curve(
    rms: np.ndarray,
    start: int,
    end: int,
    frame_s: float,
    *,
    smooth_s: float = _PHRASE_ACCENT_SMOOTH_S,
    points: int = _PHRASE_ACCENT_CURVE_POINTS,
) -> np.ndarray:
    vals = np.asarray(rms, dtype=np.float64)
    start = int(np.clip(start, 0, vals.size))
    end = int(np.clip(end, start, vals.size))
    points = max(8, int(points))
    if end <= start or frame_s <= 0:
        return np.zeros(points, dtype=np.float64)

    seg = vals[start:end]
    span = float(seg.max() - seg.min()) if seg.size else 0.0
    if span < 1e-10:
        return np.zeros(points, dtype=np.float64)

    x = (seg - float(seg.min())) / (span + 1e-12)
    smooth_win = max(1, int(round(smooth_s / frame_s)))
    x = _smooth_1d(x.astype(np.float32), win=smooth_win).astype(np.float64)
    x = np.maximum(x - float(np.quantile(x, 0.20)), 0.0)
    if float(x.max()) > 1e-12:
        x = x / float(x.max())

    xp = np.linspace(0.0, 1.0, x.size)
    yp = np.linspace(0.0, 1.0, points)
    return np.interp(yp, xp, x).astype(np.float64)


def _accent_curve_similarity(
    prev_curve: np.ndarray,
    cur_curve: np.ndarray,
    block_s: float,
    *,
    max_shift_s: float = _PHRASE_ACCENT_TOL_S,
) -> float:
    aa = np.asarray(prev_curve, dtype=np.float64).ravel()
    bb = np.asarray(cur_curve, dtype=np.float64).ravel()
    n = min(aa.size, bb.size)
    if n == 0:
        return 1.0
    aa = aa[:n]
    bb = bb[:n]
    if float(aa.max()) < 1e-9 and float(bb.max()) < 1e-9:
        return 1.0
    if block_s <= 0:
        return _shape_similarity(aa, bb)

    max_shift = int(round((max_shift_s / block_s) * n))
    max_shift = int(np.clip(max_shift, 0, max(0, n // 4)))
    best = _shape_similarity(aa, bb)
    for shift in range(1, max_shift + 1):
        best = max(best, _shape_similarity(aa[shift:], bb[:-shift]))
        best = max(best, _shape_similarity(aa[:-shift], bb[shift:]))
    return float(np.clip(best, 0.0, 1.0))


def _accent_event_similarity(
    prev_events: list[tuple[float, float]],
    cur_events: list[tuple[float, float]],
    *,
    block_s: float,
    tol_s: float = _PHRASE_ACCENT_TOL_S,
    timing_sigma_s: float = _PHRASE_ACCENT_TIMING_SIGMA_S,
) -> float:
    from scipy.optimize import linear_sum_assignment as _linear_sum_assignment

    if not prev_events and not cur_events:
        return 1.0
    if not prev_events or not cur_events or block_s <= 0:
        return 0.0

    n_total = max(len(prev_events), len(cur_events))
    scores = np.zeros((len(cur_events), len(prev_events)), dtype=np.float64)
    tol_rel = tol_s / block_s
    sigma_rel = max(timing_sigma_s / block_s, 1e-6)
    for i, (rel_c, str_c) in enumerate(cur_events):
        for j, (rel_p, str_p) in enumerate(prev_events):
            d = abs(rel_c - rel_p)
            if d > tol_rel:
                continue
            timing = float(np.exp(-((d / sigma_rel) ** 2)))
            strength = 1.0 - (abs(str_c - str_p) / (max(str_c, str_p, 1e-6)))
            scores[i, j] = timing * float(np.clip(0.8 + 0.2 * strength, 0.0, 1.0))

    if not np.any(scores > 0.0):
        return 0.0

    row_ind, col_ind = _linear_sum_assignment(-scores)
    return float(scores[row_ind, col_ind].sum()) / n_total


def _relative_vector_similarity(
    a: np.ndarray,
    b: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    aa = np.asarray(a, dtype=np.float64).ravel()
    bb = np.asarray(b, dtype=np.float64).ravel()
    n = min(aa.size, bb.size)
    if n == 0:
        return 1.0
    aa = aa[:n]
    bb = bb[:n]
    if weights is None:
        ww = np.ones(n, dtype=np.float64)
    else:
        ww = np.asarray(weights, dtype=np.float64).ravel()[:n]
    ww = np.clip(ww, 0.0, None)
    if float(ww.sum()) < 1e-12:
        return 1.0

    scale = np.maximum(np.maximum(np.abs(aa), np.abs(bb)), 0.10)
    sims = 1.0 - (np.abs(aa - bb) / scale)
    return float(np.clip(np.sum(ww * np.clip(sims, 0.0, 1.0)) / np.sum(ww), 0.0, 1.0))


def _amp_envelope_curve(
    rms: np.ndarray,
    start: int,
    end: int,
    frame_s: float,
    *,
    smooth_s: float = _PHRASE_AMP_SMOOTH_S,
    points: int = _PHRASE_AMP_CURVE_POINTS,
) -> np.ndarray:
    vals = np.asarray(rms, dtype=np.float64)
    start = int(np.clip(start, 0, vals.size))
    end = int(np.clip(end, start, vals.size))
    points = max(8, int(points))
    if end <= start or frame_s <= 0:
        return np.zeros(points, dtype=np.float64)

    seg = vals[start:end]
    smooth_win = max(1, int(round(smooth_s / frame_s)))
    x = _smooth_1d(seg.astype(np.float32), win=smooth_win).astype(np.float64)
    lo = float(np.quantile(x, 0.05))
    hi = float(np.quantile(x, 0.95))
    if hi - lo < 1e-10:
        return np.zeros(points, dtype=np.float64)

    x = np.clip((x - lo) / (hi - lo + 1e-12), 0.0, 1.0)
    xp = np.linspace(0.0, 1.0, x.size)
    yp = np.linspace(0.0, 1.0, points)
    return np.interp(yp, xp, x).astype(np.float64)


def _amp_texture_features(
    rms: np.ndarray,
    start: int,
    end: int,
    frame_s: float,
    audible_floor: float,
    *,
    smooth_s: float = _PHRASE_AMP_SMOOTH_S,
) -> np.ndarray:
    from scipy.signal import find_peaks as _find_peaks

    vals = np.asarray(rms, dtype=np.float64)
    start = int(np.clip(start, 0, vals.size))
    end = int(np.clip(end, start, vals.size))
    if end <= start or frame_s <= 0:
        return np.zeros(10, dtype=np.float64)

    seg = vals[start:end]
    smooth_win = max(1, int(round(smooth_s / frame_s)))
    smoothed = _smooth_1d(seg.astype(np.float32), win=smooth_win).astype(np.float64)
    duration_s = max((end - start) * frame_s, frame_s)
    mean = float(np.mean(smoothed))
    std = float(np.std(smoothed))
    p10 = float(np.quantile(smoothed, 0.10))
    p90 = float(np.quantile(smoothed, 0.90))
    p95 = float(np.quantile(smoothed, 0.95))
    span = max(p95 - p10, 0.0)

    lo = float(np.quantile(smoothed, 0.05))
    hi = float(np.quantile(smoothed, 0.95))
    if hi - lo > 1e-10:
        x = np.clip((smoothed - lo) / (hi - lo + 1e-12), 0.0, 1.0)
    else:
        x = np.zeros_like(smoothed)

    min_dist = max(1, int(round(_PHRASE_AMP_MIN_DIST_S / frame_s)))
    peaks, peak_props = _find_peaks(x, distance=min_dist, prominence=0.10)
    valleys, valley_props = _find_peaks(1.0 - x, distance=min_dist, prominence=0.10)
    peak_prom = np.asarray(peak_props.get("prominences", np.zeros(0)), dtype=np.float64)
    valley_prom = np.asarray(valley_props.get("prominences", np.zeros(0)), dtype=np.float64)
    slope = float(np.mean(np.abs(np.diff(x)))) if x.size > 1 else 0.0

    return np.array(
        [
            np.clip((std / (mean + 1e-9)) / 2.5, 0.0, 1.0),
            np.clip(span / (p90 + 1e-9), 0.0, 1.0),
            np.clip(((p95 / (mean + 1e-9)) - 1.0) / 4.0, 0.0, 1.0),
            np.clip(float(np.mean(smoothed > audible_floor)), 0.0, 1.0),
            np.clip(float(np.mean(x > 0.15)), 0.0, 1.0),
            np.clip(float(np.mean(x < 0.15)), 0.0, 1.0),
            np.clip((len(peaks) / duration_s) / 4.0, 0.0, 1.0),
            np.clip((len(valleys) / duration_s) / 4.0, 0.0, 1.0),
            np.clip(float(np.mean(peak_prom)) if peak_prom.size else 0.0, 0.0, 1.0),
            np.clip(4.0 * slope, 0.0, 1.0),
        ],
        dtype=np.float64,
    )


def _amp_pulse_profile(curve: np.ndarray, points: int = 32) -> np.ndarray:
    x = np.asarray(curve, dtype=np.float64).ravel()
    points = max(4, int(points))
    if x.size < 4:
        return np.zeros(points, dtype=np.float64)
    x = x - float(np.mean(x))
    denom = float(np.dot(x, x))
    if denom < 1e-12:
        return np.zeros(points, dtype=np.float64)

    max_lag = max(2, x.size // 2)
    corr = np.array([np.dot(x[:-lag], x[lag:]) / denom for lag in range(1, max_lag)], dtype=np.float64)
    corr = np.maximum(corr, 0.0)
    if corr.size == 0:
        return np.zeros(points, dtype=np.float64)
    xp = np.linspace(0.0, 1.0, corr.size)
    yp = np.linspace(0.0, 1.0, points)
    return np.interp(yp, xp, corr).astype(np.float64)


def _phrase_similarity_curves(
    Q_beat: np.ndarray,
    rms: np.ndarray,
    beat_frames: np.ndarray,
    beat_times: np.ndarray,
    frame_times: np.ndarray,
    lags: tuple[int, ...] = _PHRASE_LAGS,
    bins_per_beat: int = _PHRASE_BINS_PER_BEAT,
    block_offsets: dict[int, int] | None = None,
) -> dict[str, np.ndarray]:
    """Whole-block phrase similarity.

    It exports the plain full-block blend, spectral/envelope components,
    accent/amp metrics, and an audibility-aware variant.
    """
    n_beats = min(Q_beat.shape[1], len(beat_times), len(beat_frames))
    Q = np.log1p(Q_beat[:, :n_beats])
    rms_use = np.asarray(rms, dtype=np.float64)
    n_frames = rms_use.size
    bf = np.clip(beat_frames[:n_beats], 0, max(0, n_frames - 1)).astype(int)
    frame_s = float(np.median(np.diff(frame_times))) if len(frame_times) > 1 else 0.023
    audible_floor = float(rms_use.max()) * (10.0 ** (_PHRASE_AUDIBLE_FLOOR_DB / 20.0))
    log_rms = np.log1p(rms_use)
    attack = np.maximum(np.diff(log_rms, prepend=0.0), 0.0)
    attack = _smooth_1d(_normalize_01(attack).astype(np.float32), win=3).astype(np.float64)
    energy = _smooth_1d(_normalize_01(rms_use).astype(np.float32), win=3).astype(np.float64)

    def frame_range(b_start: int, b_end: int) -> tuple[int, int]:
        f0 = int(bf[min(b_start, n_beats - 1)])
        f1 = int(bf[b_end]) if b_end < n_beats else n_frames
        return max(0, min(f0, n_frames)), max(0, min(f1, n_frames))

    out: dict[str, np.ndarray] = {}
    bt = beat_times[:n_beats]

    for lag in lags:
        sim = {
            "phrase": np.ones(n_beats, dtype=np.float32),
            "spec": np.ones(n_beats, dtype=np.float32),
            "spec_aud": np.ones(n_beats, dtype=np.float32),
            "env": np.ones(n_beats, dtype=np.float32),
            "accent": np.ones(n_beats, dtype=np.float32),
            "amp": np.ones(n_beats, dtype=np.float32),
            "aud": np.ones(n_beats, dtype=np.float32),
        }
        nov = {k: np.zeros(n_beats, dtype=np.float32) for k in sim}
        half_sim = {k: np.ones(n_beats, dtype=np.float32) for k in sim}
        half_nov = {k: np.zeros(n_beats, dtype=np.float32) for k in sim}
        offset = _block_offset_for_lag(block_offsets, lag)

        def score_pair(prv_s: int, cur_s: int, L: int) -> dict[str, float]:
            if L < 1:
                return {
                    "phrase": 1.0,
                    "spec": 1.0,
                    "spec_aud": 1.0,
                    "env": 1.0,
                    "accent": 1.0,
                    "amp": 1.0,
                    "aud": 1.0,
                }

            cur_f0, cur_f1 = frame_range(cur_s, cur_s + L)
            prv_f0, prv_f1 = frame_range(prv_s, prv_s + L)
            bins = max(4, int(L * bins_per_beat))
            cur_aud_beat = _binned_audibility(rms_use, cur_f0, cur_f1, L, audible_floor)
            prv_aud_beat = _binned_audibility(rms_use, prv_f0, prv_f1, L, audible_floor)
            beat_audibility = np.maximum(cur_aud_beat, prv_aud_beat)
            cur_onset_beat = _binned_onset_strength(attack, cur_f0, cur_f1, L)
            prv_onset_beat = _binned_onset_strength(attack, prv_f0, prv_f1, L)

            Q_cur = Q[:, cur_s:cur_s + L]
            Q_prv = Q[:, prv_s:prv_s + L]
            spectral_plain = _cosine_01(Q_cur, Q_prv)
            spectral_audible = _weighted_cosine_01(Q_cur, Q_prv, beat_audibility)
            # Onset-biased spectral comparison: weight by the MAX of both blocks'
            # per-beat onset strength so neither block's active moments are missed.
            # Gate on current-block onset: if the current window has no detectable
            # note attacks (only noise floor or reverb tail), there is nothing
            # meaningful to compare → treat as equivalent silence (1.0).
            # Using cur_onset gated by audibility prevents noise-floor attack
            # derivatives from spuriously triggering the comparison.
            cur_aud_above = np.maximum(0.0, cur_aud_beat - _PHRASE_COVERAGE_THRESHOLD)
            cur_note_beat = cur_onset_beat * (cur_aud_above > 0.0).astype(np.float64)
            beat_onset_weight = np.maximum(cur_note_beat, prv_onset_beat)
            if float(cur_note_beat.max()) < _PHRASE_COVERAGE_THRESHOLD:
                spectral_onset = 1.0
            else:
                spectral_onset = _weighted_cosine_01(Q_cur, Q_prv, beat_onset_weight)

            cur_attack = _binned_block_shape(attack, cur_f0, cur_f1, bins, reduce="mean")
            prv_attack = _binned_block_shape(attack, prv_f0, prv_f1, bins, reduce="mean")
            cur_energy = _binned_block_shape(energy, cur_f0, cur_f1, bins, reduce="mean")
            prv_energy = _binned_block_shape(energy, prv_f0, prv_f1, bins, reduce="mean")
            cur_aud = _binned_audibility(rms_use, cur_f0, cur_f1, bins, audible_floor)
            prv_aud = _binned_audibility(rms_use, prv_f0, prv_f1, bins, audible_floor)
            audibility = np.maximum(cur_aud, prv_aud)
            silence_agreement = float(np.clip(1.0 - np.mean(np.abs(cur_aud - prv_aud)), 0.0, 1.0))
            coverage = float(np.mean(audibility > _PHRASE_COVERAGE_THRESHOLD))
            active_weight = float(np.clip(coverage ** _PHRASE_COVERAGE_POWER, 0.0, 1.0))

            attack_plain = _shape_similarity(cur_attack, prv_attack)
            energy_plain = _shape_similarity(cur_energy, prv_energy)
            peak_plain = _macro_peak_count_similarity(cur_attack, prv_attack)
            energy_peak_plain = _macro_peak_count_similarity(cur_energy, prv_energy)
            energy_valley_plain = _macro_valley_count_similarity(cur_energy, prv_energy)
            mean_sim = (
                _PHRASE_WEIGHTS["spectrum"] * spectral_plain
                + _PHRASE_WEIGHTS["attack"] * attack_plain
                + _PHRASE_WEIGHTS["energy"] * energy_plain
                + _PHRASE_WEIGHTS["peaks"] * peak_plain
            )
            mean_sim = float(np.clip(mean_sim, 0.0, 1.0))
            spec_sim = float(np.clip(spectral_onset, 0.0, 1.0))
            # Old audibility-weighted spec (kept for comparison in the dashboard).
            if float(beat_audibility.max()) < _PHRASE_COVERAGE_THRESHOLD:
                spec_aud_sim = 1.0
            else:
                spec_aud_sim = float(np.clip(spectral_audible, 0.0, 1.0))
            env_sim = (
                _PHRASE_ENV_WEIGHTS["energy"] * energy_plain
                + _PHRASE_ENV_WEIGHTS["attack"] * attack_plain
                + _PHRASE_ENV_WEIGHTS["peaks"] * energy_peak_plain
                + _PHRASE_ENV_WEIGHTS["valleys"] * energy_valley_plain
            )
            env_sim = float(np.clip(env_sim, 0.0, 1.0))
            cur_events = _accent_peak_events(rms_use, cur_f0, cur_f1, frame_s)
            prv_events = _accent_peak_events(rms_use, prv_f0, prv_f1, frame_s)
            block_s = max((min(cur_f1 - cur_f0, prv_f1 - prv_f0) - 1) * frame_s, frame_s)
            curve_sim = _accent_curve_similarity(
                _accent_envelope_curve(rms_use, prv_f0, prv_f1, frame_s),
                _accent_envelope_curve(rms_use, cur_f0, cur_f1, frame_s),
                block_s=block_s,
            )
            event_sim = _accent_event_similarity(prv_events, cur_events, block_s=block_s)
            curve_weight = float(np.clip(_PHRASE_ACCENT_CURVE_WEIGHT, 0.0, 1.0))
            accent_sim = curve_weight * curve_sim + (1.0 - curve_weight) * event_sim
            accent_sim = float(np.clip(accent_sim, 0.0, 1.0))
            prv_amp_curve = _amp_envelope_curve(rms_use, prv_f0, prv_f1, frame_s)
            cur_amp_curve = _amp_envelope_curve(rms_use, cur_f0, cur_f1, frame_s)
            amp_shape_sim = _accent_curve_similarity(prv_amp_curve, cur_amp_curve, block_s=block_s)
            amp_texture_sim = _relative_vector_similarity(
                _amp_texture_features(rms_use, prv_f0, prv_f1, frame_s, audible_floor),
                _amp_texture_features(rms_use, cur_f0, cur_f1, frame_s, audible_floor),
                weights=np.array([1.0, 1.2, 1.0, 0.8, 0.8, 0.8, 1.4, 1.0, 0.8, 1.2], dtype=np.float64),
            )
            amp_pulse_sim = _shape_similarity(_amp_pulse_profile(prv_amp_curve), _amp_pulse_profile(cur_amp_curve))
            amp_sim = (
                _PHRASE_AMP_WEIGHTS["shape"] * amp_shape_sim
                + _PHRASE_AMP_WEIGHTS["texture"] * amp_texture_sim
                + _PHRASE_AMP_WEIGHTS["pulse"] * amp_pulse_sim
            )
            amp_sim = float(np.clip(amp_sim, 0.0, 1.0))

            attack_audible = _weighted_shape_similarity(cur_attack, prv_attack, audibility)
            energy_audible = _weighted_shape_similarity(cur_energy, prv_energy, audibility)
            peak_audible = _macro_peak_count_similarity(cur_attack, prv_attack, audibility)
            active_sim = (
                _PHRASE_WEIGHTS["spectrum"] * spectral_audible
                + _PHRASE_WEIGHTS["attack"] * attack_audible
                + _PHRASE_WEIGHTS["energy"] * energy_audible
                + _PHRASE_WEIGHTS["peaks"] * peak_audible
            )
            audible_sim = active_weight * active_sim + (1.0 - active_weight) * silence_agreement
            audible_sim = float(np.clip(audible_sim, 0.0, 1.0))

            return {
                "phrase": mean_sim,
                "spec": spec_sim,
                "spec_aud": spec_aud_sim,
                "env": env_sim,
                "accent": accent_sim,
                "amp": amp_sim,
                "aud": audible_sim,
            }

        def write_scores(
            sim_arrays: dict[str, np.ndarray],
            nov_arrays: dict[str, np.ndarray],
            cur_s: int,
            L: int,
            scores: dict[str, float],
        ) -> None:
            for key, value in scores.items():
                sim_arrays[key][cur_s:cur_s + L] = value
                nov_arrays[key][cur_s:cur_s + L] = 1.0 - value

        for prv_s, cur_s, L in _full_block_pairs(n_beats, offset, lag):
            write_scores(sim, nov, cur_s, L, score_pair(prv_s, cur_s, L))

        for prv_s, cur_s, L in _matched_half_block_pairs(n_beats, offset, lag):
            write_scores(half_sim, half_nov, cur_s, L, score_pair(prv_s, cur_s, L))

        out[f"phrase_sim_{lag}"] = np.interp(frame_times, bt, sim["phrase"]).astype(np.float32)
        out[f"phrase_nov_{lag}"] = np.interp(frame_times, bt, nov["phrase"]).astype(np.float32)
        out[f"phrase_spec_sim_{lag}"] = np.interp(frame_times, bt, sim["spec"]).astype(np.float32)
        out[f"phrase_spec_nov_{lag}"] = np.interp(frame_times, bt, nov["spec"]).astype(np.float32)
        out[f"phrase_env_sim_{lag}"] = np.interp(frame_times, bt, sim["env"]).astype(np.float32)
        out[f"phrase_env_nov_{lag}"] = np.interp(frame_times, bt, nov["env"]).astype(np.float32)
        out[f"phrase_accent_sim_{lag}"] = np.interp(frame_times, bt, sim["accent"]).astype(np.float32)
        out[f"phrase_accent_nov_{lag}"] = np.interp(frame_times, bt, nov["accent"]).astype(np.float32)
        out[f"phrase_amp_sim_{lag}"] = np.interp(frame_times, bt, sim["amp"]).astype(np.float32)
        out[f"phrase_amp_nov_{lag}"] = np.interp(frame_times, bt, nov["amp"]).astype(np.float32)
        out[f"phrase_aud_sim_{lag}"] = np.interp(frame_times, bt, sim["aud"]).astype(np.float32)
        out[f"phrase_aud_nov_{lag}"] = np.interp(frame_times, bt, nov["aud"]).astype(np.float32)
        out[f"phrase_spec_aud_sim_{lag}"] = np.interp(frame_times, bt, sim["spec_aud"]).astype(np.float32)
        out[f"phrase_spec_aud_nov_{lag}"] = np.interp(frame_times, bt, nov["spec_aud"]).astype(np.float32)

        out[f"phrase_half_sim_{lag}"] = np.interp(frame_times, bt, half_sim["phrase"]).astype(np.float32)
        out[f"phrase_half_nov_{lag}"] = np.interp(frame_times, bt, half_nov["phrase"]).astype(np.float32)
        out[f"phrase_spec_half_sim_{lag}"] = np.interp(frame_times, bt, half_sim["spec"]).astype(np.float32)
        out[f"phrase_spec_half_nov_{lag}"] = np.interp(frame_times, bt, half_nov["spec"]).astype(np.float32)
        out[f"phrase_env_half_sim_{lag}"] = np.interp(frame_times, bt, half_sim["env"]).astype(np.float32)
        out[f"phrase_env_half_nov_{lag}"] = np.interp(frame_times, bt, half_nov["env"]).astype(np.float32)
        out[f"phrase_accent_half_sim_{lag}"] = np.interp(frame_times, bt, half_sim["accent"]).astype(np.float32)
        out[f"phrase_accent_half_nov_{lag}"] = np.interp(frame_times, bt, half_nov["accent"]).astype(np.float32)
        out[f"phrase_amp_half_sim_{lag}"] = np.interp(frame_times, bt, half_sim["amp"]).astype(np.float32)
        out[f"phrase_amp_half_nov_{lag}"] = np.interp(frame_times, bt, half_nov["amp"]).astype(np.float32)
        out[f"phrase_aud_half_sim_{lag}"] = np.interp(frame_times, bt, half_sim["aud"]).astype(np.float32)
        out[f"phrase_aud_half_nov_{lag}"] = np.interp(frame_times, bt, half_nov["aud"]).astype(np.float32)
        out[f"phrase_spec_aud_half_sim_{lag}"] = np.interp(frame_times, bt, half_sim["spec_aud"]).astype(np.float32)
        out[f"phrase_spec_aud_half_nov_{lag}"] = np.interp(frame_times, bt, half_nov["spec_aud"]).astype(np.float32)

    return out


_ONSET_LAGS     = _STEM_COMPARISON_LAGS
_ONSET_TOL_BEATS = 0.5    # ±0.5 beats tolerance for matching onsets between blocks
_ONSET_TIMING_SIGMA_BEATS = 0.18
_ONSET_MIN_DIST  = 13     # min frames between detected peaks (~151ms @ 44.1kHz/512hop); calibrated to ~20 bass note attacks per 11s
_ONSET_RMS_FRAC  = 0.08   # peak must be ≥8% of block's max RMS
_ONSET_ACTIVITY_BINS = 32
_ONSET_ACTIVITY_FRAC = 0.18
_ONSET_EVENT_WEIGHT = 0.75
_ONSET_ACTIVITY_WEIGHT = 1.0 - _ONSET_EVENT_WEIGHT


def _onset_similarity_curves(
    Q_mag: np.ndarray,
    rms: np.ndarray,
    beat_frames: np.ndarray,
    beat_times: np.ndarray,
    frame_times: np.ndarray,
    sr: int,
    hop_length: int,
    lags: tuple[int, ...] = _ONSET_LAGS,
    tol_beats: float = _ONSET_TOL_BEATS,
    timing_sigma_beats: float = _ONSET_TIMING_SIGMA_BEATS,
    min_dist: int = _ONSET_MIN_DIST,
    rms_frac: float = _ONSET_RMS_FRAC,
    activity_bins: int = _ONSET_ACTIVITY_BINS,
    activity_frac: float = _ONSET_ACTIVITY_FRAC,
    event_weight: float = _ONSET_EVENT_WEIGHT,
    block_offsets: dict[int, int] | None = None,
) -> dict[str, np.ndarray]:
    """Block comparison via onset-based fingerprint matching.

    For each block, finds attack peaks, extracts a log-CQT spectral fingerprint
    at each peak, then optimally matches peaks between consecutive blocks by
    relative beat position.

    Event match score = spectral_sim * timing_sim, where timing_sim decays as
    matched attacks drift away from the same relative beat position.

    The final score also includes a coarse activity mask comparison so rests,
    valleys, and sustain length affect similarity without fragile local-minimum
    detection.

    Unmatched onsets count as zero, penalising both missed and extra events.
    Captures "does the same sequence of note attacks happen at the same times?"
    rather than comparing averaged spectral content per beat window.
    """
    from scipy.optimize import linear_sum_assignment as _linear_sum_assignment
    from scipy.signal import find_peaks as _find_peaks

    n_beats  = min(len(beat_times), len(beat_frames))
    n_f_rms  = len(rms)
    n_f_cqt  = Q_mag.shape[1]
    Q_log    = np.log1p(Q_mag)   # (84, n_f_cqt)
    bf       = np.clip(beat_frames[:n_beats], 0, min(n_f_rms, n_f_cqt) - 1).astype(int)
    n_f      = min(n_f_rms, n_f_cqt)
    rms_use  = np.asarray(rms[:n_f], dtype=np.float64)

    if float(np.max(Q_mag)) > 1e-12:
        S_db = librosa.amplitude_to_db(Q_mag[:, :n_f], ref=float(np.max(Q_mag)))
        spectral_onset = librosa.onset.onset_strength(
            S=S_db, sr=sr, hop_length=hop_length
        )[:n_f]
    else:
        spectral_onset = np.zeros(n_f, dtype=np.float64)

    rms_attack = np.maximum(
        np.diff(np.log1p(rms_use), prepend=0.0),
        0.0,
    )
    rms_attack01 = _normalize_01(rms_attack)
    spectral_onset01 = _normalize_01(np.asarray(spectral_onset, dtype=np.float64))
    attack_env = rms_attack01 if float(rms_attack01.max()) > 1e-7 else spectral_onset01
    activity_env = _smooth_1d(rms_use.astype(np.float32), win=3).astype(np.float64)

    median_beat_s = float(np.median(np.diff(beat_times[:n_beats]))) if n_beats > 1 else 0.5

    def block_frame_range(b_start: int, b_end: int) -> tuple[int, int]:
        f0 = int(bf[min(b_start, n_beats - 1)])
        if b_end < n_beats:
            f1 = int(bf[b_end])
        else:
            f1 = n_f
        return max(0, min(f0, n_f)), max(0, min(f1, n_f))

    def block_time_range(b_start: int, b_end: int) -> tuple[float, float]:
        t0 = float(beat_times[min(b_start, n_beats - 1)])
        if b_end < n_beats:
            t1 = float(beat_times[b_end])
        else:
            t1 = t0 + max(b_end - b_start, 1) * median_beat_s
        return t0, max(t1, t0 + 1e-6)

    def fingerprint(pf: int) -> np.ndarray:
        fa = max(0, pf - 2)
        fb = min(n_f_cqt, pf + 3)
        fp = Q_log[:, fa:fb].mean(axis=1)
        norm = np.linalg.norm(fp) + 1e-8
        return fp / norm

    def block_onsets(b_start: int, b_end: int) -> list[tuple[float, np.ndarray]]:
        """Return [(rel_beat_pos, L2-normed fingerprint), ...] for peaks in beat range."""
        f0, f1 = block_frame_range(b_start, b_end)
        if f1 <= f0:
            return []
        seg = attack_env[f0:f1]
        seg_max = seg.max()
        if seg_max < 1e-7:
            return []

        peaks_local, _ = _find_peaks(seg, height=rms_frac * seg_max, distance=min_dist)
        if seg[0] >= rms_frac * seg_max and (seg.size == 1 or seg[0] >= seg[1]):
            peaks_local = np.unique(np.concatenate([np.array([0], dtype=int), peaks_local]))
        if peaks_local.size == 0:
            return []

        t0, t1 = block_time_range(b_start, b_end)
        beat_dur_s = (t1 - t0) / max(b_end - b_start, 1)

        events = []
        for pl in peaks_local:
            pf = int(f0 + pl)
            t  = librosa.frames_to_time(pf, sr=sr, hop_length=hop_length)
            rel_pos = (t - t0) / beat_dur_s   # fractional beat units within block
            events.append((rel_pos, fingerprint(pf)))
        return events

    def block_activity_mask(b_start: int, b_end: int) -> np.ndarray:
        f0, f1 = block_frame_range(b_start, b_end)
        if f1 <= f0:
            return np.zeros(activity_bins, dtype=bool)
        seg = activity_env[f0:f1]
        if float(seg.max()) < 1e-7:
            return np.zeros(activity_bins, dtype=bool)

        edges = np.linspace(0, len(seg), activity_bins + 1, dtype=int)
        binned = np.zeros(activity_bins, dtype=np.float64)
        for i in range(activity_bins):
            lo, hi = int(edges[i]), int(edges[i + 1])
            if hi > lo:
                binned[i] = float(seg[lo:hi].mean())
        return binned >= (activity_frac * float(binned.max()))

    def activity_score(prev_mask: np.ndarray, cur_mask: np.ndarray) -> float:
        union = int(np.logical_or(prev_mask, cur_mask).sum())
        if union == 0:
            return 1.0
        inter = int(np.logical_and(prev_mask, cur_mask).sum())
        return inter / union

    def event_match_score(prev_ev: list, cur_ev: list, tol: float) -> float | None:
        if not prev_ev and not cur_ev:
            return None
        if not prev_ev or not cur_ev:
            return 0.0

        n_total = max(len(prev_ev), len(cur_ev))
        scores = np.zeros((len(cur_ev), len(prev_ev)), dtype=np.float64)

        for i, (rel_c, fp_c) in enumerate(cur_ev):
            for j, (rel_p, fp_p) in enumerate(prev_ev):
                d = abs(rel_c - rel_p)
                if d > tol:
                    continue
                spectral = max(0.0, float(np.dot(fp_c, fp_p)))
                timing = float(np.exp(-((d / max(timing_sigma_beats, 1e-6)) ** 2)))
                scores[i, j] = spectral * timing

        if not np.any(scores > 0.0):
            return 0.0

        row_ind, col_ind = _linear_sum_assignment(-scores)
        return float(scores[row_ind, col_ind].sum()) / n_total

    def match_score(
        prev_ev: list,
        cur_ev: list,
        prev_mask: np.ndarray,
        cur_mask: np.ndarray,
    ) -> float:
        activity = activity_score(prev_mask, cur_mask)
        events = event_match_score(prev_ev, cur_ev, tol_beats)
        if events is None:
            return activity
        ew = float(np.clip(event_weight, 0.0, 1.0))
        return ew * events + (1.0 - ew) * activity

    out: dict[str, np.ndarray] = {}
    bt = beat_times[:n_beats]

    for lag in lags:
        sim_beat = np.ones(n_beats, dtype=np.float32)
        nov_beat = np.zeros(n_beats, dtype=np.float32)
        sim_half_beat = np.ones(n_beats, dtype=np.float32)
        nov_half_beat = np.zeros(n_beats, dtype=np.float32)
        offset = _block_offset_for_lag(block_offsets, lag)

        def score_pair(prv_s: int, cur_s: int, length: int) -> float:
            prev_ev = block_onsets(prv_s, prv_s + length)
            cur_ev  = block_onsets(cur_s,  cur_s  + length)
            prev_mask = block_activity_mask(prv_s, prv_s + length)
            cur_mask  = block_activity_mask(cur_s,  cur_s  + length)
            return match_score(prev_ev, cur_ev, prev_mask, cur_mask)

        for prv_s, cur_s, L in _full_block_pairs(n_beats, offset, lag):
            ms = score_pair(prv_s, cur_s, L)
            sim_beat[cur_s:cur_s + L] = ms
            nov_beat[cur_s:cur_s + L] = 1.0 - ms

        for prv_s, cur_s, L in _matched_half_block_pairs(n_beats, offset, lag):
            ms = score_pair(prv_s, cur_s, L)
            sim_half_beat[cur_s:cur_s + L] = ms
            nov_half_beat[cur_s:cur_s + L] = 1.0 - ms

        out[f"onset_sim_{lag}"] = np.interp(frame_times, bt, sim_beat).astype(np.float32)
        out[f"onset_nov_{lag}"] = np.interp(frame_times, bt, nov_beat).astype(np.float32)
        out[f"onset_half_sim_{lag}"] = np.interp(frame_times, bt, sim_half_beat).astype(np.float32)
        out[f"onset_half_nov_{lag}"] = np.interp(frame_times, bt, nov_half_beat).astype(np.float32)

    return out


_SILENCE_RMS_FRAC = 0.05
_SILENCE_MIN_S    = 0.8
_STEM_ENTER_FRAC  = 0.10
_STEM_EXIT_FRAC   = 0.05
_STEM_MIN_HOLD_S  = 0.6


def _detect_silence_events(
    rms: np.ndarray,
    times_s: np.ndarray,
    *,
    min_s: float = _SILENCE_MIN_S,
    frac: float = _SILENCE_RMS_FRAC,
) -> list[dict[str, float]]:
    """Return contiguous silent regions lasting ≥ min_s."""
    if rms.size == 0:
        return []
    thr = float(rms.max()) * frac
    below = rms < thr
    events: list[dict[str, float]] = []
    i, n = 0, len(below)
    while i < n:
        if not below[i]:
            i += 1
            continue
        j = i
        while j < n and below[j]:
            j += 1
        t0 = float(times_s[i])
        t1 = float(times_s[j - 1])
        if (t1 - t0) >= min_s:
            events.append({
                "start_s": t0,
                "end_s": t1,
                "min_rms": float(rms[i:j].min()),
            })
        i = j
    return events


def _detect_stem_transitions(
    stems: dict[str, np.ndarray],
    sr: int,
    *,
    hop_length: int,
    frame_length: int,
) -> list[dict[str, Any]]:
    """Detect per-stem enter/exit events with hysteresis to suppress flicker."""
    out: list[dict[str, Any]] = []
    min_hold_frames = max(1, int(round(_STEM_MIN_HOLD_S * sr / hop_length)))
    for name, stem_y in stems.items():
        if stem_y is None or stem_y.size == 0:
            continue
        rms = librosa.feature.rms(
            y=stem_y, frame_length=frame_length, hop_length=hop_length
        )[0]
        if rms.size == 0:
            continue
        peak = float(rms.max())
        if peak <= 0:
            continue
        enter_thr = peak * _STEM_ENTER_FRAC
        exit_thr  = peak * _STEM_EXIT_FRAC
        rms_t = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

        active = False
        run = 0
        for i, v in enumerate(rms):
            target = (v >= enter_thr) if not active else (v >= exit_thr)
            if target == active:
                run = 0  # consistent with current state — reset flip counter
            else:
                run += 1  # counting consecutive frames pushing toward a flip
            if run >= min_hold_frames:
                active = target
                out.append({
                    "stem": name,
                    "kind": "enter" if active else "exit",
                    "time_s": float(rms_t[max(0, i - min_hold_frames + 1)]),
                })
                run = 0
    return out


def compute_story(
    y: np.ndarray,
    sr: int,
    *,
    hop_length: int = 512,
    frame_length: int = 2048,
    other_y: np.ndarray | None = None,
    stems: dict[str, np.ndarray] | None = None,
    beat_times_s: list[float] | np.ndarray | None = None,
    beat_grid_source: str | None = None,
) -> dict[str, Any]:
    """
    Heuristic "song story" signals:
    - sections: coarse segmentation with motif-aware labels (same letter = same sound).
    - tension: a smooth energy/brightness curve (0..1), good for buildup/drop dynamics.
    - events: drop_times_s (sharp tension drops) and buildups (rising tension windows).

    An explicit beat_times_s grid bypasses beat tracking. Exact requested times,
    effective frame-aligned times and fallback provenance are saved in meta.
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
    # Resolve outside the section fallback handler: invalid supplied grids must
    # fail, not silently turn into a different timing hypothesis.
    beat_grid = prepare_beat_grid(y, sr, hop_length=hop_length, n_frames=n,
                                 beat_times_s=beat_times_s, source=beat_grid_source)
    beat_times: np.ndarray | None = np.asarray(beat_grid["effective_times_s"])
    section_error: str | None = None
    R_chroma_ssm: np.ndarray | None = None
    R_mfcc_ssm: np.ndarray | None = None

    try:
        # Only run SSM if the signal has meaningful temporal variation.
        # A flat waveform (constant sine, silence) produces degenerate SSMs
        # where numerical noise gets amplified into fake structure.
        _has_structure = float(np.std(onset01)) > 0.08 or float(np.std(rms01)) > 0.08

        C_sync, M_sync, C_sync_raw, M_sync_raw, beat_times, beat_dur = _beat_sync_features(
            y, sr, hop_length=hop_length, n_frames=n, mfcc=mfcc,
            beat_frames_override=np.asarray(beat_grid["frame_indices"]),
        )
        n_beats = C_sync.shape[-1]

        if _has_structure:
            R_chroma = _build_ssm(C_sync)
            R_mfcc = _build_ssm(M_sync)
            R_chroma_ssm = R_chroma
            R_mfcc_ssm = R_mfcc

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

        # Combine: scored intersection — agreed or prominent boundaries only
        internal = _score_and_filter_boundaries(
            bounds_ssm, bounds_energy,
            novelty=novelty if _has_structure else None,
            beat_times=beat_times,
            duration_s=duration_s,
        )
        bounds_s = [0.0] + internal + [duration_s]
        # min_len_s=2.0: only deduplicate near-identical boundaries from
        # different detectors; do not enforce a minimum section length.
        # Quality filtering is already handled by _score_and_filter_boundaries.
        _pre_merge_bounds = list(bounds_s)
        bounds_s = _merge_short_segments(bounds_s, min_len_s=2.0, duration_s=duration_s)
        _kept_set = {round(b, 3) for b in bounds_s}
        boundary_candidates: list[dict[str, Any]] = [
            {"time_s": float(b), "source": "ssm_discarded_short"}
            for b in _pre_merge_bounds
            if 0.0 < b < duration_s and round(b, 3) not in _kept_set
        ]
        # Inject a quiet-intro boundary AFTER the merge pass (so it's not
        # dropped by the min_len gate). The SSM checkerboard can't detect
        # song-start onset: there's no prior context to form self-similarity.
        _intro_end = _detect_intro_onset_boundary(tension, times_s, duration_s=duration_s)
        if _intro_end is not None and not any(
            0 < b < _intro_end + 5.0 for b in bounds_s[1:]  # skip 0.0
        ):
            bounds_s = sorted(set(bounds_s + [_intro_end]))

        # Inject the FIRST bass/drums entry in the opening 20% of the song as a
        # section boundary.  The SSM misses these because there is no prior
        # context at song-start; only one boundary per stem is injected so that
        # the rhythmic enter/exit pattern inside a riff doesn't produce clutter.
        if stems:
            _early_stem_tr = _detect_stem_transitions(
                stems, sr, hop_length=hop_length, frame_length=frame_length,
            )
            _early_cutoff = duration_s * 0.20
            _stems_injected: set[str] = set()
            for _tr in _early_stem_tr:
                if (
                    _tr["kind"] == "enter"
                    and _tr["stem"] in ("bass", "drums")
                    and 0 < _tr["time_s"] < _early_cutoff
                    and _tr["stem"] not in _stems_injected
                    and not any(abs(_tr["time_s"] - b) < 3.0 for b in bounds_s)
                ):
                    bounds_s = sorted(set(bounds_s + [_tr["time_s"]]))
                    _stems_injected.add(_tr["stem"])
        # Force-split any section that is still too long
        bounds_s = _force_split_long_sections(
            bounds_s,
            tension=raw_tension_smoothed,
            times_s=times_s,
            max_section_s=_MAX_SECTION_S,
        )

        _ssm_ok = True

    except Exception as exc:
        section_error = f"{type(exc).__name__}: {exc}"
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
        boundary_candidates = []

        if len(bounds_s) <= 3 and duration_s > 60.0:
            bounds_s = _tension_valley_boundaries(
                raw_tension_smoothed, times_s, min_len_s=15.0, duration_s=duration_s,
            )

    # --- Time-lag similarity matrix (beat-synced; uses "other" stem if available) ---
    y_nov = other_y if other_y is not None else y
    C_raw = librosa.feature.chroma_cqt(y=y_nov, sr=sr, hop_length=hop_length)
    mfcc_nov = librosa.feature.mfcc(y=y_nov, sr=sr, n_mfcc=20, hop_length=hop_length, n_fft=frame_length)

    if beat_times is not None and len(beat_times) > 1:
        _bf_nov = librosa.time_to_frames(beat_times, sr=sr, hop_length=hop_length)
        C_nov_sync = librosa.util.sync(C_raw, _bf_nov, aggregate=np.median)
        M_nov_sync = librosa.util.sync(mfcc_nov, _bf_nov, aggregate=np.mean)
        _bt_nov = beat_times
    else:
        _bt_nov = np.arange(0, duration_s, 0.5)
        _bf_nov = librosa.time_to_frames(_bt_nov, sr=sr, hop_length=hop_length)
        C_nov_sync = librosa.util.sync(C_raw, _bf_nov, aggregate=np.median)
        M_nov_sync = librosa.util.sync(mfcc_nov, _bf_nov, aggregate=np.mean)
        _bt_nov = _bt_nov[: C_nov_sync.shape[1]]

    h_lag_matrix = _lag_matrix_novelty(C_nov_sync, _bt_nov, times_s)
    t_lag_matrix = _lag_matrix_novelty(
        M_nov_sync[1:] if M_nov_sync.shape[0] > 1 else M_nov_sync, _bt_nov, times_s
    )
    novelty_history = _lag_history_mask(_bt_nov, times_s)
    nov_curves = _novelty_curves_from_lag(h_lag_matrix, t_lag_matrix, history_mask=novelty_history)

    if R_chroma_ssm is not None and R_mfcc_ssm is not None and beat_times is not None:
        rep = _rep_strength(R_chroma_ssm, R_mfcc_ssm, beat_times, times_s)
    else:
        rep = np.zeros(n, dtype=np.float32)

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
    sections = _merge_same_label_sections(sections, max_merged_len_s=50.0)

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

    stem_novelties: dict[str, Any] = {}
    if stems:
        alignment_stem = next(
            (
                name for name, stem_y in stems.items()
                if str(name).lower() in {"drum", "drums"} and stem_y is not None and stem_y.size > 0
            ),
            None,
        )
        shared_block_offsets: dict[int, int] | None = None
        shared_bar_grid: dict[str, Any] | None = None
        if alignment_stem is not None:
            try:
                drum_result = _stem_novelties(
                    stems[alignment_stem], sr, n_frames=n,
                    hop_length=hop_length, frame_length=frame_length,
                    beat_times=beat_times, frame_times=times_s,
                    stem_name=alignment_stem,
                    alignment_source_stem=alignment_stem,
                )
                stem_novelties[alignment_stem] = drum_result
                shared_block_offsets = {
                    int(k): int(v)
                    for k, v in drum_result.get("block_offsets", {}).items()
                }
                shared_bar_grid = dict(drum_result.get("bar_grid", {}))
            except Exception:
                shared_block_offsets = None
                shared_bar_grid = None

        for stem_name, stem_y in stems.items():
            if stem_name in stem_novelties:
                continue
            if stem_y is not None and stem_y.size > 0:
                try:
                    stem_novelties[stem_name] = _stem_novelties(
                        stem_y, sr, n_frames=n,
                        hop_length=hop_length, frame_length=frame_length,
                        beat_times=beat_times, frame_times=times_s,
                        stem_name=stem_name,
                        block_offsets_override=shared_block_offsets,
                        bar_grid_override=shared_bar_grid,
                        alignment_source_stem=alignment_stem or stem_name,
                    )
                except Exception:
                    pass

    # Build section-diff step function: how different is each section from the previous one?
    section_diff_curve = np.zeros(n, dtype=np.float32)
    for sec in sections:
        val = float(sec.get("novelty_to_prev", 0.0))
        s0_f = max(0, int(round(sec["start_s"] / hop_s)))
        s1_f = max(s0_f + 1, min(int(round(sec["end_s"] / hop_s)), n))
        section_diff_curve[s0_f:s1_f] = val

    silences = _detect_silence_events(rms[:n], times_s[:n])
    stem_transitions = (
        _detect_stem_transitions(
            stems, sr, hop_length=hop_length, frame_length=frame_length
        )
        if stems else []
    )

    return {
        "sections": sections,
        "tension": {
            "hop_s": float(hop_s),
            "times_s": times_s.astype(float).tolist(),
            "value": tension.astype(float).tolist(),
        },
        "novelties": {
            "times_s": times_s.astype(float).tolist(),
            "history_valid": novelty_history.any(axis=0).tolist(),
            "novelty_short": nov_curves["short"].astype(float).tolist(),
            "novelty_medium": nov_curves["medium"].astype(float).tolist(),
            "novelty_long": nov_curves["long"].astype(float).tolist(),
            "repetition": rep.astype(float).tolist(),
            "section_diff": section_diff_curve.astype(float).tolist(),
        },
        "stem_novelties": stem_novelties,
        "events": {
            "drop_times_s": drop_times,
            "buildups": buildups,
            "silences": silences,
            "stem_transitions": stem_transitions,
            "boundary_candidates": boundary_candidates,
        },
        "meta": {
            "duration_s": float(duration_s),
            "sample_rate": int(sr),
            "features": ["mfcc_20", "rms", "onset_strength", "spectral_centroid"],
            "beat_grid": beat_grid,
            "section_method": "ssm" if _ssm_ok else "fallback",
            "section_error": section_error,
            "boundary_method": "ssm_energy_one_to_one_v2" if _ssm_ok else "agglomerative_fallback",
            "novelty_method": "lag_cosine_v2_history_masked_no_rescale",
            "novelty_semantics": "nearest available past beat within 4/16/32 beats; missing history is unknown (zero placeholder), not surprise; clipped cosine, not a probability",
        },
    }
