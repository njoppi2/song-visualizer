"""
Bar-phase and phrase-start detection experiments.

All public functions are pure (numpy in, dict out). No audio I/O, no plotting.
Called from experiments/align_structure.py.
"""
from __future__ import annotations

from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _beat_to_frame(beat_times_s: np.ndarray, sr: int, hop_length: int) -> np.ndarray:
    return np.round(np.asarray(beat_times_s, dtype=np.float64) * sr / hop_length).astype(int)


def _sample_at_beats(
    feature: np.ndarray,
    beat_frames: np.ndarray,
    *,
    half_window: int = 2,
) -> np.ndarray:
    """Max of a small window around each beat frame (captures transient peaks)."""
    n = len(feature)
    result = np.zeros(len(beat_frames), dtype=np.float64)
    for i, bf in enumerate(beat_frames):
        lo = max(0, int(bf) - half_window)
        hi = min(n, int(bf) + half_window + 1)
        result[i] = float(np.max(feature[lo:hi])) if hi > lo else 0.0
    return result


def _phase_groups(
    n_beats: int, phase: int, beats_per_bar: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    """Return (downbeat_indices, offbeat_indices) for a candidate phase."""
    indices = np.arange(n_beats)
    positions = (indices - phase) % beats_per_bar
    return indices[positions == 0], indices[positions != 0]


def _safe_mean(arr: np.ndarray) -> float:
    return float(np.mean(arr)) if len(arr) > 0 else 0.0


def _confidence_margin(scores: dict[int, float]) -> float:
    vals = sorted(scores.values(), reverse=True)
    return max(0.0, vals[0] - vals[1]) if len(vals) >= 2 else 0.0


def _normalise_scores(scores: dict[int, float]) -> dict[int, float]:
    """Shift min→0, scale max→1."""
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    spread = hi - lo
    if spread < 1e-9:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / spread for k, v in scores.items()}


def _best_phase(scores: dict[int, float]) -> int:
    return int(max(scores, key=scores.__getitem__))


def beats_to_bar_times(
    beat_times_s: np.ndarray, bar_phase: int, beats_per_bar: int = 4
) -> np.ndarray:
    """Return times of bar-start beats (position 0 of each bar) in seconds."""
    indices = np.arange(len(beat_times_s))
    mask = (indices - bar_phase) % beats_per_bar == 0
    return np.asarray(beat_times_s)[mask]


# ---------------------------------------------------------------------------
# Feature extraction helper
# ---------------------------------------------------------------------------

def compute_beat_features(
    y_mix: np.ndarray | None,
    y_drums: np.ndarray | None,
    y_bass: np.ndarray | None,
    y_other: np.ndarray | None,
    sr: int,
    hop_length: int,
    beat_times_s: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Compute frame-level features for all available stems. Returns a dict with
    keys: onset_env, kick_energy, snare_energy, bass_onset, chroma_novelty.
    Missing stems produce absent keys.
    """
    import librosa
    from songviz.features import drums_band_energy_3

    out: dict[str, np.ndarray] = {}

    if y_mix is not None:
        out["onset_env"] = librosa.onset.onset_strength(
            y=y_mix, sr=sr, hop_length=hop_length
        ).astype(np.float32)
        chroma = librosa.feature.chroma_cqt(y=y_mix, sr=sr, hop_length=hop_length)
        chroma_diff = np.zeros(chroma.shape[1], dtype=np.float32)
        chroma_diff[1:] = np.linalg.norm(np.diff(chroma, axis=1), axis=0).astype(np.float32)
        out["chroma_novelty"] = chroma_diff

    if y_drums is not None:
        try:
            bands = drums_band_energy_3(y_drums, sr, hop_length=hop_length)
            out["kick_energy"] = bands[:, 0].astype(np.float32)
            out["snare_energy"] = bands[:, 1].astype(np.float32)
        except Exception:
            pass
        drum_onset = librosa.onset.onset_strength(
            y=y_drums, sr=sr, hop_length=hop_length
        ).astype(np.float32)
        if "onset_env" not in out:
            out["onset_env"] = drum_onset
        out["drum_onset"] = drum_onset

    if y_bass is not None:
        out["bass_onset"] = librosa.onset.onset_strength(
            y=y_bass, sr=sr, hop_length=hop_length
        ).astype(np.float32)

    return out


# ---------------------------------------------------------------------------
# Method 1 — Raw onset-strength phase vote
# ---------------------------------------------------------------------------

def onset_strength_phase_vote(
    beat_times_s: np.ndarray,
    onset_env: np.ndarray,
    sr: int,
    hop_length: int,
    *,
    beats_per_bar: int = 4,
) -> dict[str, Any]:
    """
    Score each candidate bar phase by:
        mean(onset_strength at position 0) − mean(onset_strength at positions 1…N-1)
    Simple baseline. Expects kick/full-mix onset_env.
    """
    beat_frames = np.clip(_beat_to_frame(beat_times_s, sr, hop_length), 0, len(onset_env) - 1)
    beat_onset = _sample_at_beats(onset_env, beat_frames)

    scores: dict[int, float] = {}
    for p in range(beats_per_bar):
        db_idx, ob_idx = _phase_groups(len(beat_times_s), p, beats_per_bar)
        if len(db_idx) < 2:
            scores[p] = 0.0
            continue
        scores[p] = _safe_mean(beat_onset[db_idx]) - (
            _safe_mean(beat_onset[ob_idx]) if len(ob_idx) > 0 else 0.0
        )

    bp = _best_phase(scores)
    return {
        "method": "onset_strength_phase_vote",
        "phase": bp,
        "scores": {str(k): float(v) for k, v in scores.items()},
        "scores_normalised": {str(k): float(v) for k, v in _normalise_scores(scores).items()},
        "confidence": float(_confidence_margin(scores)),
        "beats_per_bar": beats_per_bar,
    }


# ---------------------------------------------------------------------------
# Method 2 — Multi-feature metrical-role score
# ---------------------------------------------------------------------------

_DEFAULT_M2_WEIGHTS: dict[str, float] = {
    "downbeat_contrast": 0.50,
    "backbeat_score": 0.30,
    "pattern_consistency": 0.20,
}


def multi_feature_metrical_score(
    beat_times_s: np.ndarray,
    features: dict[str, np.ndarray],
    sr: int,
    hop_length: int,
    *,
    beats_per_bar: int = 4,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Per-phase score combining:
        downbeat_contrast  — kick/onset/bass higher at position 0
        backbeat_score     — snare higher at positions 1 and 3
        pattern_consistency — 4-beat patterns repeat stably under this phase

    features keys (frame-level, all optional):
        onset_env, kick_energy, snare_energy, bass_onset, chroma_novelty, drum_onset
    """
    if weights is None:
        weights = _DEFAULT_M2_WEIGHTS.copy()
    w_db = float(weights.get("downbeat_contrast", 0.50))
    w_bb = float(weights.get("backbeat_score", 0.30))
    w_pc = float(weights.get("pattern_consistency", 0.20))

    beat_frames = _beat_to_frame(beat_times_s, sr, hop_length)

    beat_feats: dict[str, np.ndarray] = {}
    for key, arr in features.items():
        if arr is None or len(arr) == 0:
            continue
        arr = np.asarray(arr, dtype=np.float64)
        bf = np.clip(beat_frames, 0, len(arr) - 1)
        beat_feats[key] = _sample_at_beats(arr, bf)

    if not beat_feats:
        empty = {str(p): 0.0 for p in range(beats_per_bar)}
        return {
            "method": "multi_feature_metrical_score",
            "phase": 0, "scores": empty, "scores_normalised": empty,
            "sub_scores": {}, "confidence": 0.0, "beats_per_bar": beats_per_bar,
            "warning": "no features provided",
        }

    # Downbeat evidence: prefer kick > drum onset > full-mix onset > bass
    for key in ("kick_energy", "drum_onset", "onset_env", "bass_onset"):
        if key in beat_feats:
            db_evidence = beat_feats[key]
            break
    else:
        db_evidence = next(iter(beat_feats.values()))

    # Backbeat evidence: snare or drum onset
    bb_evidence = beat_feats.get("snare_energy", beat_feats.get("drum_onset", db_evidence))

    # Feature matrix for pattern consistency
    feat_matrix = np.stack(list(beat_feats.values()), axis=0)  # (F, B)
    n_beats = feat_matrix.shape[1]

    scores: dict[int, float] = {}
    sub: dict[str, dict[str, float]] = {}

    for p in range(beats_per_bar):
        db_idx, ob_idx = _phase_groups(n_beats, p, beats_per_bar)

        # Downbeat contrast
        dc = (
            _safe_mean(db_evidence[db_idx]) - _safe_mean(db_evidence[ob_idx])
            if len(db_idx) >= 2 and len(ob_idx) >= 2
            else 0.0
        )

        # Backbeat score
        indices = np.arange(n_beats)
        positions = (indices - p) % beats_per_bar
        bb_idx = indices[(positions == 1) | (positions == 3)]
        non_bb_idx = indices[(positions == 0) | (positions == 2)]
        bb = (
            _safe_mean(bb_evidence[bb_idx]) - _safe_mean(bb_evidence[non_bb_idx])
            if len(bb_idx) >= 2 and len(non_bb_idx) >= 2
            else 0.0
        )

        # Pattern consistency
        bar_start = int(p % beats_per_bar)
        valid = n_beats - bar_start
        n_complete = valid // beats_per_bar
        if n_complete >= 4:
            seg = feat_matrix[:, bar_start:bar_start + n_complete * beats_per_bar]
            bars = seg.reshape(feat_matrix.shape[0], n_complete, beats_per_bar)
            mean_pat = bars.mean(axis=1, keepdims=True)
            deviation = float(np.abs(bars - mean_pat).mean())
            max_val = float(feat_matrix.max()) or 1.0
            pc = float(1.0 - np.clip(deviation / max_val, 0.0, 1.0))
        else:
            pc = 0.0

        total = w_db * dc + w_bb * bb + w_pc * pc
        scores[p] = total
        sub[str(p)] = {
            "downbeat_contrast": float(dc),
            "backbeat_score": float(bb),
            "pattern_consistency": float(pc),
            "total": float(total),
        }

    bp = _best_phase(scores)
    return {
        "method": "multi_feature_metrical_score",
        "phase": bp,
        "scores": {str(k): float(v) for k, v in scores.items()},
        "scores_normalised": {str(k): float(v) for k, v in _normalise_scores(scores).items()},
        "sub_scores": sub,
        "confidence": float(_confidence_margin(scores)),
        "beats_per_bar": beats_per_bar,
        "weights": {k: float(v) for k, v in weights.items()},
    }


# ---------------------------------------------------------------------------
# Method 3 — Phrase-grid boundary contrast
# ---------------------------------------------------------------------------

def _build_bar_novelty(
    beat_feats: dict[str, np.ndarray],
    bar_phase: int,
    beats_per_bar: int = 4,
) -> np.ndarray:
    """Return per-bar novelty scalar (mean abs change from previous bar)."""
    if not beat_feats:
        return np.zeros(1)
    n_beats = min(len(v) for v in beat_feats.values())
    feat_matrix = np.stack([v[:n_beats] for v in beat_feats.values()], axis=0)
    start = int(bar_phase) % beats_per_bar
    n_complete = (n_beats - start) // beats_per_bar
    if n_complete < 2:
        return np.zeros(max(1, n_complete))
    seg = feat_matrix[:, start:start + n_complete * beats_per_bar]
    bar_mean = seg.reshape(feat_matrix.shape[0], n_complete, beats_per_bar).mean(axis=2)
    novelty = np.zeros(n_complete, dtype=np.float64)
    novelty[1:] = np.linalg.norm(np.diff(bar_mean, axis=1), axis=0)
    return novelty


def phrase_grid_boundary_contrast(
    beat_times_s: np.ndarray,
    features: dict[str, np.ndarray],
    sr: int,
    hop_length: int,
    bar_phase: int,
    *,
    phrase_lengths: list[int] | None = None,
    beats_per_bar: int = 4,
) -> dict[str, Any]:
    """
    For phrase length L and offset o: proposed boundaries at bars o, o+L, o+2L, …
    Score = mean(novelty at boundaries) − mean(novelty elsewhere).
    Returns ranked candidate table.
    """
    if phrase_lengths is None:
        phrase_lengths = [8, 16, 32]

    beat_frames = _beat_to_frame(beat_times_s, sr, hop_length)
    beat_feats: dict[str, np.ndarray] = {}
    for key, arr in features.items():
        if arr is None or len(arr) == 0:
            continue
        arr = np.asarray(arr, dtype=np.float64)
        bf = np.clip(beat_frames, 0, len(arr) - 1)
        beat_feats[key] = _sample_at_beats(arr, bf)

    bar_novelty = _build_bar_novelty(beat_feats, bar_phase, beats_per_bar)
    n_bars = len(bar_novelty)

    candidates: list[dict[str, Any]] = []
    for L in phrase_lengths:
        if L > n_bars // 2:
            continue
        for o in range(L):
            b_idx = np.arange(o, n_bars, L)
            nb_idx = np.setdiff1d(np.arange(n_bars), b_idx)
            if len(b_idx) < 2 or len(nb_idx) < 2:
                continue
            b_score = _safe_mean(bar_novelty[b_idx])
            nb_score = _safe_mean(bar_novelty[nb_idx])
            candidates.append({
                "phrase_length": int(L),
                "offset": int(o),
                "boundary_mean_novelty": float(b_score),
                "nonboundary_mean_novelty": float(nb_score),
                "contrast": float(b_score - nb_score),
                "n_phrases": int(len(b_idx)),
            })

    if not candidates:
        return {
            "method": "phrase_grid_boundary_contrast",
            "best_phrase_length": phrase_lengths[0], "best_offset": 0,
            "candidates": [], "n_bars": int(n_bars), "bar_phase": int(bar_phase),
        }

    candidates.sort(key=lambda c: c["contrast"], reverse=True)
    best_per_length = {}
    for L in phrase_lengths:
        by_L = [c for c in candidates if c["phrase_length"] == L]
        if by_L:
            best_per_length[str(L)] = by_L[0]

    return {
        "method": "phrase_grid_boundary_contrast",
        "best_phrase_length": int(candidates[0]["phrase_length"]),
        "best_offset": int(candidates[0]["offset"]),
        "best_contrast": float(candidates[0]["contrast"]),
        "candidates": candidates[:30],
        "best_per_length": best_per_length,
        "n_bars": int(n_bars),
        "bar_phase": int(bar_phase),
    }


# ---------------------------------------------------------------------------
# Method 4 — Similarity-based phrase alignment
# ---------------------------------------------------------------------------

def _cosine_sim_01(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))


def similarity_phrase_alignment(
    beat_times_s: np.ndarray,
    features: dict[str, np.ndarray],
    sr: int,
    hop_length: int,
    bar_phase: int,
    *,
    phrase_lengths: list[int] | None = None,
    beats_per_bar: int = 4,
) -> dict[str, Any]:
    """
    For each (phrase_length, offset): segment into phrase chunks, compute
    adjacent/all-pair/within similarity metrics. Does NOT maximise similarity —
    ranks by stability_cleanliness (high adj std = sharp transitions).
    """
    if phrase_lengths is None:
        phrase_lengths = [8, 16, 32]

    beat_frames = _beat_to_frame(beat_times_s, sr, hop_length)
    beat_feats: dict[str, np.ndarray] = {}
    for key, arr in features.items():
        if arr is None or len(arr) == 0:
            continue
        arr = np.asarray(arr, dtype=np.float64)
        bf = np.clip(beat_frames, 0, len(arr) - 1)
        beat_feats[key] = _sample_at_beats(arr, bf)

    if not beat_feats:
        return {"method": "similarity_phrase_alignment", "candidates": [], "all_candidates": []}

    feat_matrix = np.stack(list(beat_feats.values()), axis=0)
    n_beats = feat_matrix.shape[1]
    bar_start = int(bar_phase) % beats_per_bar

    all_candidates: list[dict[str, Any]] = []
    for L in phrase_lengths:
        bpp = L * beats_per_bar
        for o_bars in range(L):
            seg_start = bar_start + o_bars * beats_per_bar
            if seg_start >= n_beats:
                continue
            n_phrases = (n_beats - seg_start) // bpp
            if n_phrases < 3:
                continue

            phrase_vecs = []
            for pi in range(n_phrases):
                s = seg_start + pi * bpp
                phrase_vecs.append(feat_matrix[:, s:s + bpp].mean(axis=1))
            pv = np.stack(phrase_vecs, axis=0)

            adj_sims = [_cosine_sim_01(pv[i], pv[i + 1]) for i in range(len(pv) - 1)]
            adj_mean = float(np.mean(adj_sims))
            adj_std = float(np.std(adj_sims))

            half = bpp // 2
            within_sims = []
            for pi in range(n_phrases):
                s = seg_start + pi * bpp
                if s + bpp > n_beats:
                    break
                within_sims.append(_cosine_sim_01(
                    feat_matrix[:, s:s + half].mean(axis=1),
                    feat_matrix[:, s + half:s + bpp].mean(axis=1),
                ))
            within_consistency = float(np.mean(within_sims)) if within_sims else 0.0

            P = len(pv)
            pair_sims = [
                _cosine_sim_01(pv[i], pv[j])
                for i in range(P) for j in range(i + 2, P)
            ]
            repeated_sim = float(np.mean(pair_sims)) if pair_sims else 0.0

            all_candidates.append({
                "phrase_length": int(L),
                "offset": int(o_bars),
                "n_phrases": int(n_phrases),
                "adj_similarity_mean": adj_mean,
                "adj_similarity_std": adj_std,
                "boundary_novelty": float(1.0 - adj_mean),
                "within_consistency": within_consistency,
                "repeated_phrase_similarity": repeated_sim,
                "stability_cleanliness": adj_std,
            })

    top = sorted(all_candidates, key=lambda c: c["stability_cleanliness"], reverse=True)
    return {
        "method": "similarity_phrase_alignment",
        "candidates": top[:20],
        "all_candidates": all_candidates,
    }


# ---------------------------------------------------------------------------
# Method 5 — Hybrid score
# ---------------------------------------------------------------------------

_DEFAULT_HYBRID_WEIGHTS: dict[str, float] = {
    "metrical_role_score": 0.30,
    "bar_pattern_consistency": 0.20,
    "phrase_boundary_contrast": 0.25,
    "repeated_phrase_similarity": 0.15,
    "stability_curve_cleanliness": 0.10,
}


def hybrid_score(
    m2_result: dict[str, Any],
    m3_result: dict[str, Any],
    m4_result: dict[str, Any],
    *,
    phrase_lengths: list[int] | None = None,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Combine M2/M3/M4 into ranked (bar_phase, phrase_length, phrase_offset) candidates.
    Normalises each component to [0, 1] before weighting.
    """
    if phrase_lengths is None:
        phrase_lengths = [8, 16, 32]
    if weights is None:
        weights = _DEFAULT_HYBRID_WEIGHTS.copy()

    bar_phase = int(m2_result.get("phase", m3_result.get("bar_phase", 0)))
    m2_phase_score = float(m2_result.get("scores_normalised", {}).get(str(bar_phase), 0.0))

    m3_map = {(c["phrase_length"], c["offset"]): c for c in m3_result.get("candidates", [])}
    m4_map = {(c["phrase_length"], c["offset"]): c for c in m4_result.get("all_candidates", [])}
    all_keys = set(m3_map) | set(m4_map)

    def _norm(d: dict) -> dict:
        if not d:
            return {}
        lo, hi = min(d.values()), max(d.values())
        return {k: (v - lo) / (hi - lo) if hi > lo else 0.0 for k, v in d.items()}

    norm_contrast = _norm({k: c["contrast"] for k, c in m3_map.items()})
    norm_consist = _norm({k: c["within_consistency"] for k, c in m4_map.items()})
    norm_repeat = _norm({k: c["repeated_phrase_similarity"] for k, c in m4_map.items()})
    norm_stab = _norm({k: c["stability_cleanliness"] for k, c in m4_map.items()})

    candidates_out: list[dict[str, Any]] = []
    for (L, o) in sorted(all_keys):
        pbc = norm_contrast.get((L, o), 0.0)
        bpc = norm_consist.get((L, o), 0.0)
        rep = norm_repeat.get((L, o), 0.0)
        stab = norm_stab.get((L, o), 0.0)

        total = (
            weights.get("metrical_role_score", 0.30) * m2_phase_score
            + weights.get("bar_pattern_consistency", 0.20) * bpc
            + weights.get("phrase_boundary_contrast", 0.25) * pbc
            + weights.get("repeated_phrase_similarity", 0.15) * rep
            + weights.get("stability_curve_cleanliness", 0.10) * stab
        )
        candidates_out.append({
            "bar_phase": int(bar_phase),
            "phrase_length": int(L),
            "phrase_offset": int(o),
            "total_score": float(total),
            "component_scores": {
                "metrical_role_score": float(m2_phase_score),
                "bar_pattern_consistency": float(bpc),
                "phrase_boundary_contrast": float(pbc),
                "repeated_phrase_similarity": float(rep),
                "stability_curve_cleanliness": float(stab),
            },
        })

    candidates_out.sort(key=lambda c: c["total_score"], reverse=True)
    best = candidates_out[0] if candidates_out else None
    conf = (
        candidates_out[0]["total_score"] - candidates_out[1]["total_score"]
        if len(candidates_out) >= 2
        else 0.0
    )
    return {
        "method": "hybrid_score",
        "best": best,
        "candidates": candidates_out[:20],
        "confidence": float(conf),
        "bar_phase": int(bar_phase),
        "weights": {k: float(v) for k, v in weights.items()},
    }


# ---------------------------------------------------------------------------
# Method 6 — Optional madmom comparison
# ---------------------------------------------------------------------------

def madmom_downbeat_compare(
    beat_times_s: np.ndarray,
    audio_path: str,
    *,
    beats_per_bar: int = 4,
) -> dict[str, Any]:
    """
    If madmom ≥ 0.16 is installed, run DBNDownBeatTrackingProcessor and map
    downbeats to our beat grid. Returns skipped=True if madmom is absent.
    """
    try:
        from madmom.features.downbeats import (
            DBNDownBeatTrackingProcessor,
            RNNDownBeatProcessor,
        )
    except ImportError:
        return {
            "method": "madmom_downbeat",
            "skipped": True,
            "reason": "madmom not installed; pip install madmom to enable",
        }
    try:
        act = RNNDownBeatProcessor()(audio_path)
        beats = DBNDownBeatTrackingProcessor(beats_per_bar=beats_per_bar, fps=100)(act)
        downbeat_times = beats[beats[:, 1] == 1, 0]
        if len(downbeat_times) == 0 or len(beat_times_s) == 0:
            return {"method": "madmom_downbeat", "skipped": False, "phase": 0,
                    "n_downbeats": 0, "downbeat_times": []}
        phase_votes: dict[int, int] = {}
        for dt in downbeat_times:
            idx = int(np.argmin(np.abs(np.asarray(beat_times_s) - dt)))
            p = idx % beats_per_bar
            phase_votes[p] = phase_votes.get(p, 0) + 1
        best = max(phase_votes, key=phase_votes.__getitem__)
        return {
            "method": "madmom_downbeat",
            "skipped": False,
            "phase": int(best),
            "phase_votes": {str(k): int(v) for k, v in phase_votes.items()},
            "n_downbeats": int(len(downbeat_times)),
            "downbeat_times_sample": downbeat_times.tolist()[:12],
        }
    except Exception as exc:
        return {"method": "madmom_downbeat", "skipped": False, "error": str(exc), "phase": 0}
