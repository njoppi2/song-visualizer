"""Role-independent, ordered phrase comparisons for diagnostic review.

These are acoustic candidates, not section identities or calibrated probabilities.
Inputs are beat-interval log-spectral features and RMS, not role/section labels.
"""
from __future__ import annotations

from numbers import Integral

import numpy as np


def compare_phrases(
    features: dict[str, np.ndarray],
    energy: dict[str, np.ndarray],
    beat_times: np.ndarray,
    *,
    scale_beats: int = 16,
    stride_beats: int = 4,
) -> dict:
    """Compare complete, ordered windows; never extrapolate partial phrases.

    Equal-weight audible stems contribute cosine shape similarity multiplied by
    their relative mean-RMS agreement. Shared silence contributes no evidence;
    an entry/exit in one side contributes zero. The additive pattern and
    arrangement fields are heuristic diagnostics, not recurrence identities.
    Beat phase is an input hypothesis.
    """
    if (isinstance(scale_beats, (bool, np.bool_)) or
            not isinstance(scale_beats, Integral) or
            isinstance(stride_beats, (bool, np.bool_)) or
            not isinstance(stride_beats, Integral)):
        raise ValueError("Phrase scale and stride must be integers")
    try:
        bt = np.asarray(beat_times, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Expected finite increasing beat endpoints") from exc
    if bt.ndim != 1 or bt.size < 2 or not np.isfinite(bt).all() or np.any(np.diff(bt) <= 0):
        raise ValueError("Expected finite increasing beat endpoints")
    if scale_beats < 1 or stride_beats < 1:
        raise ValueError("Phrase scale and stride must be positive")
    if not features or set(features) != set(energy):
        raise ValueError("Matching feature and energy stems are required")
    n = bt.size - 1
    starts = list(range(0, n - scale_beats + 1, stride_beats))
    spans = [{"start_beat": i, "end_beat": i + scale_beats,
              "start_s": float(bt[i]), "end_s": float(bt[i + scale_beats])} for i in starts]
    total = np.zeros((len(starts), len(starts)))
    count = np.zeros_like(total)
    components = {}
    evidence_components = {}
    for name in sorted(features):
        try:
            f = np.asarray(features[name], dtype=float)
            e = np.asarray(energy[name], dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("Each stem needs finite interval features and nonnegative RMS") from exc
        if f.ndim != 2 or f.shape[1] != n or e.shape != (n,) or not np.isfinite(f).all() or not np.isfinite(e).all() or np.any(e < 0):
            raise ValueError("Each stem needs finite interval features and nonnegative RMS")
        if not starts:
            components[name] = []
            continue
        vectors = np.array([f[:, s:s + scale_beats].ravel() for s in starts])
        norms = np.linalg.norm(vectors, axis=1)
        unit = vectors / np.maximum(norms[:, None], 1e-12)
        # This bounded cosine is unrescaled and has no RMS multiplication.
        # It is also the exact legacy shape matrix used by the combined score.
        shape = np.clip(unit @ unit.T, 0, 1)
        levels = np.array([e[s:s + scale_beats].mean() for s in starts])
        # Relative floor is only a diagnostic audibility gate, not calibrated SPL.
        floor = max(float(e.max()) * 0.02, 1e-8)
        active = (levels > floor) & (norms > 1e-8)
        either = active[:, None] | active[None, :]
        both = active[:, None] & active[None, :]
        level_ratio = np.minimum(levels[:, None], levels[None, :]) / np.maximum(np.maximum(levels[:, None], levels[None, :]), 1e-12)
        score = np.where(both, shape * level_ratio, 0)
        total += score
        count += either
        components[name] = [[float(score[i, j]) if either[i, j] else None
                             for j in range(len(starts))] for i in range(len(starts))]
        evidence_components[name] = {
            "shape": shape,
            "level_ratio": level_ratio,
            "active": active,
        }
    combined = total / np.maximum(count, 1)
    pairs = []
    pairs_by_index = {}
    for i, a in enumerate(spans):
        for j in range(i + 1, len(spans)):
            b = spans[j]
            if a['end_beat'] > b['start_beat'] or count[i, j] == 0:
                continue
            pattern_values = []
            arrangement_values = []
            stem_evidence = {}
            shared_active_stems = []
            for name in sorted(evidence_components):
                evidence = evidence_components[name]
                a_active = bool(evidence["active"][i])
                b_active = bool(evidence["active"][j])
                if a_active and b_active:
                    pattern = float(evidence["shape"][i, j])
                    level = float(evidence["level_ratio"][i, j])
                    activity = "both"
                    pattern_values.append(pattern)
                    arrangement_values.append(level)
                    shared_active_stems.append(name)
                elif a_active:
                    pattern = None
                    level = 0.0
                    activity = "a_only"
                    arrangement_values.append(level)
                elif b_active:
                    pattern = None
                    level = 0.0
                    activity = "b_only"
                    arrangement_values.append(level)
                else:
                    pattern = None
                    level = None
                    activity = "neither"
                stem_evidence[name] = {
                    "pattern_similarity": pattern,
                    "level_similarity": level,
                    "activity": activity,
                }
            pair = {
                "a": i,
                "b": j,
                # These two fields retain their historical score semantics.
                "similarity": float(combined[i, j]),
                "stem_similarities": {name: values[i][j] for name, values in components.items()},
                "pattern_similarity": (float(np.mean(pattern_values))
                                       if pattern_values else None),
                "arrangement_similarity": (float(np.mean(arrangement_values))
                                            if arrangement_values else None),
                "shared_active_stems": shared_active_stems,
                "stem_evidence": stem_evidence,
            }
            pairs.append(pair)
            pairs_by_index[(i, j)] = pair

    context = []
    for j, span in enumerate(spans):
        # This is structural context, so it is recorded even when silence means
        # the immediately preceding comparable pair has no acoustic evidence.
        prior_spans = [i for i, prior in enumerate(spans[:j])
                       if prior["end_beat"] <= span["start_beat"]]
        previous_span = (max(prior_spans,
                             key=lambda i: (spans[i]["end_beat"], i))
                         if prior_spans else None)
        local_pair = (pairs_by_index.get((previous_span, j))
                      if previous_span is not None else None)
        comparable = [
            (i, pairs_by_index[(i, j)])
            for i in prior_spans
            if (i, j) in pairs_by_index
            and pairs_by_index[(i, j)]["pattern_similarity"] is not None
        ]
        if comparable:
            best_prior_span, best_pair = max(
                comparable,
                key=lambda item: (item[1]["pattern_similarity"], item[0]),
            )
            historical_novelty = 1 - best_pair["pattern_similarity"]
        else:
            best_prior_span = None
            historical_novelty = None
        context.append({
            "span": j,
            "available_at_s": span["end_s"],
            "previous_span": previous_span,
            "local_pattern_change": (1 - local_pair["pattern_similarity"]
                                     if local_pair and local_pair["pattern_similarity"] is not None else None),
            "local_arrangement_change": (1 - local_pair["arrangement_similarity"]
                                         if local_pair and local_pair["arrangement_similarity"] is not None else None),
            "best_prior_span": best_prior_span,
            "historical_pattern_novelty": historical_novelty,
            "comparable_prior_count": len(comparable),
        })
    return {"scale_beats": scale_beats, "stride_beats": stride_beats,
            "spans": spans, "pairs": sorted(pairs, key=lambda p: (-p['similarity'], p['a'], p['b'])),
            "method": "ordered_log_cqt_cosine_times_rms_ratio_equal_audible_stems_v1",
            "evidence_method": "heuristic_bounded_log_cqt_cosine_and_mean_rms_ratio_diagnostics_v1",
            "context": context,
            "limitations": "Heuristic diagnostic evidence, not calibrated. Limited log-spectral features and log scaling can remain gain-sensitive; no verified bar phase. These phrase signals are not section labels or musical surprise. No activation or detection threshold tuning; shared silence is not recurrence evidence. The offline whole-song audibility floor uses a future maximum even though comparison context is strictly past-only; context scores become available at the target window end, not its start."}
