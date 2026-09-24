"""Beat-aligned evidence from learned music-frame representations.

This module only compares numeric representations.  It deliberately does not
name sections, infer musical meaning, or calibrate a change threshold.
"""
from __future__ import annotations

from numbers import Integral

import numpy as np


def _times(value: object, name: str, *, increasing: bool, minimum: int = 0) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if array.ndim != 1 or array.size < minimum or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    if increasing and np.any(np.diff(array) <= 0):
        raise ValueError(f"{name} must be strictly increasing")
    return array


def _positive_ints(values: tuple[int, ...], name: str) -> tuple[int, ...]:
    if (not isinstance(values, tuple) or not values or
            any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) or v < 1
                for v in values) or len(set(values)) != len(values)):
        raise ValueError(f"{name} must be a nonempty tuple of distinct positive integers")
    return tuple(int(v) for v in values)


def pool_frame_embeddings(
    embeddings: np.ndarray,
    frame_times_s: np.ndarray,
    support_start_s: np.ndarray,
    support_end_s: np.ndarray,
    beat_times_s: np.ndarray,
) -> dict:
    """Mean frame-centre embeddings in complete ``[beat_start, beat_end)`` bins.

    The returned matrix retains one row per beat interval.  Empty intervals are
    represented by an all-NaN row and ``valid=False``; their support is also
    NaN.  A zero vector is retained here but becomes unavailable comparison
    evidence, since it has no cosine direction.
    """
    try:
        frames = np.asarray(embeddings, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("embeddings must be a finite [frames, dimensions] array") from exc
    if frames.ndim != 2 or frames.shape[1] < 1 or not np.isfinite(frames).all():
        raise ValueError("embeddings must be a finite [frames, dimensions] array")
    frame_times = _times(frame_times_s, "frame_times_s", increasing=True)
    starts = _times(support_start_s, "support_start_s", increasing=False)
    ends = _times(support_end_s, "support_end_s", increasing=False)
    beats = _times(beat_times_s, "beat_times_s", increasing=True, minimum=2)
    if any(a.size != frames.shape[0] for a in (frame_times, starts, ends)):
        raise ValueError("frame times and supports must have one entry per embedding")
    if np.any(starts > frame_times) or np.any(frame_times > ends):
        raise ValueError("each frame time must lie within its encoder support")

    n_beats = beats.size - 1
    vectors = np.full((n_beats, frames.shape[1]), np.nan, dtype=float)
    valid = np.zeros(n_beats, dtype=bool)
    pooled_starts = np.full(n_beats, np.nan, dtype=float)
    pooled_ends = np.full(n_beats, np.nan, dtype=float)
    # searchsorted implements the documented half-open endpoint convention.
    bin_index = np.searchsorted(beats, frame_times, side="right") - 1
    for beat in range(n_beats):
        selected = bin_index == beat
        if not np.any(selected):
            continue
        vectors[beat] = frames[selected].mean(axis=0)
        valid[beat] = True
        pooled_starts[beat] = starts[selected].min()
        pooled_ends[beat] = ends[selected].max()
    samples = [
        {"beat": i, "nominal_start_s": float(beats[i]), "nominal_end_s": float(beats[i + 1]),
         "frame_count": int(np.count_nonzero(bin_index == i)),
         "encoder_support_start_s": _nullable(pooled_starts[i]),
         "encoder_support_end_s": _nullable(pooled_ends[i]),
         "vector": (vectors[i].tolist() if valid[i] else None)}
        for i in range(n_beats)
    ]
    return {"beat_times_s": beats.copy(), "vectors": vectors, "valid": valid,
            "encoder_support_start_s": pooled_starts,
            "encoder_support_end_s": pooled_ends, "samples": samples,
            "method": "frame_center_mean_beat_pooling_v1"}


def _nullable(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _pooled(value: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(value, dict):
        raise ValueError("pooled must be the result of pool_frame_embeddings")
    try:
        beats = _times(value["beat_times_s"], "pooled beat_times_s", increasing=True, minimum=2)
        vectors = np.asarray(value["vectors"], dtype=float)
        valid = np.asarray(value["valid"], dtype=bool)
        starts = np.asarray(value["encoder_support_start_s"], dtype=float)
        ends = np.asarray(value["encoder_support_end_s"], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("pooled must be the result of pool_frame_embeddings") from exc
    n = beats.size - 1
    if vectors.ndim != 2 or vectors.shape[0] != n or vectors.shape[1] < 1 or valid.shape != (n,) or starts.shape != (n,) or ends.shape != (n,):
        raise ValueError("pooled arrays have incompatible shapes")
    if np.any(valid & (~np.isfinite(vectors).all(axis=1) | ~np.isfinite(starts) | ~np.isfinite(ends) | (starts > ends))):
        raise ValueError("valid pooled rows require finite vectors and supports")
    return beats, vectors, valid, starts, ends


def _mean_vector(vectors: np.ndarray, valid: np.ndarray, start: int, end: int) -> np.ndarray | None:
    if not valid[start:end].all():
        return None
    rows = vectors[start:end]
    if np.any(np.linalg.norm(rows, axis=1) <= 1e-12):
        return None
    vector = rows.mean(axis=0)
    return vector if np.linalg.norm(vector) > 1e-12 else None


def _support(starts: np.ndarray, ends: np.ndarray, valid: np.ndarray, start: int, end: int) -> tuple[float | None, float | None]:
    if not valid[start:end].all():
        return None, None
    return float(starts[start:end].min()), float(ends[start:end].max())


def _cosine(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    score = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return float(np.clip(score, -1.0, 1.0))


def compute_local_contrasts(pooled: dict, *, scales: tuple[int, ...] = (2, 4, 8)) -> dict:
    """Return every supported symmetric beat-boundary contrast, without scores thresholds."""
    scales = _positive_ints(scales, "scales")
    beats, vectors, valid, starts, ends = _pooled(pooled)
    n = valid.size
    curves = []
    for h in scales:
        samples: list[dict | None] = [None] * (n + 1)
        for k in range(h, n - h + 1):
            left = _mean_vector(vectors, valid, k - h, k)
            right = _mean_vector(vectors, valid, k, k + h)
            support_start, support_end = _support(starts, ends, valid, k - h, k + h)
            cosine = _cosine(left, right)
            samples[k] = {
                "anchor_index": k, "anchor_s": float(beats[k]), "scale_beats": h,
                "nominal_before": {"start_beat": k - h, "end_beat": k,
                                   "start_s": float(beats[k - h]), "end_s": float(beats[k])},
                "nominal_after": {"start_beat": k, "end_beat": k + h,
                                  "start_s": float(beats[k]), "end_s": float(beats[k + h])},
                "encoder_support_start_s": support_start,
                "encoder_support_end_s": support_end, "available_at_s": support_end,
                "cosine_similarity": cosine,
                "cosine_distance": (None if cosine is None else float(1 - cosine)),
            }
        curves.append({"scale_beats": h, "samples": samples})
    return {"method": "mean_beat_vector_cosine_distance_v1", "times_s": beats.tolist(),
            "curves": curves,
            "limitations": ["Null evidence means an empty beat pool, a zero-norm summary, or unsupported edge; it is not zero change.",
                            "Cosine distance is 1 - clipped cosine and ranges from 0 to 2; no threshold or calibrated interpretation is supplied.",
                            "available_at_s includes complete encoder support, which can extend beyond nominal beat windows."]}


def _span_vector(vectors: np.ndarray, valid: np.ndarray, start: int, end: int) -> np.ndarray | None:
    if not valid[start:end].all():
        return None
    rows = vectors[start:end]
    norms = np.linalg.norm(rows, axis=1)
    if np.any(norms <= 1e-12):
        return None
    return (rows / norms[:, None]).ravel()


def compare_ordered_recurrence(
    pooled: dict, *, scales: tuple[int, ...] = (16, 32), stride_beats: int = 4,
) -> dict:
    """Compare baseline-aligned ordered windows, retaining all nominal pairs.

    Nominal history is nonoverlapping in beat time.  Strict history is the
    narrower subset whose full encoder supports do not overlap either.
    """
    scales = _positive_ints(scales, "scales")
    if (isinstance(stride_beats, (bool, np.bool_)) or
            not isinstance(stride_beats, Integral) or stride_beats < 1):
        raise ValueError("stride_beats must be a positive integer")
    beats, vectors, valid, starts, ends = _pooled(pooled)
    n = valid.size
    results = []
    for scale in scales:
        offsets = list(range(0, n - scale + 1, int(stride_beats)))
        spans = []
        span_vectors: list[np.ndarray | None] = []
        for start in offsets:
            end = start + scale
            support_start, support_end = _support(starts, ends, valid, start, end)
            spans.append({"start_beat": start, "end_beat": end,
                          "start_s": float(beats[start]), "end_s": float(beats[end]),
                          "encoder_support_start_s": support_start,
                          "encoder_support_end_s": support_end, "available_at_s": support_end})
            span_vectors.append(_span_vector(vectors, valid, start, end))
        pairs, lookup = [], {}
        for a in range(len(spans)):
            for b in range(a + 1, len(spans)):
                if spans[a]["end_beat"] > spans[b]["start_beat"]:
                    continue
                cosine = _cosine(span_vectors[a], span_vectors[b])
                record = {"a": a, "b": b, "cosine_similarity": cosine,
                          "cosine_distance": None if cosine is None else float(1 - cosine)}
                pairs.append(record)
                lookup[(a, b)] = record
        context = []
        for target, span in enumerate(spans):
            nominal = [i for i, prior in enumerate(spans[:target])
                       if prior["end_beat"] <= span["start_beat"]]
            strict = [i for i in nominal
                      if spans[i]["encoder_support_end_s"] is not None and span["encoder_support_start_s"] is not None
                      and spans[i]["encoder_support_end_s"] <= span["encoder_support_start_s"]]
            previous = max(nominal, default=None)
            local = lookup.get((previous, target)) if previous is not None else None
            def best(indices: list[int]) -> tuple[int | None, float | None]:
                candidates = [(i, lookup[(i, target)]["cosine_similarity"]) for i in indices
                              if lookup[(i, target)]["cosine_similarity"] is not None]
                if not candidates:
                    return None, None
                # Earlier index resolves exactly equal similarities deterministically.
                winner, similarity = max(candidates, key=lambda item: (item[1], -item[0]))
                return winner, float(1 - similarity)
            nominal_best, nominal_novelty = best(nominal)
            strict_best, strict_novelty = best(strict)
            context.append({
                "span": target, "available_at_s": span["available_at_s"],
                "previous_span": previous,
                "local_cosine_distance": None if local is None else local["cosine_distance"],
                "best_prior_span": nominal_best, "historical_novelty": nominal_novelty,
                "comparable_prior_count": sum(lookup[(i, target)]["cosine_similarity"] is not None for i in nominal),
                "nominal_history": {"best_prior_span": nominal_best, "historical_novelty": nominal_novelty,
                                    "comparable_prior_count": sum(lookup[(i, target)]["cosine_similarity"] is not None for i in nominal)},
                "strict_encoder_context_history": {"best_prior_span": strict_best, "historical_novelty": strict_novelty,
                                                    "comparable_prior_count": sum(lookup[(i, target)]["cosine_similarity"] is not None for i in strict)},
            })
        results.append({"scale_beats": scale, "stride_beats": int(stride_beats), "spans": spans,
                        "pairs": pairs, "context": context})
    return {"method": "ordered_l2_normalized_beat_embedding_flatten_cosine_v1", "results": results,
            "limitations": ["All nominally nonoverlapping pairs are retained; a null cosine is unavailable evidence, not dissimilarity.",
                            "Nominal history can have encoder-context leakage. strict_encoder_context_history excludes it and may therefore have less history.",
                            "No labels, per-song scaling, threshold, or semantic claim is used."]}


def _by_scale(rows: object) -> dict[int, dict]:
    if not isinstance(rows, list):
        return {}
    return {row.get("scale_beats"): row for row in rows if isinstance(row, dict) and isinstance(row.get("scale_beats"), int)}


def _recurrence_results(value: object) -> dict[int, dict]:
    """Accept the numeric result or the frozen recurrence's list-of-scales form."""
    if isinstance(value, dict):
        value = value.get("recurrence", value.get("results", value))
        if isinstance(value, dict):
            value = value.get("results")
    return _by_scale(value)


def _midrank_percentile(values: list[float], value: float | None) -> float | None:
    if value is None or not values:
        return None
    less = sum(item < value for item in values)
    equal = sum(item == value for item in values)
    return float(100 * (less + .5 * equal) / len(values))


def _identity_runs(reference: dict) -> list[dict]:
    """Merge adjacent annotated spans with the same explicit identity/motif."""
    runs = []
    for layer in reference.get("layers", []) if isinstance(reference, dict) else []:
        if not isinstance(layer, dict) or not isinstance(layer.get("spans"), list):
            continue
        open_run = None
        for position, span in enumerate(layer["spans"]):
            motif = span.get("motif") if isinstance(span, dict) else None
            identity = span.get("identity_id") if isinstance(span, dict) else None
            if not isinstance(motif, str) or not motif:
                open_run = None
                continue
            try:
                start, end = float(span["start_s"]), float(span["end_s"])
            except (KeyError, TypeError, ValueError):
                open_run = None
                continue
            # Missing identity must not erase an explicit positive motif.  It
            # also cannot justify merging adjacent annotations into one run.
            key = (layer.get("id"), motif, identity if isinstance(identity, str) and identity else position)
            if (open_run is not None and open_run["key"] == key and
                    open_run["end_s"] == start):
                open_run["end_s"] = end
                open_run["span_ids"].append(span.get("id"))
            else:
                open_run = {"key": key, "layer_id": layer.get("id"), "motif": motif,
                            "identity_id": identity, "start_s": start, "end_s": end,
                            "span_ids": [span.get("id")]}
                runs.append(open_run)
    return runs


def evaluate_representation(
    result: dict, baseline_recurrence: object, role_context: object, role_evaluation: dict,
    reference: dict, raw_feedback: object, *, shifted_result: dict | None = None,
) -> dict:
    """Join already-computed evidence to frozen labels without changing extraction.

    Values remain per-scale and per-pair.  This deliberately has no aggregate
    score, negative examples, threshold, or section/role inference.
    """
    if not isinstance(result, dict) or not isinstance(role_evaluation, dict) or not isinstance(reference, dict):
        raise ValueError("evaluation requires result and frozen mapping records")
    local = result.get("local")
    if not isinstance(local, dict):
        raise ValueError("result lacks local representation evidence")
    muq_local = _by_scale(local.get("curves"))
    cases = role_evaluation.get("cases")
    if not isinstance(cases, list) or len(cases) != 4:
        raise ValueError("role_evaluation must provide exactly four fixed cases")
    guided_cases = []
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str) or not isinstance(case.get("per_scale"), list):
            raise ValueError("malformed fixed role-evaluation case")
        scales = []
        for base_curve in case["per_scale"]:
            scale = base_curve.get("scale_beats") if isinstance(base_curve, dict) else None
            anchors = base_curve.get("anchor_samples") if isinstance(base_curve, dict) else None
            curve = muq_local.get(scale)
            if curve is None or not isinstance(anchors, list) or len(anchors) != 5:
                raise ValueError("fixed cases require five anchors at every matching scale")
            samples = curve.get("samples")
            rows = []
            known = [sample["cosine_distance"] for sample in samples
                     if isinstance(sample, dict) and sample.get("cosine_distance") is not None]
            for offset, baseline_sample in zip((-2, -1, 0, 1, 2), anchors):
                index = baseline_sample.get("anchor_index") if isinstance(baseline_sample, dict) else None
                muq_sample = samples[index] if isinstance(index, int) and 0 <= index < len(samples) else None
                value = muq_sample.get("cosine_distance") if isinstance(muq_sample, dict) else None
                rows.append({"offset_beats": offset, "anchor_index": index,
                             "anchor_s": baseline_sample.get("anchor_s") if isinstance(baseline_sample, dict) else None,
                             "muq_cosine_distance": value,
                             "muq_whole_curve_midrank_percentile": _midrank_percentile(known, value),
                             "muq_available_at_s": muq_sample.get("available_at_s") if isinstance(muq_sample, dict) else None,
                             # Retain exact frozen stem sides and signed changes; units differ from MuQ.
                             "baseline_stems": baseline_sample.get("stems") if isinstance(baseline_sample, dict) else None})
            scales.append({"scale_beats": scale, "whole_curve_valid_count": len(known), "anchors": rows})
        guided_cases.append({"id": case["id"], "fixed_anchor_index": case.get("fixed_anchor_index"),
                             "fixed_anchor_s": case.get("fixed_anchor_s"), "scales": scales})

    muq_recurrence = _recurrence_results(result)
    baseline = _recurrence_results(baseline_recurrence)
    shifted = _recurrence_results(shifted_result) if shifted_result is not None else {}
    runs = _identity_runs(reference)
    return_rows, summary = [], []
    for scale, representation in sorted(muq_recurrence.items()):
        spans = representation.get("spans", [])
        pair_lookup = {(p.get("a"), p.get("b")): p for p in representation.get("pairs", []) if isinstance(p, dict)}
        baseline_pairs = {(p.get("a"), p.get("b")): p for p in baseline.get(scale, {}).get("pairs", []) if isinstance(p, dict)}
        shifted_pairs = {(p.get("a"), p.get("b")): p for p in shifted.get(scale, {}).get("pairs", []) if isinstance(p, dict)}
        for left_number, left_run in enumerate(runs):
            for right_run in runs[left_number + 1:]:
                if left_run["layer_id"] != right_run["layer_id"] or left_run["motif"] != right_run["motif"]:
                    continue
                a_windows = [i for i, span in enumerate(spans) if span["start_s"] >= left_run["start_s"] and span["end_s"] <= left_run["end_s"]]
                b_windows = [i for i, span in enumerate(spans) if span["start_s"] >= right_run["start_s"] and span["end_s"] <= right_run["end_s"]]
                start_count = len(return_rows)
                for a in a_windows:
                    for b in b_windows:
                        pair, frozen, shifted_pair = pair_lookup.get((a, b)), baseline_pairs.get((a, b)), shifted_pairs.get((a, b))
                        return_rows.append({"scale_beats": scale, "layer_id": left_run["layer_id"], "motif": left_run["motif"],
                                            "a_occurrence": {key: left_run[key] for key in ("identity_id", "start_s", "end_s", "span_ids")},
                                            "b_occurrence": {key: right_run[key] for key in ("identity_id", "start_s", "end_s", "span_ids")},
                                            "a": a, "b": b,
                                            "muq_cosine_similarity": pair.get("cosine_similarity") if pair else None,
                                            "muq_cosine_distance": pair.get("cosine_distance") if pair else None,
                                            "baseline_pattern_similarity": frozen.get("pattern_similarity") if frozen else None,
                                            "baseline_arrangement_similarity": frozen.get("arrangement_similarity") if frozen else None,
                                            "shifted_muq_cosine_similarity": shifted_pair.get("cosine_similarity") if shifted_pair else None})
                rows = return_rows[start_count:]
                summary.append({"scale_beats": scale, "layer_id": left_run["layer_id"], "motif": left_run["motif"],
                                "a_run_start_s": left_run["start_s"], "b_run_start_s": right_run["start_s"],
                                "eligible_pair_count": len(rows),
                                "muq_supported_pair_count": sum(row["muq_cosine_similarity"] is not None for row in rows),
                                "baseline_supported_pair_count": sum(row["baseline_pattern_similarity"] is not None for row in rows),
                                "shifted_supported_pair_count": sum(row["shifted_muq_cosine_similarity"] is not None for row in rows)})
    return {"method": "frozen_fixed_anchor_and_explicit_positive_return_join_v1", "guided_cases": guided_cases,
            "positive_return_runs": [{key: value for key, value in run.items() if key != "key"} for run in runs],
            "positive_return_pairs": return_rows, "positive_return_summary": summary,
            "raw_feedback": raw_feedback,
            "limitations": ["Fixed anchors and explicit motif returns are evaluation inputs only; they do not affect representation extraction.",
                            "Different/unlabeled motifs are not negative examples. Baseline stem descriptors and MuQ mix embeddings have different support and units.",
                            "Midrank percentiles describe each full unscaled MuQ curve; they are not calibrated scores or a choice of scale."]}
