import numpy as np
import pytest

from songviz.music_representation import (
    compare_ordered_recurrence,
    compute_local_contrasts,
    evaluate_representation,
    pool_frame_embeddings,
)


def _pool(vectors, *, frame_times=None, supports=None, beats=None):
    vectors = np.asarray(vectors, dtype=float)
    n = len(vectors)
    return pool_frame_embeddings(
        vectors, np.arange(n, dtype=float) + .5 if frame_times is None else frame_times,
        np.arange(n, dtype=float) if supports is None else supports[0],
        np.arange(1, n + 1, dtype=float) if supports is None else supports[1],
        np.arange(n + 1, dtype=float) if beats is None else beats,
    )


def test_pool_uses_frame_centres_and_aggregates_full_encoder_support():
    pooled = _pool([[1, 0], [3, 0], [9, 0]], frame_times=np.array([.1, .9, 1.0]),
                   supports=(np.array([-.2, .2, .7]), np.array([.3, 1.2, 1.4])),
                   beats=np.array([0., 1., 2.]))
    assert np.allclose(pooled["vectors"][0], [2, 0])
    assert np.allclose(pooled["vectors"][1], [9, 0])
    assert pooled["samples"][0]["encoder_support_start_s"] == -.2
    assert pooled["samples"][0]["encoder_support_end_s"] == 1.2


def test_empty_and_zero_norm_are_unknown_evidence_with_full_grid_edges():
    pooled = _pool([[0, 0]], frame_times=np.array([1.5]),
                   supports=(np.array([1.]), np.array([2.])), beats=np.arange(5.))
    assert pooled["samples"][0]["vector"] is None
    local = compute_local_contrasts(pooled, scales=(1,))
    samples = local["curves"][0]["samples"]
    assert samples[0] is samples[-1] is None
    assert samples[1]["cosine_distance"] is None
    assert samples[2]["cosine_distance"] is None
    partial_zero = _pool([[1, 0], [0, 0], [0, 1], [0, 1]])
    assert compute_local_contrasts(partial_zero, scales=(2,))["curves"][0]["samples"][2]["cosine_distance"] is None


def test_local_distance_handles_irregular_beats_and_retains_nominal_and_encoder_context():
    pooled = _pool([[1, 0], [1, 0], [0, 1], [0, 1]], frame_times=np.array([2.1, 2.5, 4., 6.]), beats=np.array([2., 2.2, 3., 5., 8.]),
                   supports=(np.array([1.7, 2.1, 2.8, 4.9]), np.array([2.3, 3.2, 5.3, 8.2])))
    sample = compute_local_contrasts(pooled, scales=(2,))["curves"][0]["samples"][2]
    assert sample["nominal_before"] == {"start_beat": 0, "end_beat": 2, "start_s": 2., "end_s": 3.}
    assert sample["nominal_after"]["end_s"] == 8.
    assert sample["encoder_support_start_s"] == 1.7
    assert sample["available_at_s"] == 8.2
    assert sample["cosine_distance"] == pytest.approx(1)


def test_ordered_recurrence_finds_aba_and_tracks_encoder_context_history_separately():
    a = np.eye(2)
    pooled = _pool(np.vstack([a, a[::-1], a]),
                   supports=(np.array([0., 1., 1.5, 3., 4., 5.]), np.array([5., 2.5, 3.5, 4.5, 5.5, 6.])))
    result = compare_ordered_recurrence(pooled, scales=(2,), stride_beats=2)["results"][0]
    pair = next(p for p in result["pairs"] if (p["a"], p["b"]) == (0, 2))
    assert pair["cosine_similarity"] == pytest.approx(1)
    final = result["context"][2]
    assert final["best_prior_span"] == 0
    assert final["historical_novelty"] == pytest.approx(0)
    # The nominally earlier span leaks its encoder context through target start.
    assert final["strict_encoder_context_history"]["best_prior_span"] is None


def test_reordered_sequence_is_not_a_match_and_all_nominal_pairs_are_retained():
    pooled = _pool(np.vstack([np.eye(3), np.eye(3)[::-1]]))
    result = compare_ordered_recurrence(pooled, scales=(3,), stride_beats=3)["results"][0]
    assert len(result["pairs"]) == 1
    assert result["pairs"][0]["cosine_similarity"] < 1


def test_future_overlap_and_context_overlap_are_excluded_with_earlier_tie_break():
    pooled = _pool(np.tile([1., 0.], (12, 1)),
                   supports=(np.arange(12.) - .5, np.arange(12.) + 1.5))
    result = compare_ordered_recurrence(pooled, scales=(4,), stride_beats=1)["results"][0]
    item = result["context"][8]
    assert item["previous_span"] == 4
    # All eligible spans tie, and earlier index is the deliberate resolution.
    assert item["best_prior_span"] == 0
    assert item["strict_encoder_context_history"]["best_prior_span"] == 0
    assert result["context"][4]["strict_encoder_context_history"]["best_prior_span"] is None
    assert all(result["spans"][p["a"]]["end_beat"] <= result["spans"][p["b"]]["start_beat"]
               for p in result["pairs"])


def test_input_validation_and_input_preservation():
    embeddings = np.array([[1., 0.], [0., 1.]])
    original = embeddings.copy()
    _pool(embeddings)
    assert np.array_equal(embeddings, original)
    with pytest.raises(ValueError, match="finite"):
        _pool([[np.nan, 0.]])
    with pytest.raises(ValueError, match="within"):
        _pool([[1, 0]], frame_times=np.array([.5]), supports=(np.array([.6]), np.array([1.])))
    with pytest.raises(ValueError, match="strictly increasing"):
        _pool([[1, 0]], beats=np.array([0., 0.]))


def test_evaluation_keeps_fixed_anchor_baseline_stems_and_midranks_without_tuning():
    local = {"curves": [{"scale_beats": h, "samples": [
        {"anchor_index": i, "anchor_s": float(i), "available_at_s": float(i + 1),
         "cosine_distance": float(i) if i < 3 else None} for i in range(5)]} for h in (2, 4, 8)]}
    base_sample = {"anchor_index": 2, "anchor_s": 2., "stems": {"vocals": {
        "left": {"mean_rms": 1}, "right": {"mean_rms": .5},
        "changes": {"signed_rms_difference": -.5}}}}
    role_eval = {"cases": [{"id": f"case-{i}", "fixed_anchor_index": 2, "fixed_anchor_s": 2.,
                              "per_scale": [{"scale_beats": h, "anchor_samples": [
                                  {**base_sample, "anchor_index": j} for j in range(5)]} for h in (2, 4, 8)]}
                             for i in range(4)]}
    output = evaluate_representation({"local": local, "recurrence": {"results": []}}, [], {}, role_eval,
                                     {"layers": []}, {"note": "frozen"})
    joined = output["guided_cases"][0]["scales"][0]["anchors"][2]
    assert joined["muq_cosine_distance"] == 2
    assert joined["muq_whole_curve_midrank_percentile"] == pytest.approx(83.333333)
    assert joined["baseline_stems"] == base_sample["stems"]
    assert output["raw_feedback"] == {"note": "frozen"}


def test_evaluation_joins_every_separate_same_motif_run_and_merges_adjacent_identity():
    spans = [{"start_s": i * 10., "end_s": (i + 1) * 10.} for i in range(4)]
    pairs = [{"a": 0, "b": 2, "cosine_similarity": .9, "cosine_distance": .1},
             {"a": 0, "b": 3, "cosine_similarity": .8, "cosine_distance": .2},
             {"a": 1, "b": 2, "cosine_similarity": .7, "cosine_distance": .3},
             {"a": 1, "b": 3, "cosine_similarity": .6, "cosine_distance": .4}]
    recurrence = {"results": [{"scale_beats": 16, "spans": spans, "pairs": pairs}]}
    # The first two adjacent spans are one occurrence, not a return pair.
    reference = {"layers": [{"id": "main", "spans": [
        {"id": "a", "start_s": 0., "end_s": 10., "motif": "chorus", "identity_id": "c"},
        {"id": "b", "start_s": 10., "end_s": 20., "motif": "chorus", "identity_id": "c"},
        {"id": "x", "start_s": 20., "end_s": 30., "motif": "", "identity_id": None},
        {"id": "d", "start_s": 30., "end_s": 40., "motif": "chorus", "identity_id": "c"},
    ]}]}
    role_eval = {"cases": [{"id": str(i), "per_scale": [
        {"scale_beats": h, "anchor_samples": [None] * 5} for h in (2, 4, 8)]} for i in range(4)]}
    local = {"curves": [{"scale_beats": h, "samples": [None] * 4} for h in (2, 4, 8)]}
    baseline = [{"scale_beats": 16, "pairs": [{"a": 0, "b": 3, "pattern_similarity": .5, "arrangement_similarity": .4}]}]
    output = evaluate_representation({"local": local, "recurrence": recurrence}, baseline, {}, role_eval, reference, {})
    assert len(output["positive_return_runs"]) == 2
    assert output["positive_return_runs"][0]["span_ids"] == ["a", "b"]
    assert {(row["a"], row["b"]) for row in output["positive_return_pairs"]} == {(0, 3), (1, 3)}
    by_pair = {(row["a"], row["b"]): row for row in output["positive_return_pairs"]}
    assert by_pair[(0, 3)]["baseline_pattern_similarity"] == .5
    assert by_pair[(1, 3)]["baseline_pattern_similarity"] is None
    assert output["positive_return_summary"][0]["eligible_pair_count"] == 2
