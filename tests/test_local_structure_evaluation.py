from copy import deepcopy

import pytest

from songviz.local_structure_evaluation import evaluate_local_structure


def reference():
    def span(span_id, start_s, end_s, identity_id, variation=None, transition=None):
        return {"id": span_id, "start_s": start_s, "end_s": end_s, "label": span_id,
                "certainty": "clear", "identity_id": identity_id, "variation": variation,
                "transition": transition}

    return {"source": {"duration_s": 60, "song_title": "test"}, "layers": [
        {"id": "detail", "name": "detail", "spans": [
            span("a", 0, 10, "A", "low"),
            span("b", 10, 20, "A", "high"),
            span("c", 20, 30, "B", transition=True),
            span("unknown", 30, 60, None),
        ]},
        {"id": "independent", "name": "independent", "spans": [
            span("i0", 2, 23, None), span("i1", 23, 60, None),
        ]},
    ]}


def predictions():
    return {"changes": [{"id": "change-1", "time_s": 10.5}, {"id": "change-2", "time_s": 55}],
            "transitions": [{"id": "transition-1", "start_s": 21, "end_s": 29},
                            {"id": "transition-miss", "start_s": 40, "end_s": 45}]}


def recurrence():
    return [{"scale_beats": 16, "stride_beats": 4,
             "spans": [{"start_s": 0, "end_s": 10}, {"start_s": 12, "end_s": 22},
                       {"start_s": 30, "end_s": 50}],
             "context": [
                 {"span": 0, "best_prior_span": None, "historical_pattern_novelty": None,
                  "local_pattern_change": None, "local_arrangement_change": None, "comparable_prior_count": 0},
                 {"span": 1, "best_prior_span": 0, "historical_pattern_novelty": .2,
                  "local_pattern_change": .3, "local_arrangement_change": .4, "comparable_prior_count": 1},
                 {"span": 2, "best_prior_span": 1, "historical_pattern_novelty": .5,
                  "local_pattern_change": .6, "local_arrangement_change": .7, "comparable_prior_count": 2},
             ]}]


def test_boundary_identity_variation_and_unknown_are_separate_without_hierarchy():
    result = evaluate_local_structure(reference(), predictions(), [], [])
    detail, independent = result["layers"]
    rows = detail["boundary_rows"]
    assert [(row["identity_relation"], row["variation_change"]) for row in rows] == [
        ("same_explicit_group", True), ("different_named_groups", None), ("unknown", None)]
    assert independent["boundary_rows"][0]["time_s"] == 23
    assert result["counts"] == {"changes": {"count": 2, "density_per_minute": 2},
                                "transitions": {"count": 2, "density_per_minute": 2}}


def test_transition_iou_compares_two_edges_and_misses_stay_null():
    result = evaluate_local_structure(reference(), predictions(), [], [])
    row = result["layers"][0]["transition_rows"][0]
    match = row["predicted_best_overlap"]
    assert match == {"id": "transition-1", "start_s": 21, "end_s": 29,
                     "iou": pytest.approx(.8), "intersection_s": 8,
                     "start_error_s": 1, "end_error_s": -1}
    reverse = result["layers"][0]["predicted_transition_rows"]
    assert reverse[0]["reference_best_overlap"]["span_id"] == "c"
    assert reverse[1]["reference_best_overlap"] is None
    missed_reference = evaluate_local_structure(
        reference(), {"changes": [], "transitions": [{"id": "elsewhere", "start_s": 40, "end_s": 45}]}, [], [],
    )
    assert missed_reference["layers"][0]["transition_rows"][0]["predicted_best_overlap"] is None


def test_every_prediction_is_listed_so_unannotated_events_remain_visible():
    result = evaluate_local_structure(reference(), predictions(), [{"start_s": 0, "end_s": 60}], [])
    layer = result["layers"][0]
    assert [row["id"] for row in layer["predicted_change_rows"]] == ["change-1", "change-2"]
    assert [row["id"] for row in layer["predicted_transition_rows"]] == ["transition-1", "transition-miss"]
    assert layer["predicted_change_rows"][1]["nearest_annotated_boundary"]["time_s"] == 30
    boundary = layer["boundary_rows"][0]
    assert boundary["nearest_change"] == {"id": "change-1", "time_s": 10.5, "delta_s": .5}
    assert boundary["nearest_legacy"] is None


def test_recurrence_uses_first_full_post_change_window_and_keeps_nulls_near_end():
    result = evaluate_local_structure(reference(), predictions(), [], recurrence())
    first, unavailable = result["recurrence_context"]
    assert first["event_id"] == "change-1"
    assert first["window"] == {"start_s": 12, "end_s": 22}
    assert first["offset_from_change_s"] == 1.5
    assert first["available_at_s"] == 22
    assert first["best_prior_span"] == {"span_index": 0, "start_s": 0, "end_s": 10}
    assert first["historical_pattern_novelty"] == .2
    assert unavailable["event_id"] == "change-2"
    assert unavailable["window"] is None
    assert unavailable["historical_pattern_novelty"] is None
    assert unavailable["local_pattern_change"] is None
    assert unavailable["comparable_prior_count"] is None


def test_recurrence_prior_is_checked_against_target_window_not_change_time():
    # The best prior ends after the change (10.5), but before target start (12):
    # this remains valid phrase context and is described in the limitation.
    history = recurrence()
    history[0]["spans"][0]["end_s"] = 11
    result = evaluate_local_structure(reference(), predictions(), [], history)
    assert result["recurrence_context"][0]["best_prior_span"]["end_s"] == 11


@pytest.mark.parametrize(
    "best_prior, windows, message",
    [
        (1, [{"start_s": 0, "end_s": 10}, {"start_s": 12, "end_s": 22}], "must precede"),
        (0, [{"start_s": 0, "end_s": 13}, {"start_s": 12, "end_s": 22}], "overlaps its target"),
    ],
)
def test_recurrence_rejects_future_or_overlapping_best_prior(best_prior, windows, message):
    history = recurrence()
    history[0]["spans"] = windows
    history[0]["context"] = [{"span": 1, "best_prior_span": best_prior,
                              "historical_pattern_novelty": .2, "local_pattern_change": .3,
                              "local_arrangement_change": .4, "comparable_prior_count": 1}]
    with pytest.raises(ValueError, match=message):
        evaluate_local_structure(reference(), {"changes": [{"id": "event", "time_s": 10.5}], "transitions": []}, [], history)


def test_reference_spans_must_be_ordered_nonoverlapping_with_known_certainty():
    ref = reference()
    ref["layers"][0]["spans"][1]["start_s"] = 9
    with pytest.raises(ValueError, match="unsorted or overlaps"):
        evaluate_local_structure(ref, predictions(), [], [])
    ref = reference()
    ref["layers"][0]["spans"][1]["start_s"] = 20
    ref["layers"][0]["spans"][1]["end_s"] = 30
    ref["layers"][0]["spans"][2]["start_s"] = 10
    ref["layers"][0]["spans"][2]["end_s"] = 20
    with pytest.raises(ValueError, match="unsorted or overlaps"):
        evaluate_local_structure(ref, predictions(), [], [])
    ref = reference()
    ref["layers"][0]["spans"][0]["certainty"] = "probably"
    with pytest.raises(ValueError, match="certainty is invalid"):
        evaluate_local_structure(ref, predictions(), [], [])


def test_recurrence_scores_must_be_bounded_or_null():
    history = recurrence()
    history[0]["context"][1]["historical_pattern_novelty"] = 1.01
    with pytest.raises(ValueError, match="between zero and one"):
        evaluate_local_structure(reference(), predictions(), [], history)


def test_inputs_are_not_mutated_and_invalid_prediction_bounds_are_rejected():
    ref, predicted, history = reference(), predictions(), recurrence()
    before = deepcopy((ref, predicted, history))
    evaluate_local_structure(ref, predicted, [], history)
    assert (ref, predicted, history) == before
    predicted["transitions"][0]["end_s"] = 61
    with pytest.raises(ValueError, match="within source duration"):
        evaluate_local_structure(ref, predicted, [], history)
