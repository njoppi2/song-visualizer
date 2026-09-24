import json

import numpy as np
import pytest

import experiments.diagnose_transition_components as diagnostic
from experiments.diagnose_transition_components import (
    RESERVE,
    extrema,
    filtered_candidates,
    reserve_visible,
    run,
)
from songviz.local_structure import detect_local_structure
from songviz.local_structure_variants import detect_local_structure_variant


def _result() -> dict:
    return {"times_s": [0.0, 1.0, 2.0, 3.0], "changes": [], "transitions": [], "curves": [
        {"scale_beats": 2, "pattern_change": [None, .4, .8, .8],
         "arrangement_change": [None, .6, .6, .2], "combined_change": [None, .6, .8, .8], "threshold": .2},
    ]}


def test_filter_requires_entire_candidate_support_inside_bounds() -> None:
    result = _result()
    result["changes"] = [
        {"id": "inside", "support_start_s": RESERVE[0], "support_end_s": RESERVE[1]},
        {"id": "left", "support_start_s": RESERVE[0] - .01, "support_end_s": RESERVE[1]},
        {"id": "right", "support_start_s": RESERVE[0], "support_end_s": RESERVE[1] + .01},
    ]
    assert [row["id"] for row in filtered_candidates(result, RESERVE)] == ["inside"]


def test_component_extrema_remain_independent_and_keep_ties() -> None:
    summary = extrema(_result(), (0.0, 3.0))["2"]
    assert summary["pattern_change"] == {"score": .8, "anchor_indices": [2, 3], "anchor_times_s": [2.0, 3.0]}
    assert summary["arrangement_change"] == {"score": .6, "anchor_indices": [1, 2], "anchor_times_s": [1.0, 2.0]}


def test_reserve_view_masks_outside_curves_and_removes_auxiliary_full_track_arrays() -> None:
    result = _result() | {"channel_curves": [{"outside": [1]}], "activity_curves": [{"outside": [1]}]}
    result["times_s"] = [RESERVE[0], RESERVE[0] + 1, RESERVE[1] - 1, RESERVE[1]]
    result["curves"][0]["scale_beats"] = 1
    visible = reserve_visible(result)
    assert "channel_curves" not in visible and "activity_curves" not in visible
    assert visible["curves"][0]["pattern_change"] == [None, .4, .8, None]


def test_reserve_uses_the_common_eight_beat_style_eligibility_for_every_scale() -> None:
    result = _result()
    result["times_s"] = [RESERVE[0], RESERVE[0] + 1, RESERVE[0] + 2,
                         RESERVE[1] - 2, RESERVE[1] - 1, RESERVE[1]]
    result["curves"] = [
        {"scale_beats": 1, "pattern_change": [0] * 6, "arrangement_change": [0] * 6,
         "combined_change": [0] * 6, "threshold": .2},
        {"scale_beats": 2, "pattern_change": [0] * 6, "arrangement_change": [0] * 6,
         "combined_change": [0] * 6, "threshold": .2},
    ]
    visible = reserve_visible(result)
    # Index 1 has one-beat support but lacks the common two-beat support.
    assert visible["curves"][0]["pattern_change"] == [None, None, 0, 0, None, None]


def test_run_refuses_existing_output_before_reading_inputs(tmp_path) -> None:
    output = tmp_path / "already-there"
    output.mkdir()
    with pytest.raises(FileExistsError, match="overwrite"):
        run(tmp_path / "missing-repo", output)


def test_run_refuses_a_mismatched_pinned_input(tmp_path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    pinned = repo / "pinned.json"
    pinned.write_text(json.dumps({"not": "the pinned bytes"}))
    monkeypatch.setattr(diagnostic, "PINS", {"pinned.json": "0" * 64})
    with pytest.raises(ValueError, match="Fingerprint mismatch"):
        run(repo, tmp_path / "new-output")


def test_manifest_dictionary_sources_and_outputs_have_distinct_roots(tmp_path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    source = tmp_path / "source.json"
    source.write_text("source")
    output = package / "prediction.json"
    output.write_text("prediction")
    manifest = {"sources": {"source.json": diagnostic.digest(source)},
                "outputs": {"prediction.json": diagnostic.digest(output)}}
    diagnostic.verify_records(tmp_path, package, manifest, ("sources", "outputs"))


def test_real_detector_and_separate_channel_schema_serializes() -> None:
    beats = np.arange(21, dtype=float)
    energy = {name: np.r_[np.ones(10), np.full(10, 2.0)] for name in ("bass", "drums", "other", "vocals")}
    features = {name: np.log1p(np.vstack((energy[name], energy[name] * .5))) for name in energy}
    control = detect_local_structure(features, energy, beats)
    separate = detect_local_structure_variant(features, energy, beats, variant="separate_channels")
    assert [curve["scale_beats"] for curve in control["curves"]] == [2, 4, 8]
    assert [curve["scale_beats"] for curve in separate["channel_curves"]] == [2, 4, 8]
    assert json.loads(json.dumps(separate))["variant"] == "separate_channels"
