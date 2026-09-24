import json

import numpy as np
import pytest

import experiments.evaluate_arctic_arrangement as runner
from songviz.local_structure import detect_local_structure
from songviz.local_structure_variants import detect_local_structure_variant


def _result() -> dict:
    return {
        "times_s": [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.],
        "changes": [{"id": "inside", "beat_index": 9, "support_start_s": 1., "support_end_s": 17.},
                    {"id": "outside", "beat_index": 9, "support_start_s": .9, "support_end_s": 17.}],
        "transitions": [],
        "curves": [{"scale_beats": 2, "pattern_change": list(range(19)),
                    "arrangement_change": list(range(19)), "combined_change": list(range(19)), "threshold": .2}],
    }


def test_run_refuses_overwrite_and_hash_mismatch_before_extraction(tmp_path) -> None:
    output = tmp_path / "exists"
    output.mkdir()
    with pytest.raises(FileExistsError, match="overwrite"):
        runner.run(tmp_path / "missing", tmp_path / "missing-registration.json", output)
    repo = tmp_path / "repo"
    repo.mkdir()
    input_path = repo / "wrong.bin"
    input_path.write_bytes(b"actual")
    registration = repo / "registration.json"
    registration.write_text(json.dumps({"inputs": [{"path": "wrong.bin", "sha256": "0" * 64}],
                                        "bounds_s": [1., 2.], "human_song_time_s": 1.5}))
    with pytest.raises(ValueError, match="Fingerprint mismatch"):
        runner.run(repo, registration, repo / "new-output")
    assert not (repo / "new-output").exists()


def test_common_eight_beat_filter_masks_every_curve_and_requires_full_candidate_support() -> None:
    visible = runner.visible_result(_result(), (1., 17.))
    # Index 8 is inside the literal time bounds, but lacks the common eight-beat left support.
    assert visible["eligible_anchor_indices"] == [9]
    assert visible["curves"][0]["pattern_change"] == [None] * 9 + [9] + [None] * 9
    assert [item["id"] for item in visible["changes"]] == ["inside"]


def test_registration_accepts_bound_context_and_byte_counts(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "REQUIRED_INPUTS", {"source"})
    inputs = []
    for name in ("source", "context"):
        path = tmp_path / name
        path.write_text(name)
        inputs.append({"path": name, "sha256": runner.digest(path), "bytes": path.stat().st_size})
    registration = tmp_path / "registration.json"
    value = {"inputs": inputs, "bounds_s": [1., 24.], "human_song_time_s": 17.}
    registration.write_text(json.dumps(value))
    assert len(runner.load_registration(tmp_path, registration)[1]) == 2
    value["inputs"][0]["bytes"] += 1
    registration.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Byte count mismatch"):
        runner.load_registration(tmp_path, registration)


def test_stem_metadata_must_bind_the_registered_source(tmp_path) -> None:
    metadata = tmp_path / runner.STEMS_META
    metadata.parent.mkdir(parents=True)
    metadata.write_text(json.dumps({"input": {"path": runner.SOURCE, "sha256": "0" * 64}}))
    with pytest.raises(ValueError, match="does not bind"):
        runner.validate_cache(tmp_path, (124., 148.), "1" * 64)


def test_tiny_real_detectors_keep_the_existing_three_policy_schema() -> None:
    beats = np.arange(25, dtype=float)
    energy = {name: np.r_[np.ones(12), np.full(12, 2.)] for name in ("bass", "drums", "other", "vocals")}
    features = {name: np.log1p(np.vstack((values, values * .5))) for name, values in energy.items()}
    control = detect_local_structure(features, energy, beats)
    separate = detect_local_structure_variant(features, energy, beats, variant="separate_channels")
    sustained = detect_local_structure_variant(features, energy, beats, variant="sustained_activity")
    assert [curve["scale_beats"] for curve in control["curves"]] == [2, 4, 8]
    assert [curve["scale_beats"] for curve in separate["channel_curves"]] == [2, 4, 8]
    assert [curve["scale_beats"] for curve in sustained["activity_curves"]] == [4, 8]
    visible = runner.visible_result(sustained, (8., 16.))
    assert isinstance(visible["activity_curves"][0]["stem_evidence"], dict)
    assert json.loads(json.dumps({"control": control, "separate": separate, "sustained": sustained}))
