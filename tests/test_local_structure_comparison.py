import json
from pathlib import Path
from copy import deepcopy

import numpy as np
import pytest

from experiments import build_local_structure_review as control_builder
from experiments import compare_local_structure as comparison
from songviz.ingest import sha256_file
from songviz.local_structure_variants import LocalStructureVariantConfig, detect_local_structure_variant

# The established synthetic parent/audio fixture is intentionally reused: this
# test covers the new comparison package, not a second version of its inputs.
from test_local_structure_builder import _package, _refresh_record, _write_json


def _control_package(tmp_path: Path) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    package = _package(tmp_path)
    control = tmp_path / "frozen-control"
    control_builder.build(parent=package["parent"], audio_review=package["audio_review"], out=control)
    package["control"] = control
    return package


def _build(package: dict[str, Path], out: Path) -> None:
    comparison.build(parent=package["parent"], control=package["control"],
                     audio_review=package["audio_review"], out=out)


def test_builds_four_variant_package_with_complete_fingerprints(tmp_path: Path) -> None:
    package = _control_package(tmp_path)
    out = tmp_path / "comparison"
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    _build(package, out)

    assert {path: path.read_bytes() for path in before} == before
    manifest = json.loads((out / "manifest.json").read_text())
    assert set(manifest["variants"]) == set(comparison.VARIANTS)
    assert manifest["numpy_version"] == np.__version__
    assert manifest["python_version"]
    for group in ("sources", "input_snapshots", "outputs"):
        assert manifest[group]
        for item in manifest[group]:
            path = Path(item["path"])
            if not path.is_absolute():
                path = comparison.ROOT / path
            assert path.is_file()
            assert sha256_file(path) == item["sha256"]
    assert {"index.html", "review.json", "report.md"}.issubset({Path(x["path"]).name for x in manifest["outputs"]})
    for relative in comparison.CODE:
        assert (out / "inputs" / relative).is_file()
    for variant in comparison.VARIANTS:
        assert (out / "variants" / variant / "predictions.json").is_file()
        assert (out / "variants" / variant / "evaluation.json").is_file()
    review = json.loads((out / "review.json").read_text())
    complete_control = json.loads((out / "variants" / "control" / "predictions.json").read_text())
    page_control = review["variants"]["control"]["predictions"]
    assert complete_control["changes"] == page_control["changes"]
    assert complete_control["transitions"] == page_control["transitions"]
    assert not {"curves", "channel_curves", "activity_curves"} & page_control.keys()
    assert "recurrence_context" not in review["variants"]["control"]["evaluation"]
    assert all(review["summary"]["variants"][name]["transition_overlap_unchanged_from_control"]
               for name in comparison.VARIANTS)
    assert "\\u003c" in comparison.embedded_json({"label": "</script>"})


def test_control_mutation_and_parent_hash_mismatch_refuse_before_output(tmp_path: Path) -> None:
    package = _control_package(tmp_path)
    (package["control"] / "predictions.json").write_text("{}\n")
    out = tmp_path / "bad-control"
    with pytest.raises(ValueError, match="mismatched input"):
        _build(package, out)
    assert not out.exists()

    package = _control_package(tmp_path / "second")
    package["features"].write_bytes(b"mutated")
    out = tmp_path / "bad-parent"
    with pytest.raises(ValueError, match="features\\.npz"):
        _build(package, out)
    assert not out.exists()


def test_grid_mismatch_and_existing_output_refuse(tmp_path: Path) -> None:
    package = _control_package(tmp_path)
    timing = json.loads(package["timing"].read_text())
    timing["requested_times_s"][4] += .25
    _write_json(package["timing"], timing)
    _refresh_record(package["parent_manifest"], package["timing"])
    with pytest.raises(ValueError, match="Feature and timing grids differ"):
        _build(package, tmp_path / "grid-bad")

    package = _control_package(tmp_path / "existing")
    out = tmp_path / "existing-out"
    _build(package, out)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        _build(package, out)
    with pytest.raises(ValueError, match="Output must be separate"):
        _build(package, package["parent"] / "nested")


def test_human_label_edits_do_not_change_predictions(tmp_path: Path) -> None:
    package = _control_package(tmp_path)
    first, second = tmp_path / "first", tmp_path / "second"
    _build(package, first)
    before = {name: json.loads((first / "variants" / name / "predictions.json").read_text())
              for name in comparison.VARIANTS}
    reference = json.loads(package["reference"].read_text())
    reference["layers"][0]["spans"][0]["label"] = "human wording | changed\nnot detector input"
    reference["layers"][0]["spans"][1]["variation"] = "another analyst interpretation"
    _write_json(package["reference"], reference)
    _refresh_record(package["parent_manifest"], package["reference"])
    _build(package, second)
    after = {name: json.loads((second / "variants" / name / "predictions.json").read_text())
             for name in comparison.VARIANTS}
    assert after == before


def test_regenerated_control_or_variant_dip_mutation_refuses_before_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package = _control_package(tmp_path)

    def changed_control(features, energy, beat_times, *, variant, config):
        result = detect_local_structure_variant(features, energy, beat_times, variant=variant, config=config)
        if variant == "control":
            result = deepcopy(result)
            result["method"] = "mutated-control-test-only"
        return result

    monkeypatch.setattr(comparison, "_variant_detector", lambda: (changed_control, LocalStructureVariantConfig))
    with pytest.raises(ValueError, match="Regenerated control"):
        _build(package, tmp_path / "control-mismatch")
    assert not (tmp_path / "control-mismatch").exists()

    package = _control_package(tmp_path / "dip")

    def changed_dip(features, energy, beat_times, *, variant, config):
        result = detect_local_structure_variant(features, energy, beat_times, variant=variant, config=config)
        if variant == "combined":
            result = deepcopy(result)
            result["transitions"] = result["transitions"] + [{"id": "mutated-dip", "start_s": 8.0, "end_s": 10.0}]
        return result

    monkeypatch.setattr(comparison, "_variant_detector", lambda: (changed_dip, LocalStructureVariantConfig))
    with pytest.raises(ValueError, match="energy-dip intervals"):
        _build(package, tmp_path / "dip-mismatch")
    assert not (tmp_path / "dip-mismatch").exists()


def test_report_escapes_pipe_and_newline_in_human_layer_text() -> None:
    variants = {name: {"predictions": {}, "evaluation": {"counts": {"changes": {"count": 0, "density_per_minute": 0},
                                                                       "transitions": {"count": 0, "density_per_minute": 0}}}}
                for name in comparison.VARIANTS}
    data = {"song_title": "song|title\nnext", "variants": variants, "notes": [],
            "summary": {"variants": {name: {"candidate_counts": variants[name]["evaluation"]["counts"],
                                                "exact_time_set_difference_from_control": {"gained_times_s": [], "lost_times_s": []}}
                                     for name in comparison.VARIANTS},
                        "human_boundary_nearest_signed_deltas_s": [{"layer_name": "raw|layer\nnext", "time_s": 1,
                                                                       "identity_relation": "unknown", "variation_change": None,
                                                                       "nearest_legacy_signed_delta_s": None,
                                                                       "nearest_signed_deltas_s": {name: None for name in comparison.VARIANTS}}],
                        "limitations": []}}
    report = comparison.report_markdown(data)
    assert "song\\|title next" in report
    assert "raw\\|layer next" in report


def test_template_uses_strict_numbers_and_plain_text_rendering() -> None:
    template = (comparison.ROOT / "experiments/templates/local_structure_comparison.html").read_text()
    assert "typeof v==='number'&&Number.isFinite(v)" in template
    assert "textContent=txt" in template
    assert "new MutationObserver(clip)" in template
