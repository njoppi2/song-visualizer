from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pytest
import soundfile as sf

from experiments import build_local_structure_review as builder
from songviz.ingest import sha256_file
from songviz.recurrence import compare_phrases


DURATION_S = 48.0
SAMPLE_RATE = 100
STEMS = ("bass", "vocals")


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _record(path: Path) -> dict:
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _write_wav(path: Path, value: float, duration_s: float = DURATION_S) -> None:
    sf.write(path, np.full(round(SAMPLE_RATE * duration_s), value, dtype=np.float32), SAMPLE_RATE)


def _assert_records(records: list[dict]) -> None:
    for record in records:
        path = Path(record["path"])
        if not path.is_absolute():
            path = builder.ROOT / path
        assert sha256_file(path) == record["sha256"]
        assert path.stat().st_size == record["bytes"]


def _refresh_record(manifest_path: Path, file_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text())
    for group in ("outputs", "sources"):
        for record in manifest.get(group, []):
            if Path(record["path"]).resolve() == file_path.resolve():
                record.update(_record(file_path))
                _write_json(manifest_path, manifest)
                return
    raise AssertionError(f"No manifest record for {file_path}")


def _package(tmp_path: Path) -> dict[str, Path]:
    parent = tmp_path / "parent"
    audio_review = tmp_path / "audio-review"
    parent.mkdir()
    audio_review.mkdir()
    source = tmp_path / "rawsource.wav"
    review_audio = audio_review / "original.wav"
    _write_wav(source, 0.07)
    _write_wav(review_audio, 0.19)
    assert source.read_bytes() != review_audio.read_bytes()

    beats = np.arange(49, dtype=float)
    features = {
        "bass": np.vstack((np.where(beats[:-1] < 24, 1.0, 2.0), np.full(48, 0.5))),
        "vocals": np.vstack((np.full(48, 0.75), 0.5 + (beats[:-1] % 5) / 10)),
    }
    energy = {"bass": np.where(beats[:-1] < 24, 0.2, 0.4), "vocals": np.full(48, 0.3)}
    features_path = parent / "features.npz"
    np.savez(features_path, beat_times_s=beats,
             **{f"{name}_features": values for name, values in features.items()},
             **{f"{name}_rms": values for name, values in energy.items()})
    timing_path = parent / "timing.json"
    _write_json(timing_path, {"requested_times_s": beats.tolist(), "timing_note": "synthetic 48-beat cache"})
    recurrence_path = parent / "recurrence.json"
    _write_json(recurrence_path, {"results": [
        compare_phrases(features, energy, beats, scale_beats=scale, stride_beats=4)
        for scale in (16, 32)
    ]})
    legacy_path = parent / "legacy-sections.json"
    _write_json(legacy_path, {"sections": [
        {"start_s": 0.0, "end_s": 24.0, "label": "fixed A"},
        {"start_s": 24.0, "end_s": 48.0, "label": "fixed B"},
    ]})
    reference_path = parent / "reference.json"
    source_fields = {"audio_sha256": sha256_file(review_audio),
                     "source_audio_sha256": sha256_file(source),
                     "duration_s": DURATION_S, "song_title": "Tiny <Synthetic>"}
    _write_json(reference_path, {"source": source_fields, "layers": [{
        "id": "human", "name": "Human variations", "spans": [
            {"id": "v1", "start_s": 0.0, "end_s": 16.0, "label": "A", "certainty": "unspecified",
             "identity_id": "motif", "variation": "lighter", "transition": None},
            {"id": "v2", "start_s": 16.0, "end_s": 32.0, "label": "A varied", "certainty": "unspecified",
             "identity_id": "motif", "variation": "denser", "transition": False},
            {"id": "unknown", "start_s": 32.0, "end_s": 48.0, "label": "?", "certainty": "unspecified",
             "identity_id": None, "variation": None, "transition": None},
        ],
    }]})
    review_json = audio_review / "review.json"
    _write_json(review_json, {"duration_s": DURATION_S, "waveform": [{"time_s": 0.0, "value": 0.0}]})
    audio_manifest = audio_review / "manifest.json"
    _write_json(audio_manifest, {"outputs": [_record(review_audio), _record(review_json)]})
    parent_manifest = parent / "manifest.json"
    _write_json(parent_manifest, {
        "kind": "songviz-structural-development-evaluation",
        "outputs": [_record(path) for path in (reference_path, features_path, timing_path, recurrence_path, legacy_path)],
        "sources": [_record(audio_manifest), _record(source)],
    })
    return {"parent": parent, "audio_review": audio_review, "source": source,
            "review_audio": review_audio, "reference": reference_path, "features": features_path,
            "timing": timing_path, "recurrence": recurrence_path, "review_json": review_json, "audio_manifest": audio_manifest,
            "parent_manifest": parent_manifest}


def test_builds_verified_sidecar_without_mutating_real_inputs(tmp_path: Path) -> None:
    package = _package(tmp_path)
    out = tmp_path / "local-review"
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    builder.build(parent=package["parent"], audio_review=package["audio_review"], out=out)

    assert {path: path.read_bytes() for path in before} == before
    manifest = json.loads((out / "manifest.json").read_text())
    _assert_records(manifest["sources"])
    _assert_records(manifest["input_snapshots"])
    _assert_records(manifest["outputs"])
    review = json.loads((out / "review.json").read_text())
    assert (out / unquote(review["audio_path"])).resolve() == package["review_audio"].resolve()
    assert (out / "inputs" / "songviz" / "local_structure.py").is_file()
    assert json.loads((out / "predictions.json").read_text()) == review["predictions"]


def test_embedded_json_escapes_script_content_and_round_trips() -> None:
    payload = {"unsafe": "</script><img src=x>&", "nested": ["<tag>"]}
    encoded = builder.embedded_json(payload)
    assert "<" not in encoded and ">" not in encoded and "&" not in encoded
    assert json.loads(encoded) == payload


def test_refuses_existing_and_nested_outputs(tmp_path: Path) -> None:
    package = _package(tmp_path)
    out = tmp_path / "review"
    builder.build(parent=package["parent"], audio_review=package["audio_review"], out=out)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        builder.build(parent=package["parent"], audio_review=package["audio_review"], out=out)
    with pytest.raises(ValueError, match="Output must be separate"):
        builder.build(parent=package["parent"], audio_review=package["audio_review"], out=package["parent"] / "nested")


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("features", r"Changed or mismatched input: .*features\.npz"),
        ("reference", r"Changed or mismatched input: .*reference\.json"),
        ("timing", r"Changed or mismatched input: .*timing\.json"),
        ("recurrence_window", "Recurrence windows differ from the feature beat grid"),
        ("audio_manifest", r"Changed or mismatched input: .*audio-review.*/manifest\.json"),
        ("audio", r"Changed or mismatched input: .*audio-review.*/original\.wav"),
        ("source", r"Changed or mismatched input: .*rawsource\.wav"),
        ("audio_duration", "Audio duration differs from reference"),
        ("review_duration", "Review waveform duration differs from audio"),
    ],
)
def test_corrupt_inputs_fail_before_creating_output(tmp_path: Path, damage: str, message: str) -> None:
    package = _package(tmp_path)
    if damage in {"features", "reference", "timing"}:
        package[damage].write_bytes(b"corrupt")
    elif damage == "recurrence_window":
        recurrence = json.loads(package["recurrence"].read_text())
        recurrence["results"][0]["spans"][0]["end_s"] += 0.001
        _write_json(package["recurrence"], recurrence)
        _refresh_record(package["parent_manifest"], package["recurrence"])
    elif damage == "audio_manifest":
        package["audio_manifest"].write_bytes(b"corrupt")
    elif damage == "audio":
        _write_wav(package["review_audio"], 0.31)
    elif damage == "source":
        _write_wav(package["source"], 0.42)
    elif damage == "review_duration":
        _write_json(package["review_json"], {"duration_s": DURATION_S - 1, "waveform": []})
        _refresh_record(package["audio_manifest"], package["review_json"])
        _refresh_record(package["parent_manifest"], package["audio_manifest"])
    else:
        _write_wav(package["review_audio"], 0.31, DURATION_S - 1)
        _refresh_record(package["audio_manifest"], package["review_audio"])
        _refresh_record(package["parent_manifest"], package["audio_manifest"])
        reference = json.loads(package["reference"].read_text())
        reference["source"]["audio_sha256"] = sha256_file(package["review_audio"])
        _write_json(package["reference"], reference)
        _refresh_record(package["parent_manifest"], package["reference"])
    out = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match=message):
        builder.build(parent=package["parent"], audio_review=package["audio_review"], out=out)
    assert not out.exists()


def test_human_reference_edits_do_not_change_acoustic_predictions(tmp_path: Path) -> None:
    package = _package(tmp_path)
    first, second = tmp_path / "first", tmp_path / "second"
    builder.build(parent=package["parent"], audio_review=package["audio_review"], out=first)
    original_predictions = json.loads((first / "predictions.json").read_text())

    reference = json.loads(package["reference"].read_text())
    reference["layers"][0]["spans"][0]["label"] = "completely replaced human wording"
    reference["layers"][0]["spans"][1]["variation"] = "other human interpretation"
    _write_json(package["reference"], reference)
    _refresh_record(package["parent_manifest"], package["reference"])
    builder.build(parent=package["parent"], audio_review=package["audio_review"], out=second)

    assert json.loads((second / "predictions.json").read_text()) == original_predictions
