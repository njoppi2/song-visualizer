from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from experiments import evaluate_structure_feedback as builder
from songviz.ingest import sha256_file
from songviz.recurrence import compare_phrases


DURATION_S = 48.0
SAMPLE_RATE = 100
STEMS = ("bass", "drums", "other", "vocals")


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _record(path: Path) -> dict:
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _fixed_evidence() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict]:
    beats = np.arange(int(DURATION_S) + 1, dtype=float)
    features = {
        name: np.vstack(
            [
                1.0 + (index + 1) * np.sin(beats[:-1] / (3 + index)),
                1.0 + (index + 1) * np.cos(beats[:-1] / (5 + index)),
            ]
        )
        for index, name in enumerate(STEMS)
    }
    energy = {name: np.full(beats.size - 1, 0.25 + index * 0.1) for index, name in enumerate(STEMS)}
    timing = {
        "sample_rate": SAMPLE_RATE,
        "hop_length": 1,
        "requested_times_s": beats.tolist(),
        "feature_frame_indices": list(range(beats.size)),
        "feature_frame_times_s": beats.tolist(),
        "timing_note": "synthetic fixed-grid test evidence",
    }
    return features, energy, timing


def _grid_sha256(beats: np.ndarray) -> str:
    return hashlib.sha256(
        json.dumps(beats.tolist(), separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _write_wav(path: Path, *, value: float = 0.0, duration_s: float = DURATION_S) -> None:
    samples = np.full(round(SAMPLE_RATE * duration_s), value, dtype=np.float32)
    sf.write(path, samples, SAMPLE_RATE)


def _make_package(tmp_path: Path, *, with_interpretations: bool) -> dict[str, Path | dict]:
    parent = tmp_path / "parent"
    editor = tmp_path / "editor"
    feedback_dir = tmp_path / "feedback"
    stem_dir = tmp_path / "stems"
    parent.mkdir()
    editor.mkdir()
    feedback_dir.mkdir()
    stem_dir.mkdir()
    source = tmp_path / "source.wav"
    original = parent / "original.wav"
    _write_wav(source, value=0.1)
    _write_wav(original, value=0.2)
    stems = {name: stem_dir / f"{name}.wav" for name in STEMS}
    for index, path in enumerate(stems.values()):
        _write_wav(path, value=0.01 * (index + 1))

    beats = np.arange(int(DURATION_S) + 1, dtype=float)
    features, energy, _ = _fixed_evidence()
    recurrence = {
        "beat_grid_sha256": _grid_sha256(beats),
        "results": [
            compare_phrases(features, energy, beats, scale_beats=scale, stride_beats=4)
            for scale in (16, 32)
        ],
    }
    story = {
        "meta": {
            "duration_s": DURATION_S,
            "beat_grid": {
                "explicit": True,
                "fallback": None,
                "requested_times_s": beats.tolist(),
                "requested_sha256": recurrence["beat_grid_sha256"],
            },
        },
        "sections": [
            {"start_s": 0.0, "end_s": 24.0, "role": "legacy-a", "label": "A"},
            {"start_s": 24.0, "end_s": DURATION_S, "role": "legacy-b", "label": "B"},
        ],
    }
    story_path = parent / "candidate-story.json"
    recurrence_path = parent / "recurrence.json"
    _write_json(story_path, story)
    _write_json(recurrence_path, recurrence)
    parent_manifest = {
        "outputs": [_record(path) for path in (original, story_path, recurrence_path)],
        "sources": [_record(source), *[_record(path) for path in stems.values()]],
    }
    _write_json(parent / "manifest.json", parent_manifest)

    source_fields = {
        "audio_sha256": sha256_file(original),
        "source_audio_sha256": sha256_file(source),
        "duration_s": DURATION_S,
        "song_title": "Synthetic structural feedback",
    }
    editor_data = dict(source_fields)
    editor_data_path = editor / "editor.json"
    _write_json(editor_data_path, editor_data)
    _write_json(editor / "manifest.json", {"outputs": [_record(editor_data_path)]})

    segments = [
        {"id": "span-a", "start_s": 0.0, "end_s": 12.0, "label": "opaque A", "motif": "idea", "notes": "", "certainty": "unspecified"},
        {"id": "span-b", "start_s": 12.0, "end_s": 24.0, "label": "opaque B", "motif": "", "notes": "", "certainty": "unspecified"},
        {"id": "span-c", "start_s": 24.0, "end_s": 36.0, "label": "opaque C", "motif": "idea", "notes": "", "certainty": "unspecified"},
        {"id": "span-d", "start_s": 36.0, "end_s": DURATION_S, "label": "opaque D", "motif": "", "notes": "", "certainty": "unspecified"},
    ]
    feedback = {
        "schema_version": 1,
        "kind": "songviz-section-annotations",
        "manifest_sha256": sha256_file(editor / "manifest.json"),
        "source": source_fields,
        "annotations": {
            "layers": [{"id": "parts", "name": "Opaque parts", "segments": segments}],
            "active_layer_id": "parts",
            "selected_segment_id": "span-d",
            "global_notes": "",
        },
    }
    feedback_path = feedback_dir / "raw-feedback.json"
    _write_json(feedback_path, feedback)
    interpretation_path: Path | None = None
    if with_interpretations:
        interpretation_path = feedback_dir / "interpretations.json"
        _write_json(
            interpretation_path,
            {
                "schema_version": 1,
                "feedback_sha256": sha256_file(feedback_path),
                "segments": {
                    "span-b": {"transition": True, "rationale": "Explicit test-only interpretation."},
                    "span-c": {"variation": "changed arrangement", "rationale": "Explicit test-only interpretation."},
                },
            },
        )
    return {
        "feedback": feedback_path,
        "interpretations": interpretation_path,
        "parent": parent,
        "editor": editor,
        "out": tmp_path / "output",
        "source": source,
        "original": original,
        "stems": stems,
    }


def _resolve_record_path(record: dict) -> Path:
    path = Path(record["path"])
    return path if path.is_absolute() else builder.ROOT / path


def _assert_fingerprints(records: list[dict]) -> None:
    for entry in records:
        path = _resolve_record_path(entry)
        assert sha256_file(path) == entry["sha256"]
        assert path.stat().st_size == entry["bytes"]


@pytest.mark.parametrize("with_interpretations", [False, True])
def test_builds_bounded_package_with_verified_snapshots_and_no_annotation_acoustics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_interpretations: bool
) -> None:
    package = _make_package(tmp_path, with_interpretations=with_interpretations)
    features, energy, timing = _fixed_evidence()
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fixed_extract_features(*args: object, **kwargs: object) -> tuple[dict, dict, dict]:
        calls.append((args, kwargs))
        return features, energy, timing

    monkeypatch.setattr(builder, "extract_features", fixed_extract_features)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    builder.build(
        feedback=package["feedback"],
        interpretations=package["interpretations"],
        parent=package["parent"],
        editor=package["editor"],
        out=package["out"],
    )

    assert {path: path.read_bytes() for path in before} == before
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert kwargs == {}
    assert len(args) == 2
    assert set(args[0]) == set(STEMS)
    assert all(isinstance(path, Path) for path in args[0].values())
    assert np.array_equal(args[1], np.arange(int(DURATION_S) + 1, dtype=float))
    # The only extractor inputs are cached stem paths and the beat grid: no
    # labels, annotation reference, certainty, or human identity is provided.
    assert not any(isinstance(value, str) and "idea" in value for value in args)

    out = package["out"]
    manifest = json.loads((out / "manifest.json").read_text())
    _assert_fingerprints(manifest["sources"])
    _assert_fingerprints(manifest["input_snapshots"])
    _assert_fingerprints(manifest["outputs"])
    assert manifest["settings"]["feedback_used_in_acoustic_extraction"] is False
    assert (out / "inputs" / package["feedback"].name).read_bytes() == package["feedback"].read_bytes()
    if with_interpretations:
        assert (out / "inputs" / package["interpretations"].name).read_bytes() == package["interpretations"].read_bytes()
    reference = json.loads((out / "reference.json").read_text())
    assert {span["certainty"] for span in reference["layers"][0]["spans"]} == {"unspecified"}
    assert reference["layers"][0]["spans"][0]["identity_id"] is not None
    if with_interpretations:
        assert reference["layers"][0]["spans"][1]["transition"] is True
    else:
        assert all(span["transition"] is None for span in reference["layers"][0]["spans"])


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("feedback_audio", "differs"),
        ("feedback_source", "differs"),
        ("editor", "Changed or mismatched"),
        ("candidate", "Changed or mismatched"),
        ("recurrence", "Changed or mismatched"),
        ("audio", "Changed or mismatched"),
        ("stem_duration", "differs"),
    ],
)
def test_invalid_packages_fail_before_feature_extraction_or_output_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str, message: str
) -> None:
    package = _make_package(tmp_path, with_interpretations=False)
    feedback_path = package["feedback"]
    if damage.startswith("feedback_"):
        feedback = json.loads(feedback_path.read_text())
        key = "audio_sha256" if damage == "feedback_audio" else "source_audio_sha256"
        feedback["source"][key] = "0" * 64
        _write_json(feedback_path, feedback)
    elif damage == "editor":
        (package["editor"] / "editor.json").write_text('{"altered": true}\n')
    elif damage == "candidate":
        (package["parent"] / "candidate-story.json").write_text('{"altered": true}\n')
    elif damage == "recurrence":
        (package["parent"] / "recurrence.json").write_text('{"altered": true}\n')
    elif damage == "audio":
        with (package["original"]).open("ab") as changed:
            changed.write(b"changed")
    else:
        stem = package["stems"]["bass"]
        _write_wav(stem, duration_s=DURATION_S - 1)
        manifest_path = package["parent"] / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        stem_record = next(entry for entry in manifest["sources"] if Path(entry["path"]) == stem)
        stem_record.update(_record(stem))
        _write_json(manifest_path, manifest)

    extracted = False

    def must_not_extract(*args: object, **kwargs: object) -> tuple[dict, dict, dict]:
        nonlocal extracted
        extracted = True
        raise AssertionError("invalid input reached acoustic extraction")

    monkeypatch.setattr(builder, "extract_features", must_not_extract)
    with pytest.raises(ValueError, match=message):
        builder.build(
            feedback=package["feedback"],
            interpretations=None,
            parent=package["parent"],
            editor=package["editor"],
            out=package["out"],
        )
    assert extracted is False
    assert not package["out"].exists()
