from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments import build_music_representation_review as builder
from experiments.probe_music_representation import chunk_geometry
from songviz.ingest import sha256_file


def _record(path: Path) -> dict:
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _rewrite_frames(
    runtime: Path, manifest: dict, *, support_start: np.ndarray | None = None, support_end: np.ndarray | None = None
) -> None:
    """Rewrite synthetic embeddings and deliberately refresh only their binding."""
    frames = runtime / "frames.npz"
    with np.load(frames, allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    if support_start is not None:
        arrays["support_start_s"] = support_start
    if support_end is not None:
        arrays["support_end_s"] = support_end
    np.savez(frames, **arrays)
    manifest["frames"]["sha256"] = sha256_file(frames)
    (runtime / "manifest.json").write_text(json.dumps(manifest))


def _runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A short but otherwise exact runtime artifact; it never invokes MuQ."""
    runtime = tmp_path / "runtime"
    runtime.mkdir(parents=True)
    files = {}
    for name in ("source.flac", "weights.safetensors", "config.json", "code.py", "library.py", "snapshot.py"):
        path = runtime / name
        path.write_text(name)
        files[name] = path
    monkeypatch.setattr(builder, "SOURCE_SHA256", sha256_file(files["source.flac"]))
    monkeypatch.setattr(builder, "MUQ_WEIGHTS_SHA256", sha256_file(files["weights.safetensors"]))
    monkeypatch.setattr(builder, "MUQ_CONFIG_SHA256", sha256_file(files["config.json"]))

    source_frames, source_rate = 44_100, 44_100
    chunks = chunk_geometry(source_frames, source_rate, 0.0)
    assert len(chunks) == 1
    expected = [(chunk["start_s"] + index / 25, chunk["support_start_s"], chunk["support_end_s"])
                for chunk in chunks for index in chunk["retained_frame_indices"]]
    frames = runtime / "frames.npz"
    np.savez(
        frames,
        embeddings=np.ones((len(expected), 1024), dtype=np.float32),
        frame_times_s=np.array([row[0] for row in expected], dtype=np.float64),
        support_start_s=np.array([row[1] for row in expected], dtype=np.float64),
        support_end_s=np.array([row[2] for row in expected], dtype=np.float64),
    )
    manifest = {
        "schema_version": 1,
        "kind": "songviz-muq-runtime",
        "clock": {"source_seconds": True, "frame_hz": 25.0, "chunk_seconds": 10, "hop_seconds": 5,
                  "offset_seconds": 0.0, "ownership": "closest eligible chunk center; earlier chunk wins exact ties"},
        "source": {**_record(files["source.flac"]), "source_audio_sha256": builder.SOURCE_SHA256,
                   "sample_rate": source_rate, "channels": 2, "frames": source_frames,
                   "duration_s": source_frames / source_rate},
        "checkpoint": {**_record(files["weights.safetensors"]), "id": builder.MUQ_CHECKPOINT_ID,
                       "revision": builder.MUQ_CHECKPOINT_REVISION},
        "config": {**_record(files["config.json"]), "test": True},
        "code": _record(files["code.py"]),
        "extraction": {"encoder_input_sample_rate": 24_000, "dtype": "float32", "eval": True,
                       "selected_layer": "last_hidden_state", "source_cut_before_resample": True,
                       "mono": "arithmetic mean of stereo float32 PCM"},
        "chunks": chunks,
        "frames": {"path": "frames.npz", "sha256": sha256_file(frames), "source_audio_sha256": builder.SOURCE_SHA256,
                   "shape": [len(expected), 1024], "finite": True},
        "library_sources": [_record(files["library.py"])],
        "input_snapshots": [_record(files["snapshot.py"])],
    }
    (runtime / "manifest.json").write_text(json.dumps(manifest))
    return runtime


def test_load_runtime_accepts_exact_short_fingerprinted_geometry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _runtime(tmp_path, monkeypatch)
    manifest, arrays, inputs = builder.load_runtime(runtime)
    assert manifest["checkpoint"]["revision"] == builder.MUQ_CHECKPOINT_REVISION
    assert arrays["embeddings"].shape == (25, 1024)
    assert arrays["embeddings"].dtype == float
    assert arrays["frame_times_s"].dtype == float
    assert len(inputs) == 8  # manifest, frames, source/model/config/code, library, snapshot


@pytest.mark.parametrize("field", ["source", "config", "code"])
def test_load_runtime_rejects_missing_fingerprinted_provenance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str) -> None:
    runtime = _runtime(tmp_path, monkeypatch)
    manifest_path = runtime / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.pop(field)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="fingerprints"):
        builder.load_runtime(runtime)


@pytest.mark.parametrize("name", [
    "source.flac", "weights.safetensors", "config.json", "code.py", "library.py", "snapshot.py",
])
def test_load_runtime_rejects_changed_actual_provenance_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    runtime = _runtime(tmp_path, monkeypatch)
    (runtime / name).write_text("tampered")
    with pytest.raises(ValueError, match="Changed or missing fingerprinted input"):
        builder.load_runtime(runtime)


def test_load_runtime_requires_pinned_source_model_config_and_24khz_protocol(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _runtime(tmp_path, monkeypatch)
    manifest_path = runtime / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source"]["source_audio_sha256"] = "wrong"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="pinned source/model/config"):
        builder.load_runtime(runtime)
    runtime = _runtime(tmp_path / "protocol", monkeypatch)
    manifest_path = runtime / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["extraction"]["encoder_input_sample_rate"] = 44_100
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="pinned MuQ extraction"):
        builder.load_runtime(runtime)


@pytest.mark.parametrize(("key", "value"), [("sample_rate", 48_000), ("channels", 1), ("duration_s", 0.5)])
def test_load_runtime_requires_complete_pinned_source_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str, value: float | int
) -> None:
    runtime = _runtime(tmp_path, monkeypatch)
    manifest_path = runtime / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source"][key] = value
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="pinned stereo 44.1kHz source extent"):
        builder.load_runtime(runtime)


@pytest.mark.parametrize("mutation,error", [
    ("ownership", "source_seconds clock"),
    ("chunk_support", "pinned ownership geometry"),
    ("chunk_center", "pinned ownership geometry"),
    ("frame_support", "exactly reconstruct"),
])
def test_load_runtime_rejects_independent_geometry_tampering_even_with_refreshed_frame_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str, error: str
) -> None:
    runtime = _runtime(tmp_path, monkeypatch)
    manifest_path = runtime / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if mutation == "ownership":
        manifest["clock"]["ownership"] = "anything else"
        manifest_path.write_text(json.dumps(manifest))
    elif mutation == "chunk_support":
        manifest["chunks"][0]["support_end_s"] = 0.9
        manifest_path.write_text(json.dumps(manifest))
    elif mutation == "chunk_center":
        manifest["chunks"][0]["center_s"] = 0.25
        manifest_path.write_text(json.dumps(manifest))
    else:
        _rewrite_frames(runtime, manifest, support_end=np.full(25, 0.98, dtype=np.float64))
    with pytest.raises(ValueError, match=error):
        builder.load_runtime(runtime)


def test_load_runtime_rejects_tampered_frame_bytes_even_when_geometry_is_valid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _runtime(tmp_path, monkeypatch)
    (runtime / "frames.npz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="does not match"):
        builder.load_runtime(runtime)
