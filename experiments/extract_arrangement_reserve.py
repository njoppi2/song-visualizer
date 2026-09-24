#!/usr/bin/env python3
"""Freeze source-bound CQT/RMS inputs for the Agnes prospective reserve.

This script deliberately performs no candidate generation, scoring, plotting, or
reserve selection.  It reuses only cached stems and the same stored beat grid.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import resource
import shutil
import sys
import time

import librosa
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from songviz.ingest import sha256_file


STEMS = ("bass", "drums", "other", "vocals")
EXPECTED = {
    "source_sha256": "ac23d00df4892e03b8df1696927d1b066bcc147122112b7154239ea3dbb43fbb",
    "source_bytes": 49023445,
    "stems_meta_sha256": "21a15386ce7fb95ef01a4f7c594829daad8af03eb5fe49f7ac8abc9ec17a0f2a",
    "stems_json_sha256": "942e40fd74acc0f981028c0c6981305afe032662646f398cc11eb51a9a79b74d",
    "analysis_sha256": "fc96f5afa1b8b1ef685bd9dde2074be7bd5baddbc77f49cd0b9da22739e57c63",
    "analysis_bytes": 13828274,
    "protocol_sha256": "06bc4bb0aef7658871ef59b4b68a601b3ed92c08175f905763f230bc54ca2307",
    "stems": {
        "bass": "b32083f120a3003a1654a6c9344cb5dfe859778ba357be4b139bbd685af344f8",
        "drums": "6160dd92a92aeb9ce335f092ca1e13d49ffd187298f52da95b9ae2f40b350d3b",
        "other": "bbac507c7a23460ea1f929acfc2fb86093d8fc9d5d3b8b3bfa587c0cc34b084a",
        "vocals": "3335ca6459a745ffb18f4c63eb24e9911c035110f0e01a402e2e998faa35c00d",
    },
}
RESERVE = {"start_s": 93.294263, "end_s": 117.294263}
SR, HOP, N_BINS, FRAME_LENGTH = 22050, 512, 84, 2048


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def record(path: Path) -> dict[str, object]:
    return {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def require_file(path: Path, digest: str, *, expected_bytes: int | None = None) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"Missing required input: {path}")
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise ValueError(f"Unexpected byte count: {path}")
    actual = sha256_file(path)
    if actual != digest:
        raise ValueError(f"Hash mismatch: {path}")
    return record(path)


def extract(stems: dict[str, Path], beat_times: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, object]]:
    """Match evaluate_structure_feedback.extract_features exactly."""
    frame_indices = librosa.time_to_frames(beat_times, sr=SR, hop_length=HOP)
    if frame_indices.size < 2 or np.any(np.diff(frame_indices) <= 0):
        raise ValueError("Beat grid collapses at feature-frame resolution")
    features: dict[str, np.ndarray] = {}
    rms_by_stem: dict[str, np.ndarray] = {}
    for name in STEMS:
        y, _ = librosa.load(stems[name], sr=SR, mono=True)
        cqt = np.abs(librosa.cqt(y, sr=SR, hop_length=HOP, n_bins=N_BINS,
                                  fmin=librosa.note_to_hz("C1")))
        rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP)
        if frame_indices[0] < 0 or frame_indices[-1] >= min(cqt.shape[1], rms.shape[1]):
            raise ValueError(f"Requested grid exceeds {name} feature support")
        synced_features = np.log1p(librosa.util.sync(cqt, frame_indices, aggregate=np.mean, pad=False))
        synced_rms = librosa.util.sync(rms, frame_indices, aggregate=np.mean, pad=False)[0]
        if synced_features.shape != (N_BINS, beat_times.size - 1) or synced_rms.shape != (beat_times.size - 1,):
            raise ValueError(f"Unexpected synced shape for {name}")
        if not np.isfinite(synced_features).all() or not np.isfinite(synced_rms).all() or np.any(synced_rms < 0):
            raise ValueError(f"Nonfinite or negative extracted values for {name}")
        features[name] = synced_features.astype(np.float32, copy=False)
        rms_by_stem[name] = synced_rms.astype(np.float32, copy=False)
    timing = {
        "sample_rate": SR, "hop_length": HOP, "cqt_n_bins": N_BINS,
        "cqt_fmin_hz": float(librosa.note_to_hz("C1")), "rms_frame_length": FRAME_LENGTH,
        "requested_times_s": beat_times.tolist(), "feature_frame_indices": frame_indices.tolist(),
        "feature_frame_times_s": librosa.frames_to_time(frame_indices, sr=SR, hop_length=HOP).tolist(),
        "timing_note": "Window times use cached requested beats; features use floor-quantized frames (< one hop earlier), matching structure-evaluation-03.",
    }
    return features, rms_by_stem, timing


def main(out: Path) -> None:
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {out}")
    started = time.monotonic()
    source = ROOT / "songs/Agnes - MILK.flac"
    stems_json = ROOT / "outputs/Agnes - MILK/stems/stems.json"
    analysis = ROOT / "outputs/Agnes - MILK/analysis/analysis.json"
    protocol = ROOT / "outputs/reviews/arrangement-continuity-registration-01/protocol.md"
    stem_paths = {name: ROOT / f"outputs/Agnes - MILK/stems/{name}.wav" for name in STEMS}

    # Verify the analysis cache bytes/hash before reading its beat grid.
    source_record = require_file(source, EXPECTED["source_sha256"], expected_bytes=EXPECTED["source_bytes"])
    stems_json_record = require_file(stems_json, EXPECTED["stems_json_sha256"])
    analysis_record = require_file(analysis, EXPECTED["analysis_sha256"], expected_bytes=EXPECTED["analysis_bytes"])
    protocol_record = require_file(protocol, EXPECTED["protocol_sha256"])
    metadata = json.loads(stems_json.read_text())
    if metadata.get("meta_sha256") != EXPECTED["stems_meta_sha256"]:
        raise ValueError("Unexpected stems metadata hash field")
    if metadata.get("input", {}).get("sha256") != EXPECTED["source_sha256"]:
        raise ValueError("Stem metadata does not bind the expected source")
    if metadata.get("input", {}).get("path") != "songs/Agnes - MILK.flac":
        raise ValueError("Stem metadata source path differs")
    stem_records = {name: require_file(stem_paths[name], EXPECTED["stems"][name]) for name in STEMS}

    cached = json.loads(analysis.read_text())
    beat_times = np.asarray(cached.get("beats", {}).get("beat_times_s"), dtype=np.float64)
    duration = float(cached.get("meta", {}).get("duration_s"))
    if (beat_times.ndim != 1 or beat_times.size < 2 or not np.isfinite(beat_times).all()
            or np.any(np.diff(beat_times) <= 0) or beat_times[0] < 0 or beat_times[-1] > duration):
        raise ValueError("Invalid cached beat grid")
    for name, path in stem_paths.items():
        if abs(sf.info(path).duration - duration) > 0.05:
            raise ValueError(f"Stem duration differs from cached analysis: {name}")

    out.mkdir(parents=True)
    inputs = out / "inputs"
    inputs.mkdir()
    snapshot = inputs / Path(__file__).name
    shutil.copy2(Path(__file__), snapshot)
    protocol_snapshot = inputs / "protocol.md"
    shutil.copy2(protocol, protocol_snapshot)
    features, rms_by_stem, timing = extract(stem_paths, beat_times)
    np.savez_compressed(out / "features.npz", beat_times_s=beat_times,
                        **{f"{name}_features": features[name] for name in STEMS},
                        **{f"{name}_rms": rms_by_stem[name] for name in STEMS})
    elapsed = time.monotonic() - started
    write_json(out / "timing.json", timing)
    write_json(out / "manifest.json", {
        "schema_version": 1,
        "kind": "songviz-arrangement-continuity-reserve-features",
        "status": "complete-feature-extraction-only",
        "reserve_s": RESERVE,
        "provenance_gap": "The cached analysis beat grid has no independent source hash; association relies on its same-song cache location plus verified source-bound stems metadata. Beat accuracy is unvalidated.",
        "sources": [source_record, stems_json_record, analysis_record, protocol_record, *stem_records.values()],
        "expected_pins": EXPECTED,
        "versions": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "librosa": librosa.__version__, "soundfile": sf.__version__},
        "runtime": {"device": "cpu", "thread_env": {key: __import__("os").environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")}, "elapsed_s": elapsed, "maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
        "settings": {"sample_rate": SR, "hop_length": HOP, "cqt": "abs, n_bins=84, fmin=C1, log1p(mean sync)", "rms": "frame_length=2048, mean sync", "beat_grid": "cached analysis.beats.beat_times_s; floor-quantized feature frames"},
        "inputs": [record(snapshot), record(protocol_snapshot)],
        "outputs": [record(out / "features.npz"), record(out / "timing.json")],
        "limitations": ["No candidate generation, inference, labels, plots, or musical interpretation occurred.", "Full-track features are saved to preserve future whole-track RMS-floor calculation; only later fixed-support reporting may use the reserve."],
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.output.resolve())
