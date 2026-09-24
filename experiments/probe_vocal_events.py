#!/usr/bin/env python3
"""Frozen local YAMNet vocal-event screen; input recipes are fixed in doc 30.

The module deliberately keeps inference and replay simple.  It records every
model support, full score row and same-support RMS so the saved package can be
reviewed without reopening audio or invoking TFLite.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

RATE = 16_000
WINDOW = 15_600
HOP = 7_680
SCORE_COUNT = 521
THRESHOLD = 0.5
SPEECH_IDS = (0, 24, 31)
LAUGHTER_IDS = (13, 14, 15, 16, 17, 18)
MODEL_SHA256 = "4d8b4a53282dc83ef04e3e7dbc4fbc98082e34e44ed798e16c3a0cdd4c584faf"
MODEL_BYTES = 4_126_810
CLASS_MAP_SHA256 = "cdf24d193e196d9e95912a2667051ae203e92a2ba09449218ccb40ef787c6df2"
ORIGINAL_SHA256 = "657af9333edafbb6cf8cc573446650773d1169b0117c784106d1895cf802cb44"
DEMUCS_VOCALS_SHA256 = "737a2305facfc8a570ae445675014c8e900e5bbfacdd25550a1be5155b9be3a4"
ROFORMER_VOCALS_SHA256 = "3ae6a47fd86dc398ad1ddf55697f61a39b76ef548612b5c853c7766e40ec0fb8"
ROFORMER_OTHER_SHA256 = "14d418a527f2882441467b0053d46bd6dc112bcc66852829cc1c7de3eab73612"
RUNTIME_MANIFEST_SHA256 = "dcef6475cf23a7861268ecfada5a847dfcebd8d7fb4dcf61d1beac75b6a7a646"
RUNTIME_UPSTREAM_SHA256 = "b2ee1ee69283e1140ada66eeb4c361575d0bb52df297d66228552b0a8523b9b9"
ROFORMER_MANIFEST_SHA256 = "caa24ea0c30bf3b7637e2ce06dad221da8a78b7bc70cdc3d7fb48b485a68aa5c"
LISTENER_RECORD_SHA256 = "b9f09d7da82f81d125cb61ca8fefb86762834966deae3020a803d6269a3e96db"
BENCHMARK_MANIFEST_SHA256 = "684b38314825c61c743459337631085abee87e12bd8157aab2b6fef952ff4f97"
ROLE_CONTEXT_MANIFEST_SHA256 = "129eab2556bc43bcef9a351015eed4847ca4b81b3555fe1afbeb615943af6cbe"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise ValueError(f"missing {label}: {path}")
    actual = digest(path)
    if actual != expected:
        raise ValueError(f"provenance failure for {label}: expected {expected}, got {actual}")


def windows(length: int) -> tuple[list[tuple[int, int]], int]:
    """Complete [start,end) waveform windows and explicit uncovered tail."""
    if length < 0:
        raise ValueError("waveform length must be non-negative")
    spans = [(start, start + WINDOW) for start in range(0, max(0, length - WINDOW + 1), HOP)]
    covered_end = spans[-1][1] if spans else 0
    return spans, length - covered_end


def support_mask(spans: list[tuple[int, int]], clip_start_s: float, support: tuple[float, float]) -> np.ndarray:
    """True only where a whole model waveform support lies inside support."""
    lo, hi = support
    return np.asarray([
        clip_start_s + start / RATE >= lo and clip_start_s + end / RATE <= hi
        for start, end in spans
    ], dtype=bool)


def consecutive(values: np.ndarray, mask: np.ndarray) -> bool:
    """Find two adjacent evaluated windows; a masked gap breaks a run."""
    run = 0
    for value, included in zip(values, mask):
        if included and bool(value):
            run += 1
            if run >= 2:
                return True
        else:
            run = 0
    return False


def fraction(values: np.ndarray, mask: np.ndarray) -> float | None:
    count = int(mask.sum())
    return None if count == 0 else float(np.count_nonzero(values & mask) / count)


def family_scores(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if scores.ndim != 2 or scores.shape[1] != SCORE_COUNT:
        raise ValueError(f"scores must have shape [n,{SCORE_COUNT}]")
    if not np.isfinite(scores).all():
        raise ValueError("nonfinite scores")
    return scores[:, SPEECH_IDS].max(axis=1), scores[:, LAUGHTER_IDS].max(axis=1)


def fixed_gate(scores: np.ndarray, spans: list[tuple[int, int]], clip_start_s: float) -> dict:
    """Apply doc-30's fixed screen, returning abstention when supports are empty."""
    speech, laughter = family_scores(scores)
    early = support_mask(spans, clip_start_s, (119.0, 123.72))
    late = support_mask(spans, clip_start_s, (125.45, 132.0))
    speech_positive, laughter_positive = speech >= THRESHOLD, laughter >= THRESHOLD
    early_speech, late_speech = fraction(speech_positive, early), fraction(speech_positive, late)
    early_laughter, late_laughter = fraction(laughter_positive, early), fraction(laughter_positive, late)
    evaluable = None not in (early_speech, late_speech, early_laughter, late_laughter)
    early_run = consecutive(speech_positive, early)
    late_run = consecutive(laughter_positive, late)
    shift = (late_laughter > early_laughter and late_speech < early_speech) if evaluable else False
    return {
        "threshold": THRESHOLD, "early_support": [119.0, 123.72], "late_support": [125.45, 132.0],
        "early_window_count": int(early.sum()), "late_window_count": int(late.sum()),
        "early_speech_singing_positive_fraction": early_speech,
        "late_speech_singing_positive_fraction": late_speech,
        "early_laughter_positive_fraction": early_laughter,
        "late_laughter_positive_fraction": late_laughter,
        "early_two_consecutive_speech_singing": early_run,
        "late_two_consecutive_laughter": late_run,
        "shift_fractions": shift,
        "narrow_behavior_support": bool(evaluable and early_run and late_run and shift),
        "status": "evaluated" if evaluable else "abstain-insufficient-contained-support",
    }


def validate_class_map(path: Path) -> dict:
    verify(path, CLASS_MAP_SHA256, "class map")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != SCORE_COUNT:
        raise ValueError(f"class map must have {SCORE_COUNT} rows")
    required = {0: "Speech", 13: "Laughter", 24: "Singing", 31: "Rapping"}
    for index, name in required.items():
        row = rows[index]
        if int(row["index"]) != index or row["display_name"] != name:
            raise ValueError(f"class map mismatch at {index}: expected {name}")
    return {"path": str(path), "sha256": digest(path), "count": len(rows), "required_ids": required, "display_names": [x["display_name"] for x in rows]}


def validate_runtime(runtime: Path, model_dir: Path) -> dict:
    manifest_path = runtime / "manifest.json"
    upstream_path = runtime / "upstream.json"
    result_path = runtime / "synthetic-result.json"
    for path in (manifest_path, upstream_path, result_path):
        if not path.is_file():
            raise ValueError(f"runtime record missing {path.name}")
    verify(manifest_path, RUNTIME_MANIFEST_SHA256, "accepted runtime manifest")
    verify(upstream_path, RUNTIME_UPSTREAM_SHA256, "accepted runtime upstream record")
    manifest, upstream, result = (json.loads(x.read_text()) for x in (manifest_path, upstream_path, result_path))
    if manifest.get("kind") != "songviz-yamnet-local-runtime-screen" or result.get("song_or_stem_opened") is not False:
        raise ValueError("runtime record is not the completed synthetic-only YAMNet screen")
    if manifest.get("synthetic_execution", {}).get("result_sha256") != digest(result_path):
        raise ValueError("runtime manifest does not bind synthetic result")
    model = model_dir / "yamnet.tflite"
    verify(model, MODEL_SHA256, "model")
    if model.stat().st_size != MODEL_BYTES or upstream.get("model", {}).get("sha256") != MODEL_SHA256:
        raise ValueError("model runtime pins do not match fixed asset")
    class_map = validate_class_map(model_dir / "yamnet_class_map.csv")
    with zipfile.ZipFile(model) as archive:
        embedded = archive.read("yamnet_label_list.txt").decode("utf-8").splitlines()
    if embedded != class_map["display_names"]:
        raise ValueError("embedded model labels do not exactly match pinned class map")
    del class_map["display_names"]
    observed = result.get("interpreter", {})
    if observed.get("input_details", [{}])[0].get("shape") != [WINDOW] or observed.get("output_details", [{}])[0].get("shape") != [1, SCORE_COUNT]:
        raise ValueError("runtime interpreter shape pins do not match model contract")
    return {"runtime_files": {p.name: digest(p) for p in (manifest_path, upstream_path, result_path)}, "model": {"path": str(model), "sha256": digest(model), "bytes": model.stat().st_size}, "class_map": class_map}


def validate_waveform(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 1 or not np.isfinite(samples).all() or np.any(np.abs(samples) > 1.0):
        raise ValueError("resampled mono input must be finite float32 within [-1,1]")
    return samples


def make_window_records(spans: list[tuple[int, int]], clip_start_s: float, rms: np.ndarray, scores: np.ndarray) -> list[dict]:
    early, late = support_mask(spans, clip_start_s, (119, 123.72)), support_mask(spans, clip_start_s, (125.45, 132))
    return [{"index": i, "start_sample": a, "end_sample": b, "start_s": clip_start_s + a / RATE,
             "end_s": clip_start_s + b / RATE, "early_contained": bool(early[i]),
             "late_contained": bool(late[i]), "rms": float(rms[i]),
             "selected_scores": {"speech": float(scores[i, 0]), "singing": float(scores[i, 24]),
                                 "rapping": float(scores[i, 31]), "laughter": float(scores[i, 13]),
                                 "laughter_subtypes_14_18": [float(x) for x in scores[i, 14:19]]}}
            for i, (a, b) in enumerate(spans)]


def infer(interpreter, data: np.ndarray, spans: list[tuple[int, int]]) -> np.ndarray:
    """Run a supplied allocated TFLite interpreter; imported lazily by CLI only."""
    details, output = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    rows = []
    for start, end in spans:
        interpreter.set_tensor(details["index"], data[start:end])
        interpreter.invoke()
        row = np.asarray(interpreter.get_tensor(output["index"]), dtype=np.float32).reshape(-1)
        if row.shape != (SCORE_COUNT,):
            raise ValueError("model did not return 521 scores")
        if not np.isfinite(row).all() or np.any(row < 0) or np.any(row > 1):
            raise ValueError("model returned nonfinite or out-of-range score")
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)


def write_report(output: Path, conditions: list[dict], provenance: dict) -> None:
    gates = {item["input_id"]: item.get("gate") for item in conditions}
    gain_fragility = gates["input-03"]["narrow_behavior_support"] != gates["input-08"]["narrow_behavior_support"]
    payload = {"schema_version": 1, "kind": "songviz-vocal-event-saved-score-report", "provenance": provenance, "conditions": conditions,
               "gain_comparison": {"inputs": ["input-03", "input-08"], "gain_fragility": gain_fragility,
                                   "meaning": "Different fixed-gate outcomes flag gain fragility; neither outcome is an automatic benchmark pass."},
               "limitations": "Scores are AudioSet event scores. They do not establish musical leadership, importance, section identity, or verse ending. Tails and uncovered/straddling windows are retained, not negative evidence."}
    (output / "report.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Local vocal-event saved-score report\n", "This is a fixed operational screen, not an automatic benchmark pass.\n"]
    for item in conditions:
        gate = item.get("gate")
        lines.append(f"## {item['input_id']}\n\nWindows: {item['window_count']}; uncovered tail: {item['uncovered_tail_samples']} samples.\n")
        if gate: lines.append(f"Gate: `{gate['status']}`; narrow behavior support: `{gate['narrow_behavior_support']}`.\n")
    lines.append(f"## Gain comparison\n\n`input-03` versus `input-08` gain fragility: `{gain_fragility}`.\n")
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def load_resampled(path: Path, start_s: float, end_s: float) -> np.ndarray:
    """Decode exactly the declared source range, average channels, then resample."""
    import soundfile as sf
    from scipy.signal import resample_poly
    info = sf.info(path)
    expected_frames = round((end_s - start_s) * info.samplerate)
    data, sample_rate = sf.read(path, start=round(start_s * info.samplerate), frames=expected_frames,
                                always_2d=True, dtype="float32")
    if len(data) != expected_frames:
        raise ValueError(f"decoded source has {len(data)} frames; expected {expected_frames}")
    mono = data.mean(axis=1, dtype=np.float32)
    divisor = int(np.gcd(sample_rate, RATE))
    resampled = validate_waveform(resample_poly(mono, RATE // divisor, sample_rate // divisor,
                                                window=("kaiser", 5.0), padtype="constant").astype(np.float32))
    expected_samples = round((end_s - start_s) * RATE)
    if len(resampled) != expected_samples:
        raise ValueError(f"resampled input has {len(resampled)} samples; expected {expected_samples}")
    return resampled


def verify_sources(repo: Path) -> dict:
    original = repo / "songs/Gorillaz - Feel Good Inc (featuring De La Soul).flac"
    demucs = repo / "outputs/Gorillaz - Feel Good Inc (featuring De La Soul)/stems/vocals.wav"
    control = repo / "outputs/reviews/stem-separation-control-01"
    benchmark_manifest = repo / "outputs/reviews/development-benchmark-02/manifest.json"
    role_manifest = repo / "outputs/reviews/role-context-02/manifest.json"
    manifest_path, listener = control / "manifest.json", control / "candidate-manipulation-check.json"
    verify(original, ORIGINAL_SHA256, "original mix")
    verify(demucs, DEMUCS_VOCALS_SHA256, "Demucs vocals")
    verify(manifest_path, ROFORMER_MANIFEST_SHA256, "accepted RoFormer manifest")
    verify(listener, LISTENER_RECORD_SHA256, "completed listener-control record")
    verify(benchmark_manifest, BENCHMARK_MANIFEST_SHA256, "accepted benchmark manifest")
    verify(role_manifest, ROLE_CONTEXT_MANIFEST_SHA256, "accepted role-context manifest")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("blind_check", {}).get("status") != "complete; candidate-other=no and resum-positive=yes":
        raise ValueError("listener control is not recorded as completed/passed")
    outputs = manifest["run"]["outputs"]
    vocals = control / outputs["vocals"]["path"]
    other = control / outputs["other"]["path"]
    verify(vocals, ROFORMER_VOCALS_SHA256, "RoFormer vocals")
    verify(other, ROFORMER_OTHER_SHA256, "RoFormer instrumental")
    return {"original": original, "demucs_vocals": demucs, "roformer_vocals": vocals, "roformer_other": other,
            "hashes": {"original": ORIGINAL_SHA256, "demucs_vocals": DEMUCS_VOCALS_SHA256,
                       "roformer_vocals": ROFORMER_VOCALS_SHA256, "roformer_other": ROFORMER_OTHER_SHA256,
                       "roformer_manifest": ROFORMER_MANIFEST_SHA256, "listener_record": LISTENER_RECORD_SHA256,
                       "benchmark_manifest": BENCHMARK_MANIFEST_SHA256, "role_context_manifest": ROLE_CONTEXT_MANIFEST_SHA256}}


def prepare_inputs(repo: Path) -> tuple[list[dict], dict]:
    """Create all eight fixed arrays. No semantic metadata enters inference."""
    source = verify_sources(repo)
    recipes = [
        ("input-01", source["original"], 16.0, 26.0, 16.0), ("input-02", source["original"], 74.0, 85.0, 74.0),
        ("input-03", source["original"], 119.0, 132.0, 119.0), ("input-04", source["original"], 57.0, 70.0, 57.0),
        ("input-05", source["demucs_vocals"], 119.0, 132.0, 119.0),
        # RoFormer files are themselves exact 119--132 source excerpts.
        ("input-06", source["roformer_vocals"], 0.0, 13.0, 119.0), ("input-07", source["roformer_other"], 0.0, 13.0, 119.0),
    ]
    prepared = []
    for input_id, path, start, end, global_start in recipes:
        prepared.append({"input_id": input_id, "samples": load_resampled(path, start, end), "clip_start_s": global_start,
                         "source_path": str(path.relative_to(repo)), "source_range_s": [start, end]})
    primary = prepared[2]["samples"]
    prepared.append({"input_id": "input-08", "samples": (np.float32(.5) * primary).astype(np.float32), "clip_start_s": 119.0,
                     "source_path": "derived:input-03", "source_range_s": [119.0, 132.0], "derivation": "exact float32 0.5 * input-03"})
    return prepared, source["hashes"]


def _validate_live_interpreter(interpreter) -> None:
    input_detail, output_detail = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    if tuple(input_detail["shape"]) != (WINDOW,) or input_detail["dtype"] is not np.float32:
        raise ValueError("live interpreter input is not float32 [15600]")
    if tuple(output_detail["shape"]) != (1, SCORE_COUNT) or output_detail["dtype"] is not np.float32:
        raise ValueError("live interpreter output is not float32 [1,521]")


def validate_environment() -> dict:
    """Bind the live process to the isolated runtime's declared package pins."""
    expected = {"numpy": "1.26.4", "scipy": "1.15.3", "soundfile": "0.13.1", "tflite-runtime": "2.14.0"}
    actual = {"numpy": np.__version__}
    for distribution in ("scipy", "soundfile", "tflite-runtime"):
        actual[distribution] = importlib.metadata.version(distribution)
    if actual != expected:
        raise ValueError(f"live environment pins differ from isolated runtime: {actual}")
    return {"packages": actual, "python": sys.version, "platform": platform.platform()}


def run_screen(repo: Path, runtime: Path, protocol: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if not protocol.is_file():
        raise ValueError("protocol snapshot is missing")
    provenance = validate_runtime(runtime, repo / ".songviz/vocal-event-models/yamnet-01")
    provenance["environment"] = validate_environment()
    prepared, source_hashes = prepare_inputs(repo)
    # TFLite is imported here, after every source/provenance gate has passed.
    from tflite_runtime.interpreter import Interpreter
    interpreter = Interpreter(model_path=provenance["model"]["path"], num_threads=2)
    interpreter.allocate_tensors()
    _validate_live_interpreter(interpreter)
    output.mkdir(parents=True)
    conditions = []
    for item in prepared:
        data = item["samples"]; spans, tail = windows(len(data))
        if not spans: raise ValueError(f"{item['input_id']} has no complete model window")
        score = infer(interpreter, data, spans)
        rms = np.asarray([np.sqrt(np.mean(np.square(data[a:b], dtype=np.float32), dtype=np.float32)) for a, b in spans], dtype=np.float32)
        np.save(output / f"{item['input_id']}-input.npy", data)
        np.save(output / f"{item['input_id']}-scores.npy", score)
        condition = {key: value for key, value in item.items() if key != "samples"}
        condition.update({"input_file": f"{item['input_id']}-input.npy", "scores_file": f"{item['input_id']}-scores.npy",
                          "input_sha256": digest(output / f"{item['input_id']}-input.npy"), "scores_sha256": digest(output / f"{item['input_id']}-scores.npy"),
                          "window_count": len(spans), "uncovered_tail_samples": tail,
                          "windows": make_window_records(spans, item["clip_start_s"], rms, score),
                          "gate": fixed_gate(score, spans, item["clip_start_s"]) if item["input_id"] in {"input-03", "input-05", "input-06", "input-07", "input-08"} else None})
        if item["input_id"] == "input-07":
            condition["operational_control_pass"] = not condition["gate"]["late_two_consecutive_laughter"]
            condition["control_limit"] = "Operational control only; it does not establish specificity or absence of a true event."
        elif item["input_id"] in {"input-05", "input-06"}:
            condition["interpretation_limit"] = "Separated-vocal sensitivity condition; it cannot replace primary full-mix evidence."
        conditions.append(condition)
    provenance["source_hashes"] = source_hashes
    provenance["protocol_sha256"] = digest(protocol)
    provenance["runner_sha256"] = digest(Path(__file__))
    shutil.copy2(Path(__file__), output / "probe_vocal_events.py")
    shutil.copy2(protocol, output / "protocol_snapshot.md")
    write_report(output, conditions, provenance)
    manifest = {"schema_version": 1, "kind": "songviz-vocal-event-screen", "status": "complete-all-eight-conditions",
                "provenance": provenance, "outputs": {path.name: digest(path) for path in sorted(output.iterdir()) if path.is_file() and path.name != "manifest.json"}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        run_screen(args.repo.resolve(), args.runtime.resolve(), args.protocol.resolve(), args.output.resolve())
    except (OSError, ValueError, KeyError, FileExistsError) as exc:
        print(f"probe preflight failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
