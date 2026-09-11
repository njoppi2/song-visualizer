"""Build an immutable four-variant local-structure comparison review.

This experiment consumes frozen feature/timing/reference/recurrence evidence and
the frozen local-structure control.  It neither trains on nor passes human
annotations to a detector.  It writes only a fresh comparison directory.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
from urllib.parse import quote

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from songviz.ingest import sha256_file
from songviz.local_structure_evaluation import evaluate_local_structure

VARIANTS = ("control", "separate_channels", "sustained_activity", "combined")
CODE = (
    "songviz/local_structure_variants.py", "songviz/local_structure.py",
    "songviz/local_structure_evaluation.py", "songviz/ingest.py",
    "experiments/compare_local_structure.py",
    "experiments/templates/local_structure_comparison.html",
    "experiments/check_local_structure_comparison.cjs",
)


def record(path: Path) -> dict:
    path = path.resolve()
    return {"path": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
            "sha256": sha256_file(path), "bytes": path.stat().st_size}


def verify(path: Path, digest: str) -> None:
    if not path.is_file() or sha256_file(path) != digest:
        raise ValueError(f"Changed or mismatched input: {path}")


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def json_text(value: object) -> str:
    return json.dumps(value, indent=2, allow_nan=False) + "\n"


def embedded_json(value: object) -> str:
    return json.dumps(value, allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def _by_name(records: list[dict], name: str, package: Path) -> dict:
    matches = [r for r in records if Path(r.get("path", "")).name == name]
    if len(matches) != 1:
        raise ValueError(f"Manifest has no unique {name} record in {package}")
    return matches[0]


def _manifest_records(package: Path, manifest: dict, key: str) -> list[Path]:
    records = manifest.get(key)
    if not isinstance(records, list) or not records:
        raise ValueError(f"Missing {key} fingerprints in {package}/manifest.json")
    paths = []
    for item in records:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not isinstance(item.get("sha256"), str):
            raise ValueError(f"Invalid {key} fingerprint in {package}/manifest.json")
        path = Path(item["path"])
        if not path.is_absolute():
            path = ROOT / path
        verify(path, item["sha256"])
        paths.append(path.resolve())
    return paths


def _read_parent(parent: Path, audio_review: Path) -> tuple[dict, dict, dict, dict, dict, dict, np.ndarray, dict, dict, float, list[Path]]:
    manifest_path = parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("kind") != "songviz-structural-development-evaluation":
        raise ValueError("Expected the dimension-separated structural evaluation package")
    # Verify all declared parent evidence, then select only the small, already
    # frozen artifacts needed by this experiment.  This catches a stale parent
    # package without copying large caches or recurrence data into the review.
    consumed = [manifest_path, *_manifest_records(parent, manifest, "outputs"),
                *_manifest_records(parent, manifest, "sources")]
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list):
        raise ValueError("Parent manifest lacks output fingerprints")
    for name in ("reference.json", "features.npz", "timing.json", "recurrence.json", "legacy-sections.json"):
        item = _by_name(outputs, name, parent)
        verify(parent / name, item["sha256"])
    reference = json.loads((parent / "reference.json").read_text())
    timing = json.loads((parent / "timing.json").read_text())
    recurrence = json.loads((parent / "recurrence.json").read_text())
    legacy = json.loads((parent / "legacy-sections.json").read_text())

    audio_manifest_path = audio_review / "manifest.json"
    source_records = [r for r in manifest.get("sources", []) if isinstance(r, dict) and r.get("path")]
    linked_audio = [r for r in source_records if (ROOT / r["path"]).resolve() == audio_manifest_path.resolve()]
    if len(linked_audio) != 1:
        raise ValueError("Audio review is not the fingerprinted structural source")
    verify(audio_manifest_path, linked_audio[0]["sha256"])
    audio_manifest = json.loads(audio_manifest_path.read_text())
    for name in ("original.wav", "review.json"):
        item = _by_name(audio_manifest.get("outputs", []), name, audio_review)
        verify(audio_review / name, item["sha256"])
        consumed.append(audio_review / name)
    consumed.append(audio_manifest_path)
    verify(audio_review / "original.wav", reference["source"]["audio_sha256"])
    info = sf.info(audio_review / "original.wav")
    if abs(info.duration - reference["source"]["duration_s"]) > 1e-6:
        raise ValueError("Audio duration differs from reference")
    raw = [r for r in source_records if r.get("sha256") == reference["source"].get("source_audio_sha256")]
    if len(raw) != 1:
        raise ValueError("Original source fingerprint not unique")
    source = ROOT / raw[0]["path"]
    verify(source, raw[0]["sha256"])
    consumed.append(source)
    audio_review_json = json.loads((audio_review / "review.json").read_text())
    if abs(audio_review_json.get("duration_s", -1) - info.duration) > 1e-6:
        raise ValueError("Review waveform duration differs from audio")
    with np.load(parent / "features.npz", allow_pickle=False) as cache:
        if "beat_times_s" not in cache.files:
            raise ValueError("Missing beat times")
        bt = cache["beat_times_s"]
        names = sorted(k[:-9] for k in cache.files if k.endswith("_features"))
        expected = {"beat_times_s"} | {n + "_features" for n in names} | {n + "_rms" for n in names}
        if not names or set(cache.files) != expected:
            raise ValueError("Unexpected or incomplete feature cache schema")
        features = {name: cache[name + "_features"] for name in names}
        energy = {name: cache[name + "_rms"] for name in names}
    if not np.array_equal(bt, np.asarray(timing.get("requested_times_s"))):
        raise ValueError("Feature and timing grids differ")
    if bt.ndim != 1 or bt.size < 2 or bt[0] < 0 or bt[-1] > info.duration:
        raise ValueError("Beat grid lies outside the audio")
    for result in recurrence.get("results", []):
        for span in result.get("spans", []):
            a, b = span.get("start_beat"), span.get("end_beat")
            if (isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, int) or not isinstance(b, int)
                    or not 0 <= a < b < len(bt) or span.get("start_s") != bt[a] or span.get("end_s") != bt[b]):
                raise ValueError("Recurrence windows differ from the feature beat grid")
    return manifest, reference, timing, recurrence, legacy, audio_review_json, bt, features, energy, info.duration, consumed


def _read_control(control: Path) -> tuple[dict, list[Path]]:
    manifest_path = control / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("kind") != "songviz-local-structure-review":
        raise ValueError("Expected frozen local-structure review control")
    consumed = [manifest_path]
    # A frozen control is evidence only if both its declared outputs and the
    # source snapshot that produced them still verify.  Current source files are
    # deliberately not compared to those historical snapshots.
    consumed.extend(_manifest_records(control, manifest, "outputs"))
    snapshots = _manifest_records(control, manifest, "input_snapshots")
    if not any(path.as_posix().endswith("inputs/songviz/local_structure.py") for path in snapshots):
        raise ValueError("Frozen control lacks its detector source snapshot")
    consumed.extend(snapshots)
    for name in ("predictions.json", "evaluation.json", "review.json"):
        item = _by_name(manifest["outputs"], name, control)
        verify(control / name, item["sha256"])
    template = control / "inputs/experiments/templates/local_structure_review.html"
    review = control / "review.json"
    index = control / "index.html"
    if not index.is_file():
        raise ValueError("Frozen control lacks index.html")
    expected_page = template.read_text().replace("{{REVIEW_JSON}}", embedded_json(json.loads(review.read_text()))).replace(
        "{{MANIFEST_SHA}}", sha256_file(manifest_path))
    if index.read_text() != expected_page:
        raise ValueError("Frozen control index.html does not match its snapshotted template and manifest")
    consumed.append(index)
    return manifest, consumed


def _variant_detector():
    try:
        module = importlib.import_module("songviz.local_structure_variants")
        return module.detect_local_structure_variant, module.LocalStructureVariantConfig
    except (ImportError, AttributeError) as exc:
        raise RuntimeError("Comparison requires songviz.local_structure_variants from the bounded detector worker") from exc


def _time_set(predictions: dict) -> set[float]:
    return {float(event["time_s"]) for event in predictions.get("changes", [])}


def _comparison_summary(variants: dict[str, dict]) -> dict:
    control_times = _time_set(variants["control"]["predictions"])
    rows = []
    for layer in variants["control"]["evaluation"].get("layers", []):
        for boundary in layer.get("boundary_rows", []):
            item = {"layer_id": layer.get("id"), "layer_name": layer.get("name"), "time_s": boundary["time_s"],
                    "identity_relation": boundary.get("identity_relation"), "variation_change": boundary.get("variation_change"),
                    "nearest_legacy_signed_delta_s": (boundary.get("nearest_legacy") or {}).get("delta_s"),
                    "nearest_signed_deltas_s": {}}
            for variant in VARIANTS:
                matching = next((candidate for candidate in variants[variant]["evaluation"].get("layers", [])
                                 if candidate.get("id") == layer.get("id")), {})
                row = next((candidate for candidate in matching.get("boundary_rows", [])
                            if candidate.get("time_s") == boundary["time_s"]), {})
                nearest = row.get("nearest_change")
                item["nearest_signed_deltas_s"][variant] = None if nearest is None else nearest.get("delta_s")
            rows.append(item)
    by_variant = {}
    def transition_evidence(evaluation: dict) -> list[dict]:
        return [{"id": layer.get("id"), "transition_rows": layer.get("transition_rows", []),
                 "predicted_transition_rows": layer.get("predicted_transition_rows", [])}
                for layer in evaluation.get("layers", [])]

    control_transition_rows = transition_evidence(variants["control"]["evaluation"])
    for variant in VARIANTS:
        predictions, evaluation = variants[variant]["predictions"], variants[variant]["evaluation"]
        times = _time_set(predictions)
        transitions = evaluation.get("layers", [])
        by_variant[variant] = {
            "candidate_counts": evaluation.get("counts"),
            "all_new_candidates_nearest_annotation": [
                {"layer_id": layer.get("id"), "change_rows": layer.get("predicted_change_rows", [])}
                for layer in transitions
            ],
            "exact_time_set_difference_from_control": {
                "gained_times_s": sorted(times - control_times), "lost_times_s": sorted(control_times - times),
                "meaning": "Exact serialized candidate timestamps only; this is not musical-event pairing or an accuracy claim.",
            },
            "transition_overlap_unchanged_from_control": transition_evidence(evaluation) == control_transition_rows,
        }
    return {"human_boundary_nearest_signed_deltas_s": rows, "variants": by_variant,
            "limitations": [
                "All nearest distances are descriptive and many-to-one; no tolerance, acceptance score, precision or recall is selected.",
                "Candidates without an annotation remain unknown, not false positives.",
                "Raw human labels and analyst interpretation fields are displayed separately; neither enters prediction generation.",
            ]}


def _page_variants(variants: dict[str, dict]) -> dict[str, dict]:
    """Return only evidence rendered by the HTML review.

    Complete per-scale curves and phrase recurrence diagnostics remain in each
    ``variants/<name>/*.json`` file.  Copying them four times into ``review.json``
    would make an audio-review page needlessly large without making its timeline
    more inspectable.
    """
    page: dict[str, dict] = {}
    for name, result in variants.items():
        prediction = deepcopy(result["predictions"])
        for key in ("curves", "channel_curves", "activity_curves"):
            prediction.pop(key, None)
        evaluation = deepcopy(result["evaluation"])
        evaluation.pop("recurrence_context", None)
        page[name] = {"predictions": prediction, "evaluation": evaluation}
    return page


def report_markdown(data: dict) -> str:
    def cell(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["# Fixed local-structure ablation comparison", "", cell(data["song_title"]), "",
             "Four label-free detector variants share one frozen acoustic cache and beat grid. This report is descriptive: it makes no acceptance, accuracy, or selected-tolerance claim.",
             "", "## Proposal density and exact timestamp-set differences", "",
             "| Variant | Changes | Changes/min | Transitions | Transitions/min | Gained exact times vs control | Lost exact times vs control |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for name in VARIANTS:
        item = data["summary"]["variants"][name]
        counts = item["candidate_counts"]
        change, transition = counts["changes"], counts["transitions"]
        differences = item["exact_time_set_difference_from_control"]
        joined = lambda values: ", ".join(f"{v:.3f}" for v in values) or "—"
        lines.append(f"| {name} | {change['count']} | {change['density_per_minute']:.3f} | {transition['count']} | {transition['density_per_minute']:.3f} | {joined(differences['gained_times_s'])} | {joined(differences['lost_times_s'])} |")
    lines += ["", "Exact timestamp sets are an audit convenience, not musical-event matching.", "", "## Every human boundary: nearest signed delta by variant", "",
             "| Layer | Boundary (s) | Raw identity relation | Analyst variation change | Legacy | Control | Separate channels | Sustained activity | Combined |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in data["summary"]["human_boundary_nearest_signed_deltas_s"]:
        delta = row["nearest_signed_deltas_s"]
        value = lambda key: "unknown" if delta[key] is None else f"{delta[key]:.3f}"
        legacy = "unknown" if row["nearest_legacy_signed_delta_s"] is None else f"{row['nearest_legacy_signed_delta_s']:.3f}"
        lines.append(f"| {cell(row['layer_name'])} | {row['time_s']:.3f} | {cell(row['identity_relation'])} | {cell(row['variation_change'])} | {legacy} | {value('control')} | {value('separate_channels')} | {value('sustained_activity')} | {value('combined')} |")
    lines += ["", "## Evidence access", "", "Each variant's complete predictions and evaluation are under `variants/<name>/`. Every new candidate's nearest annotated-boundary distance remains in its evaluation JSON; unannotated candidates are unknown rather than false positives. Transition overlap records are unchanged only when the full per-layer transition evaluation is exactly equal to the frozen-control regeneration.", "", "## Limits", ""]
    lines += ["- " + item for item in data["notes"] + data["summary"]["limitations"]]
    return "\n".join(lines) + "\n"


def build(*, parent: Path, control: Path, audio_review: Path, out: Path) -> None:
    parent, control, audio_review, out = (path.resolve() for path in (parent, control, audio_review, out))
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    if any(source == out or source in out.parents for source in (parent, control, audio_review)):
        raise ValueError("Output must be separate from input packages")
    _, reference, _, recurrence, legacy, audio_review_json, bt, features, energy, duration, parent_inputs = _read_parent(parent, audio_review)
    control_manifest, control_inputs = _read_control(control)
    code_paths = [ROOT / relative for relative in CODE]
    for path in code_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required comparison source is missing: {path}")
    before = {path.resolve(): record(path) for path in parent_inputs + control_inputs + code_paths}
    detect, config_type = _variant_detector()
    predictions: dict[str, dict] = {}
    evaluations: dict[str, dict] = {}
    for variant in VARIANTS:
        # The detector receives only numeric cached acoustics and timings.
        prediction = detect(features, energy, bt, variant=variant, config=config_type())
        json_text(prediction)  # validates finite, JSON-serializable output before writes
        predictions[variant] = prediction
        evaluations[variant] = evaluate_local_structure(reference, prediction, legacy["sections"], recurrence["results"])
    frozen_control = (control / "predictions.json").read_text()
    if json_text(predictions["control"]) != frozen_control:
        raise ValueError("Regenerated control is not exact JSON-serialized equality with frozen control predictions")
    frozen_transitions = predictions["control"].get("transitions")
    if any(json_text(predictions[name].get("transitions")) != json_text(frozen_transitions) for name in VARIANTS):
        raise ValueError("All variant energy-dip intervals must be exact frozen-control equality")
    variants = {name: {"predictions": predictions[name], "evaluation": evaluations[name]} for name in VARIANTS}
    relative_audio = quote(Path(os.path.relpath(audio_review / "original.wav", out)).as_posix(), safe="/")
    # Keep this browser payload bounded.  The separate variant JSON files retain
    # the complete curves and recurrence evidence used to build the summaries.
    page_variants = _page_variants(variants)
    data = {"schema_version": 1, "kind": "songviz-local-structure-comparison", "song_title": reference["source"]["song_title"],
            "duration_s": duration, "audio_path": relative_audio, "audio_sha256": reference["source"]["audio_sha256"],
            "reference": reference, "legacy_sections": legacy["sections"], "waveform": audio_review_json.get("waveform", []),
            "variants": page_variants, "summary": _comparison_summary(variants),
            "notes": ["Raw human labels are evidence; analyst variation and transition fields are a separate interpretation.",
                      "No labels, reference spans, recurrence values, or legacy cuts are supplied to the detector.",
                      "Native original audio is referenced by relative URL only; this package does not copy it.",
                      "The control comparison is against its frozen source snapshot and manifest/output hashes. Current detector files may legitimately evolve."],
            "frozen_control_manifest_sha256": sha256_file(control / "manifest.json"),
            "frozen_control_outputs": control_manifest.get("outputs")}
    encoded = embedded_json(data)
    for item in before.values():
        path = Path(item["path"])
        if not path.is_absolute(): path = ROOT / path
        verify(path, item["sha256"])
    out.mkdir(parents=True)
    for relative in CODE:
        destination = out / "inputs" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    for source, destination in ((parent / "manifest.json", out / "inputs" / "parent-manifest.json"),
                                (control / "manifest.json", out / "inputs" / "control-manifest.json"),
                                (audio_review / "manifest.json", out / "inputs" / "audio-manifest.json")):
        shutil.copy2(source, destination)
    for name in VARIANTS:
        (out / "variants" / name).mkdir(parents=True, exist_ok=True)
        write_json(out / "variants" / name / "predictions.json", predictions[name])
        write_json(out / "variants" / name / "evaluation.json", evaluations[name])
    write_json(out / "review.json", data)
    (out / "report.md").write_text(report_markdown(data))
    template = (out / "inputs" / "experiments/templates/local_structure_comparison.html").read_text()
    (out / "index.html").write_text(template.replace("{{REVIEW_JSON}}", encoded).replace("{{REVIEW_SHA}}", sha256_file(out / "review.json")))
    for item in before.values():
        path = Path(item["path"])
        if not path.is_absolute(): path = ROOT / path
        verify(path, item["sha256"])
    manifest = {"schema_version": 1, "kind": "songviz-local-structure-comparison", "created_utc": datetime.now(timezone.utc).isoformat(),
                "sources": list(before.values()), "input_snapshots": [record(path) for path in sorted((out / "inputs").rglob("*")) if path.is_file()],
                "outputs": [record(path) for path in sorted(out.rglob("*")) if path.is_file() and path.name != "manifest.json"],
                "variants": list(VARIANTS), "scope": "Fixed local-structure ablation; label-free prediction and descriptive evaluation only.",
                "numpy_version": np.__version__, "python_version": platform.python_version(),
                "page_integrity": "index.html is derived from the snapshotted template and review.json SHA-256 embedded in the page; the manifest records the page hash, avoiding a self-referential manifest hash."}
    write_json(out / "manifest.json", manifest)
    print(f"Ready: {out}/index.html", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=ROOT / "outputs/reviews/structure-evaluation-03")
    parser.add_argument("--control", type=Path, default=ROOT / "outputs/reviews/local-structure-02")
    parser.add_argument("--audio-review", type=Path, default=ROOT / "outputs/reviews/structure-review-03")
    parser.add_argument("--out", type=Path, required=True)
    build(**vars(parser.parse_args()))
