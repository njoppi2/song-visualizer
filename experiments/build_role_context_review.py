"""Build an immutable, continuous all-stem role-context review package.

The detector receives only cached acoustic features, RMS and beat times.  The
four guided-listening responses are preserved verbatim for evaluation, never
used to select or tune role-context samples.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from songviz.ingest import sha256_file

CODE = (
    "experiments/build_role_context_review.py",
    "experiments/templates/role_context_review.html",
    "experiments/check_role_context_review.cjs",
    "songviz/role_context.py",
    "songviz/change_episodes.py",
    "songviz/local_structure.py",
    "songviz/local_structure_variants.py",
    "songviz/local_structure_evaluation.py",
    "experiments/compare_local_structure.py",
    "songviz/ingest.py",
)
FEEDBACK_SHA256 = "f69b14f0343d0ebe4af12d37f66e88c76aa5d8e7673f7529345333589d5c47b6"
ANSWER_IDS = ("drum-entry", "within-passage", "verse-ending", "transition-extent")
ANSWER_VALUES = {"none", "subtle", "local", "broad"}
DESCRIPTORS = (
    "signed_rms_difference", "signed_active_fraction_difference",
    "signed_rms_power_share_difference", "signed_spectral_concentration_difference",
    "signed_adjacent_spectral_change_difference",
)


def record(path: Path) -> dict:
    path = path.resolve()
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def verify(path: Path, digest: str) -> None:
    if not path.is_file() or sha256_file(path) != digest:
        raise ValueError(f"Changed or mismatched input: {path}")


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def embedded_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def _path(row: dict) -> Path:
    path = Path(row["path"])
    return path if path.is_absolute() else ROOT / path


def _verify_manifest(package: Path, manifest: dict, groups: tuple[str, ...]) -> list[Path]:
    consumed = [package / "manifest.json"]
    for group in groups:
        rows = manifest.get(group)
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"Missing {group} fingerprints in {package}/manifest.json")
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("path"), str) or not isinstance(row.get("sha256"), str):
                raise ValueError(f"Invalid {group} fingerprint in {package}/manifest.json")
            path = _path(row).resolve()
            verify(path, row["sha256"])
            consumed.append(path)
    return consumed


def _find_record(records: list[dict], path: Path) -> dict:
    rows = [row for row in records if _path(row).resolve() == path.resolve()]
    if len(rows) != 1:
        raise ValueError(f"Manifest does not uniquely bind {path}")
    return rows[0]


def _read_feedback(feedback_path: Path, listening: Path) -> tuple[dict, dict, list[Path]]:
    if sha256_file(feedback_path) != FEEDBACK_SHA256:
        raise ValueError("Listening feedback SHA-256 does not match the frozen export")
    manifest_path = listening / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("kind") != "songviz-listening-examples":
        raise ValueError("Expected verified guided-listening package")
    consumed = _verify_manifest(listening, manifest, ("input_snapshots", "outputs"))
    review_path = listening / "review.json"
    _find_record(manifest["outputs"], review_path)
    review, feedback = json.loads(review_path.read_text()), json.loads(feedback_path.read_text())
    if feedback.get("kind") != "songviz-listening-examples-feedback" or feedback.get("schema_version") != 1:
        raise ValueError("Feedback schema is not the guided-listening export")
    if feedback.get("review_sha256") != sha256_file(review_path):
        raise ValueError("Feedback is not hash-bound to the listening review")
    for key in ("example_set_id", "source_audio_sha256", "audio_sha256"):
        if feedback.get(key) != review.get(key):
            raise ValueError(f"Feedback {key} does not match listening source metadata")
    answers = feedback.get("answers")
    if not isinstance(answers, list) or len(answers) != 4 or [x.get("example_id") for x in answers if isinstance(x, dict)] != list(ANSWER_IDS):
        raise ValueError("Feedback must contain each of the four guided example IDs exactly once")
    if any(not isinstance(x.get("notes"), str) or x.get("perceived_change") not in ANSWER_VALUES for x in answers):
        raise ValueError("Feedback note text or perceived-change enum is invalid")
    examples = review.get("examples")
    if not isinstance(examples, list) or {x.get("id") for x in examples} != set(ANSWER_IDS):
        raise ValueError("Listening review no longer has the expected four examples")
    return feedback, review, [*consumed, feedback_path.resolve(), review_path.resolve()]


def _detector():
    try:
        return importlib.import_module("songviz.role_context").compute_role_context
    except (ImportError, AttributeError) as exc:
        raise RuntimeError("Role-context builder requires songviz.role_context.compute_role_context") from exc


def _supported(sample: object) -> bool:
    return isinstance(sample, dict) and isinstance(sample.get("anchor_index"), int)


def _curve_by_scale(result: dict) -> dict[int, dict]:
    curves = result.get("curves")
    if not isinstance(curves, list):
        raise ValueError("Role-context detector has no curves")
    mapped = {curve.get("scale_beats"): curve for curve in curves if isinstance(curve, dict)}
    scales = result.get("config", {}).get("scales")
    if not isinstance(scales, list) or set(mapped) != set(scales) or len(mapped) != len(scales):
        raise ValueError("Role-context curves do not match configured scales")
    return {int(scale): mapped[scale] for scale in scales}


def _validate_result(result: dict, times: list[float]) -> None:
    if not isinstance(result, dict) or result.get("schema_version") != 1 or result.get("kind") != "songviz-role-context":
        raise ValueError("Role-context detector returned an incompatible result")
    if result.get("times_s") != times or not isinstance(result.get("stem_names"), list) or not isinstance(result.get("audibility_floors"), dict):
        raise ValueError("Role-context detector metadata is incomplete")
    n = len(times) - 1
    for scale, curve in _curve_by_scale(result).items():
        samples = curve.get("samples")
        if not isinstance(samples, list) or len(samples) != n + 1:
            raise ValueError("Role-context samples must align to every beat boundary")
        for index, sample in enumerate(samples):
            expected = scale <= index <= n - scale
            if expected != _supported(sample):
                raise ValueError("Role-context support/null boundary invariant failed")
            if sample is None:
                continue
            required = ("anchor_index", "anchor_s", "scale_beats", "support_start_s", "support_end_s", "available_at_s", "left_support", "right_support", "stems", "physical_onset_s", "physical_settled_s")
            if not all(key in sample for key in required) or sample["anchor_index"] != index or sample["scale_beats"] != scale:
                raise ValueError("Malformed role-context sample")
            if (sample["anchor_s"] != times[index] or sample["support_start_s"] != times[index - scale]
                    or sample["support_end_s"] != times[index + scale] or sample["available_at_s"] != times[index + scale]
                    or sample["left_support"] != {"start_s": times[index - scale], "end_s": times[index]}
                    or sample["right_support"] != {"start_s": times[index], "end_s": times[index + scale]}):
                raise ValueError("Role-context support does not match the beat grid")
            if sample["physical_onset_s"] is not None or sample["physical_settled_s"] is not None:
                raise ValueError("Role-context must not claim physical timing")
            if set(sample["stems"]) != set(result["stem_names"]):
                raise ValueError("Role-context sample stem coverage differs from metadata")
            for stem in result["stem_names"]:
                item = sample["stems"].get(stem) if isinstance(sample["stems"], dict) else None
                if (not isinstance(item, dict) or not {"left", "right", "changes", "musical_role", "vocal_function", "perceived_importance"} <= set(item)
                        or item["musical_role"] is not None or item["vocal_function"] is not None or item["perceived_importance"] is not None
                        or not isinstance(item["changes"], dict) or set(item["changes"]) != set(DESCRIPTORS)):
                    raise ValueError("Role-context must retain all stems and semantic unknowns")


def _nearest_anchor(times: list[float], midpoint: float) -> int:
    return min(range(len(times)), key=lambda index: (abs(times[index] - midpoint), index))


def _descriptor_summary(samples: list[dict | None], stems: list[str]) -> dict:
    result: dict[str, dict] = {}
    for stem in stems:
        values: dict[str, list[float | None]] = defaultdict(list)
        for sample in samples:
            changes = sample.get("stems", {}).get(stem, {}).get("changes", {}) if _supported(sample) else {}
            for descriptor in DESCRIPTORS:
                value = changes.get(descriptor)
                values[descriptor].append(value if isinstance(value, (int, float)) and not isinstance(value, bool) else None)
        result[stem] = {}
        for descriptor in DESCRIPTORS:
            found = [float(value) for value in values[descriptor] if value is not None]
            result[stem][descriptor] = {"min": min(found) if found else None, "max": max(found) if found else None,
                "positive_count": sum(value > 0 for value in found), "negative_count": sum(value < 0 for value in found),
                "zero_count": sum(value == 0 for value in found), "unknown_count": len(values[descriptor]) - len(found)}
    return result


def _guided_cases(listening_review: dict, feedback: dict, result: dict) -> tuple[list[dict], dict]:
    answers = {item["example_id"]: item for item in feedback["answers"]}
    times, curves, stems = result["times_s"], _curve_by_scale(result), result["stem_names"]
    cases, evaluation = [], {"policy": {"anchor": "nearest beat-grid boundary to focus midpoint; earliest boundary wins ties", "neighbor_offsets_beats": [-2, -1, 0, 1, 2], "quality_score": None}, "cases": []}
    for example in listening_review["examples"]:
        bounds = {"start_s": float(example["start_s"]), "end_s": float(example["end_s"])}
        focus = {"start_s": float(example["focus_start_s"]), "end_s": float(example["focus_end_s"])}
        anchor = _nearest_anchor(times, (focus["start_s"] + focus["end_s"]) / 2)
        answer = answers[example["id"]]
        local_curves, scale_rows = [], []
        for scale, curve in curves.items():
            local = [sample for sample in curve["samples"] if _supported(sample) and bounds["start_s"] <= sample["anchor_s"] <= bounds["end_s"]]
            selected = [curve["samples"][anchor + offset] if 0 <= anchor + offset < len(times) else None for offset in (-2, -1, 0, 1, 2)]
            local_curves.append({"scale_beats": scale, "samples": local})
            scale_rows.append({"scale_beats": scale, "anchor_samples": selected, "descriptor_summary": _descriptor_summary(selected, stems)})
        case = {"id": example["id"], "excerpt_bounds_s": bounds, "focus_bounds_s": focus,
                "perceived_change": answer["perceived_change"], "raw_response_notes": answer["notes"],
                "fixed_anchor_index": anchor, "fixed_anchor_s": times[anchor], "local_curves": local_curves}
        cases.append(case)
        evaluation["cases"].append({"id": example["id"], "focus_bounds_s": focus, "fixed_anchor_index": anchor,
            "fixed_anchor_s": times[anchor], "per_scale": scale_rows,
            "cross_scale_sensitivity": "Descriptive comparison of all configured scales and fixed neighbor anchors; no best scale, selected anchor, or quality score is computed."})
    return cases, evaluation


def _page_data(data: dict) -> dict:
    # review.json intentionally contains only the four excerpts; role-context.json
    # preserves the continuous full-song results.
    return {key: data[key] for key in ("schema_version", "kind", "song_title", "duration_s", "audio_path", "audio_sha256", "method", "config", "stem_names", "audibility_floors", "limitations", "guided_cases", "notes")}


def report_markdown(data: dict) -> str:
    lines = ["# Continuous role-context review", "", data["song_title"], "",
             "All supported beat-grid anchors are computed independently of episode candidates. Acoustic RMS-power shares are not musical leadership; vocal function and perceived importance are unknown.", ""]
    for case in data["evaluation"]["cases"]:
        lines += [f"## {case['id']}", "", f"Fixed anchor: {case['fixed_anchor_s']:.3f}s (index {case['fixed_anchor_index']}); offsets −2, −1, 0, +1, +2 beats at every scale.", "",
                  "| Stem | Scale | Descriptor | Min | Max | + | − | 0 | Unknown |", "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for row in case["per_scale"]:
            for stem, descriptors in row["descriptor_summary"].items():
                for name, value in descriptors.items():
                    fmt = lambda x: "unknown" if x is None else f"{x:.5f}"
                    lines.append(f"| {stem} | {row['scale_beats']} | {name} | {fmt(value['min'])} | {fmt(value['max'])} | {value['positive_count']} | {value['negative_count']} | {value['zero_count']} | {value['unknown_count']} |")
        def counts(stem: str, descriptor: str) -> tuple[int, int, int, int]:
            rows = [row["descriptor_summary"].get(stem, {}).get(descriptor, {}) for row in case["per_scale"]]
            return tuple(sum(int(row.get(key, 0)) for row in rows) for key in ("positive_count", "negative_count", "zero_count", "unknown_count"))
        if case["id"] == "within-passage":
            rms, adjacent = counts("other", "signed_rms_difference"), counts("other", "signed_adjacent_spectral_change_difference")
            lines += [f"Observed fixed-neighbor signs: other-stem RMS is negative {rms[1]} times and adjacent spectral change is negative {adjacent[1]} times across the configured scales (with {rms[3]}/{adjacent[3]} unknown). This is retained beside the user's `none` judgment: stable acoustic direction is not importance.", ""]
        if case["id"] == "verse-ending":
            rms, share = counts("vocals", "signed_rms_difference"), counts("vocals", "signed_rms_power_share_difference")
            concentration, adjacent = counts("vocals", "signed_spectral_concentration_difference"), counts("vocals", "signed_adjacent_spectral_change_difference")
            lines += [f"Observed fixed-neighbor signs: vocal RMS/share are negative {rms[1]}/{share[1]} times; concentration has +/−/0/unknown {concentration[0]}/{concentration[1]}/{concentration[2]}/{concentration[3]} and adjacent change {adjacent[0]}/{adjacent[1]}/{adjacent[2]}/{adjacent[3]}. Mixed spectral signs do not recognize vocal behavior or function.", ""]
        lines.append("")
    lines += ["## Limits", "", "- Values are windowed acoustic descriptors with explicit before/after support, not role, leadership, vocal-function, importance, physical-onset, or physical-settling estimates.", "- Cross-scale sensitivity is descriptive; no best anchor, scale, or quality score is selected. Stable signs cannot substitute for perceived importance, as the fixed `none` case demonstrates."]
    return "\n".join(lines) + "\n"


def build(*, parent: Path, listening: Path, feedback: Path, audio_review: Path, out: Path) -> None:
    parent, listening, feedback, audio_review, out = (path.resolve() for path in (parent, listening, feedback, audio_review, out))
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    if any(source == out or source in out.parents for source in (parent, listening, audio_review)):
        raise ValueError("Output must be separate from input packages")
    from experiments import compare_local_structure as comparison
    _, reference, _, _, _, _, bt, features, energy, duration, parent_inputs = comparison._read_parent(parent, audio_review)
    feedback_data, listening_review, listening_inputs = _read_feedback(feedback, listening)
    source = reference["source"]
    if any(listening_review.get(key) != source.get(key) for key in ("source_audio_sha256", "audio_sha256")) or listening_review.get("duration_s") != duration:
        raise ValueError("Listening review is not bound to the same source audio and duration")
    result = _detector()(features, energy, bt)
    _validate_result(result, bt.tolist())
    cases, evaluation = _guided_cases(listening_review, feedback_data, result)
    relative_audio = quote(Path(os.path.relpath(audio_review / "original.wav", out)).as_posix(), safe="/")
    data = {"schema_version": 1, "kind": "songviz-role-context-review", "song_title": source["song_title"], "duration_s": duration,
            "audio_path": relative_audio, "audio_sha256": source["audio_sha256"], "method": result["method"], "config": result["config"],
            "stem_names": result["stem_names"], "audibility_floors": result["audibility_floors"], "limitations": result["limitations"], "guided_cases": cases, "evaluation": evaluation,
            "notes": ["Human listening data enters only this evaluation and is excluded from compute_role_context.", "Participação RMS² é uma participação acústica relativa, não liderança, função vocal ou importância percebida.", "Função vocal e importância percebida permanecem desconhecidas; não há formulário novo nesta revisão."]}
    code_paths = [ROOT / item for item in CODE]
    if any(not path.is_file() for path in code_paths):
        raise FileNotFoundError("Required role-context source is missing")
    inputs = {path.resolve(): record(path) for path in [*parent_inputs, *listening_inputs, *code_paths]}
    for item in inputs.values(): verify(Path(item["path"]), item["sha256"])
    out.mkdir(parents=True)
    for relative in CODE:
        dest = out / "inputs" / relative; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT / relative, dest)
    for source_path, name in ((parent / "manifest.json", "parent-manifest.json"), (listening / "manifest.json", "listening-manifest.json"), (audio_review / "manifest.json", "audio-manifest.json"), (feedback, "listening-feedback.json")):
        shutil.copy2(source_path, out / "inputs" / name)
    write_json(out / "role-context.json", result)
    write_json(out / "evaluation.json", evaluation)
    page = _page_data(data); write_json(out / "review.json", page)
    (out / "report.md").write_text(report_markdown(data))
    template = (out / "inputs/experiments/templates/role_context_review.html").read_text()
    (out / "index.html").write_text(template.replace("{{REVIEW_JSON}}", embedded_json(page)).replace("{{REVIEW_SHA}}", sha256_file(out / "review.json")))
    for item in inputs.values(): verify(Path(item["path"]), item["sha256"])
    manifest = {"schema_version": 1, "kind": "songviz-role-context-review", "created_utc": datetime.now(timezone.utc).isoformat(),
        "sources": list(inputs.values()), "input_snapshots": [record(p) for p in sorted((out / "inputs").rglob("*")) if p.is_file()],
        "outputs": [record(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name != "manifest.json"],
        "python_version": platform.python_version(), "scope": "Continuous all-stem label-free acoustic role context; four frozen listening cases are descriptive evaluation only.",
        "page_integrity": "index.html is derived from the snapshotted template and review.json SHA-256; manifest does not self-reference."}
    write_json(out / "manifest.json", manifest)
    print(f"Ready: {out}/index.html", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=ROOT / "outputs/reviews/structure-evaluation-03")
    parser.add_argument("--listening", type=Path, default=ROOT / "outputs/reviews/listening-examples-01")
    parser.add_argument("--feedback", type=Path, default=ROOT / "benchmark/feedback/listening-examples-01.json")
    parser.add_argument("--audio-review", type=Path, default=ROOT / "outputs/reviews/structure-review-03")
    parser.add_argument("--out", type=Path, required=True)
    build(**vars(parser.parse_args()))
