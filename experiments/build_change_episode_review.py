"""Build an immutable, label-free change-episode review package.

The detector receives no human material: only frozen numeric features, energy
and beat times. Its intervals are windowed contrast responses, not claimed physical transition
durations or a replacement for the five interpreted transition spans.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
from urllib.parse import quote, unquote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from songviz.ingest import sha256_file

CODE = (
    "experiments/build_change_episode_review.py",
    "experiments/templates/change_episode_review.html",
    "experiments/check_change_episode_review.cjs",
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
    consumed: list[Path] = [package / "manifest.json"]
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
    matches = [row for row in records if _path(row).resolve() == path.resolve()]
    if len(matches) != 1:
        raise ValueError(f"Manifest does not uniquely bind {path}")
    return matches[0]


def _read_comparison(comparison_dir: Path) -> tuple[dict, dict, list[Path]]:
    manifest_path = comparison_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("kind") != "songviz-local-structure-comparison":
        raise ValueError("Expected verified local-structure comparison")
    consumed = _verify_manifest(comparison_dir, manifest, ("input_snapshots", "outputs"))
    review_path = comparison_dir / "review.json"
    _find_record(manifest["outputs"], review_path)
    review = json.loads(review_path.read_text())
    control_path = comparison_dir / "variants/control/predictions.json"
    _find_record(manifest["outputs"], control_path)
    control = json.loads(control_path.read_text())
    frozen = control.get("transitions")
    page_control = review.get("variants", {}).get("control", {}).get("predictions", {}).get("transitions")
    if not isinstance(frozen, list) or frozen != page_control:
        raise ValueError("Frozen control dip snapshots differ from comparison review")
    counts = review.get("summary", {}).get("variants", {}).get("control", {}).get("candidate_counts")
    if not isinstance(counts, dict) or not isinstance(counts.get("changes", {}).get("count"), int) or not isinstance(counts.get("transitions", {}).get("count"), int):
        raise ValueError("Comparison baseline counts are missing")
    if counts["transitions"]["count"] != len(frozen):
        raise ValueError("Comparison baseline dip count differs from frozen control")
    return review, {"dips": frozen, "counts": counts}, consumed


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
    review = json.loads(review_path.read_text())
    feedback = json.loads(feedback_path.read_text())
    if feedback.get("kind") != "songviz-listening-examples-feedback" or feedback.get("schema_version") != 1:
        raise ValueError("Feedback schema is not the guided-listening export")
    if feedback.get("review_sha256") != sha256_file(review_path):
        raise ValueError("Feedback is not hash-bound to the listening review")
    for key in ("example_set_id", "source_audio_sha256", "audio_sha256"):
        if feedback.get(key) != review.get(key):
            raise ValueError(f"Feedback {key} does not match listening source metadata")
    answers = feedback.get("answers")
    if not isinstance(answers, list) or [item.get("example_id") for item in answers if isinstance(item, dict)] != list(ANSWER_IDS) or len(answers) != 4:
        raise ValueError("Feedback must contain each of the four guided example IDs exactly once")
    for item in answers:
        if not isinstance(item.get("notes"), str) or item.get("perceived_change") not in ANSWER_VALUES:
            raise ValueError("Feedback note text or perceived-change enum is invalid")
    examples = review.get("examples")
    if not isinstance(examples, list) or {item.get("id") for item in examples} != set(ANSWER_IDS):
        raise ValueError("Listening review no longer has the expected four examples")
    consumed.extend([feedback_path.resolve(), review_path.resolve()])
    return feedback, review, consumed


def _detector():
    try:
        module = importlib.import_module("songviz.change_episodes")
        return module.detect_change_episodes
    except (ImportError, AttributeError) as exc:
        raise RuntimeError("Change-episode builder requires songviz.change_episodes.detect_change_episodes") from exc


def _interval_overlap(a: dict, b: dict) -> float:
    return max(0.0, min(float(a["end_s"]), float(b["end_s"])) - max(float(a["start_s"]), float(b["start_s"])))


def _iou(a: dict, b: dict) -> float:
    common = _interval_overlap(a, b)
    return common / (float(a["end_s"]) - float(a["start_s"]) + float(b["end_s"]) - float(b["start_s"]) - common) if common else 0.0


def _transition_spans(reference: dict) -> list[dict]:
    spans = [{**span, "layer_id": layer["id"], "layer_name": layer["name"]}
             for layer in reference.get("layers", []) for span in layer.get("spans", []) if span.get("transition") is True]
    if len(spans) != 5:
        raise ValueError("Expected all five interpreted human transition spans")
    return spans


def evaluate(episodes: list[dict], reference: dict, dips: list[dict]) -> dict:
    spans = _transition_spans(reference)
    rows = []
    for span in spans:
        overlapping = [episode for episode in episodes if _interval_overlap(span, episode) > 0]
        best = max(overlapping, key=lambda episode: (_iou(span, episode), episode["id"]), default=None)
        rows.append({"reference": span, "overlapping_episode_ids": [item["id"] for item in overlapping],
                     "best_iou": None if best is None else {"episode_id": best["id"], "iou": _iou(span, best),
                         "start_error_s": best["start_s"] - span["start_s"], "end_error_s": best["end_s"] - span["end_s"]}})
    dip_rows = []
    for dip in dips:
        comparisons = []
        for span in spans:
            overlap = _interval_overlap(dip, span)
            comparisons.append({"transition_id": span["id"], "overlap_s": overlap, "iou": _iou(dip, span),
                                "start_error_s": dip["start_s"] - span["start_s"], "end_error_s": dip["end_s"] - span["end_s"]})
        best = max(comparisons, key=lambda item: (item["iou"], item["transition_id"]))
        dip_rows.append({"dip": dip, "against_all_five_transition_spans": comparisons, "best_iou": best})
    return {"human_transition_response_interval_overlap": rows,
            "frozen_two_dip_overlap": dip_rows,
            "limitations": ["These are windowed contrast response intervals, not known physical transition durations.",
                            "A response interval can overstate the width of a persistent step.",
                            "IoU and signed endpoint errors are descriptive and many-to-one, not accuracy or acceptance scores."]}


def _summaries(episodes: list[dict]) -> dict:
    groups: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    by_stem: dict[str, list[float]] = defaultdict(list)
    by_scale: dict[int, list[float]] = defaultdict(list)
    by_channel: dict[str, list[float]] = defaultdict(list)
    for episode in episodes:
        duration = float(episode["end_s"]) - float(episode["start_s"])
        stem, scale, channel = str(episode["stem"]), int(episode["scale_beats"]), str(episode["channel"])
        groups[(stem, scale, channel)].append(duration); by_stem[stem].append(duration); by_scale[scale].append(duration); by_channel[channel].append(duration)
    def rows_for(items, names):
        return [{**dict(zip(names, key if isinstance(key, tuple) else (key,))), "count": len(durations),
                 "duration_s": {"min": min(durations), "median": float(__import__("numpy").median(durations)), "max": max(durations)}}
                for key, durations in sorted(items.items())]
    return {"total_episode_count": len(episodes), "by_stem": rows_for(by_stem, ("stem",)),
            "by_scale_beats": rows_for(by_scale, ("scale_beats",)), "by_channel": rows_for(by_channel, ("channel",)),
            "by_stem_scale_channel": rows_for(groups, ("stem", "scale_beats", "channel")),
            "note": "Counts and duration distributions are separated by stem, scale and channel; no single quality score is claimed."}


def _guided_cases(listening_review: dict, feedback: dict, episodes: list[dict]) -> list[dict]:
    answers = {item["example_id"]: item for item in feedback["answers"]}
    cases = []
    for example in listening_review["examples"]:
        start, end = float(example["start_s"]), float(example["end_s"])
        focus_a, focus_b = float(example["focus_start_s"]), float(example["focus_end_s"])
        mid = (focus_a + focus_b) / 2
        overlaps = [episode for episode in episodes if _interval_overlap({"start_s": start, "end_s": end}, episode) > 0]
        closest = {}
        for stem in sorted({str(item["stem"]) for item in episodes}):
            choices = [item for item in episodes if item["stem"] == stem]
            if choices:
                item = min(choices, key=lambda item: (abs(float(item["peak_s"]) - mid), item["id"]))
                closest[stem] = {**item, "within_excerpt": start <= item["peak_s"] <= end,
                                 "within_focus": focus_a <= item["peak_s"] <= focus_b}
        answer = answers[example["id"]]
        cases.append({"id": example["id"], "excerpt_bounds_s": {"start_s": start, "end_s": end},
                      "focus_bounds_s": {"start_s": focus_a, "end_s": focus_b},
                      "perceived_change": answer["perceived_change"], "raw_response_notes": answer["notes"],
                      "overlapping_episodes": overlaps, "closest_peak_to_focus_mid_by_stem": closest})
    return cases


def _page_data(data: dict) -> dict:
    cases = []
    times = data["detector_metadata"].get("times_s", [])
    for case in data["guided_cases"]:
        # Full details remain in JSON; HTML only needs this excerpt's bands and a compact summary.
        compact = [{key: episode[key] for key in ("id", "stem", "scale_beats", "channel", "start_s", "end_s", "peak_s")}
                   for episode in case["overlapping_episodes"]]
        chosen = min(case["overlapping_episodes"], key=lambda item: (abs(float(item["peak_s"]) - (case["focus_bounds_s"]["start_s"] + case["focus_bounds_s"]["end_s"]) / 2), item["id"]), default=None)
        lo = next((index for index, value in enumerate(times) if value >= case["excerpt_bounds_s"]["start_s"]), 0)
        hi = next((index for index, value in enumerate(times) if value > case["excerpt_bounds_s"]["end_s"]), len(times))
        excerpt_curves = [{**{key: curve[key] for key in ("stem", "scale_beats", "channel", "high_threshold", "low_threshold")},
                           "times_s": times[lo:hi], "values": curve.get("values", [])[lo:hi]}
                          for curve in data["curves"]]
        cases.append({**{key: case[key] for key in ("id", "excerpt_bounds_s", "focus_bounds_s", "perceived_change", "raw_response_notes")}, "episodes": compact,
                      "closest": {stem: {key: item[key] for key in ("id", "peak_s", "start_s", "end_s", "scale_beats", "channel", "within_excerpt", "within_focus")}
                                  for stem, item in case["closest_peak_to_focus_mid_by_stem"].items()}})
        cases[-1]["excerpt_curves"] = excerpt_curves
        peak_index, scale = (None, None) if chosen is None else (chosen.get("peak_index"), chosen["scale_beats"])
        exact_windows = None
        if isinstance(peak_index, int) and not isinstance(peak_index, bool) and isinstance(scale, int) and peak_index - scale >= 0 and peak_index + scale < len(times):
            exact_windows = {"left": {"start_s": times[peak_index - scale], "end_s": times[peak_index]},
                             "right": {"start_s": times[peak_index], "end_s": times[peak_index + scale]}}
        cases[-1]["chosen_episode_context"] = None if chosen is None else {"episode_id": chosen["id"], "stem": chosen["stem"], "peak_s": chosen["peak_s"], "scale_beats": chosen["scale_beats"], "channel": chosen["channel"], "start_s": chosen["start_s"], "end_s": chosen["end_s"], "support_start_s": chosen["support_start_s"], "support_end_s": chosen["support_end_s"], "peak_windows_s": exact_windows, "peak_context": chosen.get("peak_context")}
        cases[-1]["frozen_dips"] = [dip for dip in data["baseline_control"]["dips"] if _interval_overlap({"start_s": case["excerpt_bounds_s"]["start_s"], "end_s": case["excerpt_bounds_s"]["end_s"]}, dip) > 0]
    return {"song_title": data["song_title"], "audio_path": data["audio_path"], "duration_s": data["duration_s"], "cases": cases,
            "message": "As faixas mostram regiões de mudança nos sinais; a duração musical ainda é incerta."}


def report_markdown(data: dict) -> str:
    lines = ["# Change-episode response-interval review", "", data["song_title"], "",
             "Episode intervals are windowed contrast responses, not physical transition durations. A persistent level step can make the response interval look wider than the musical change.", "",
             "## Counts and response-interval durations", "", "| Stem | Scale (beats) | Channel | Episodes | Min s | Median s | Max s |", "| --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    for row in data["summary"]["by_stem_scale_channel"]:
        d = row["duration_s"]
        lines.append(f"| {row['stem']} | {row['scale_beats']} | {row['channel']} | {row['count']} | {d['min']:.3f} | {d['median']:.3f} | {d['max']:.3f} |")
    lines += ["", "No single quality score is computed.", "", "## Five interpreted human transition spans", ""]
    for row in data["evaluation"]["human_transition_response_interval_overlap"]:
        best = row["best_iou"]
        detail = "none" if best is None else f"{best['iou']:.3f}, endpoint errors {best['start_error_s']:.3f}/{best['end_error_s']:.3f}s"
        lines.append(f"- {row['reference']['label']} ({row['reference']['start_s']:.3f}–{row['reference']['end_s']:.3f}s): {len(row['overlapping_episode_ids'])} overlapping response intervals; best IoU {detail}.")
    lines += ["", "## Frozen two-dip control against all five spans", "", "| Dip | Human span | IoU | Start error s | End error s |", "| --- | --- | ---: | ---: | ---: |"]
    for dip in data["evaluation"]["frozen_two_dip_overlap"]:
        for row in dip["against_all_five_transition_spans"]:
            lines.append(f"| {dip['dip']['id']} | {row['transition_id']} | {row['iou']:.3f} | {row['start_error_s']:.3f} | {row['end_error_s']:.3f} |")
    lines += ["", "## Limits", ""] + ["- " + item for item in data["evaluation"]["limitations"]]
    return "\n".join(lines) + "\n"


def build(*, parent: Path, comparison: Path, listening: Path, feedback: Path, audio_review: Path, out: Path) -> None:
    parent, comparison, listening, feedback, audio_review, out = (path.resolve() for path in (parent, comparison, listening, feedback, audio_review, out))
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    if any(source == out or source in out.parents for source in (parent, comparison, listening, audio_review)):
        raise ValueError("Output must be separate from input packages")
    from experiments import compare_local_structure as comparison_module
    _, reference, _, _, _, audio_review_json, bt, features, energy, duration, parent_inputs = comparison_module._read_parent(parent, audio_review)
    comparison_review, baseline, comparison_inputs = _read_comparison(comparison)
    source = reference["source"]
    if comparison_review.get("audio_sha256") != source.get("audio_sha256") or comparison_review.get("duration_s") != duration:
        raise ValueError("Comparison is not bound to the parent's review audio and duration")
    control_grid = json.loads((comparison / "variants/control/predictions.json").read_text()).get("times_s")
    if control_grid != bt.tolist():
        raise ValueError("Comparison control times_s differs from the parent's numeric beat grid")
    detector = _detector()
    # Human material is excluded from detector inputs; evaluation follows below.
    result = detector(features, energy, bt)
    if not isinstance(result, dict) or not isinstance(result.get("episodes"), list) or not isinstance(result.get("curves"), list):
        raise ValueError("Change-episode detector returned an incomplete result")
    episodes = result["episodes"]
    for item in episodes:
        required = ("id", "stem", "scale_beats", "channel", "start_s", "end_s", "peak_s", "support_start_s", "support_end_s", "available_at_s")
        if not all(key in item for key in required) or not item["start_s"] < item["end_s"] or not item["start_s"] <= item["peak_s"] <= item["end_s"]:
            raise ValueError("Invalid change-episode response interval")
    feedback_data, listening_review, listening_inputs = _read_feedback(feedback, listening)
    if (listening_review.get("source_audio_sha256") != source.get("source_audio_sha256")
            or listening_review.get("audio_sha256") != source.get("audio_sha256")
            or listening_review.get("duration_s") != duration):
        raise ValueError("Listening review is not bound to the same source audio and duration")
    evaluation = evaluate(episodes, reference, baseline["dips"])
    relative_audio = quote(Path(os.path.relpath(audio_review / "original.wav", out)).as_posix(), safe="/")
    data = {"schema_version": 1, "kind": "songviz-change-episode-review", "song_title": reference["source"]["song_title"],
            "duration_s": duration, "audio_path": relative_audio, "audio_sha256": reference["source"]["audio_sha256"],
            "method": result.get("method"), "config": result.get("config"), "episodes": episodes, "curves": result["curves"],
            "detector_metadata": {key: value for key, value in result.items() if key not in {"episodes", "curves"}},
            "summary": _summaries(episodes), "evaluation": evaluation, "guided_cases": _guided_cases(listening_review, feedback_data, episodes),
            "baseline_control": baseline, "notes": ["Human material is excluded from detector inputs; the detector receives only numeric features, energy and beat times.", "The vocal response is not a claim that laughter was detected.", "The within-passage response remains 'none' even if a signal score is high."]}
    inputs = {path.resolve(): record(path) for path in [*parent_inputs, *comparison_inputs, *listening_inputs, *[ROOT / item for item in CODE]]}
    for item in inputs.values(): verify(Path(item["path"]), item["sha256"])
    out.mkdir(parents=True)
    for relative in CODE:
        destination = out / "inputs" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    for source, name in ((parent / "manifest.json", "parent-manifest.json"), (comparison / "manifest.json", "comparison-manifest.json"),
                         (listening / "manifest.json", "listening-manifest.json"), (feedback, "listening-feedback.json")):
        shutil.copy2(source, out / "inputs" / name)
    write_json(out / "episodes.json", result)
    write_json(out / "evaluation.json", {"summary": data["summary"], "evaluation": evaluation, "guided_cases": data["guided_cases"], "baseline_control": baseline})
    write_json(out / "review.json", _page_data(data))
    (out / "report.md").write_text(report_markdown(data))
    template = (out / "inputs/experiments/templates/change_episode_review.html").read_text()
    (out / "index.html").write_text(template.replace("{{REVIEW_JSON}}", embedded_json(_page_data(data))).replace("{{REVIEW_SHA}}", sha256_file(out / "review.json")))
    for item in inputs.values(): verify(Path(item["path"]), item["sha256"])
    manifest = {"schema_version": 1, "kind": "songviz-change-episode-review", "created_utc": datetime.now(timezone.utc).isoformat(),
                "sources": list(inputs.values()), "input_snapshots": [record(path) for path in sorted((out / "inputs").rglob("*")) if path.is_file()],
                "outputs": [record(path) for path in sorted(out.rglob("*")) if path.is_file() and path.name != "manifest.json"],
                "python_version": platform.python_version(), "scope": "Label-free multiscale response-interval experiment; descriptive evaluation only.",
                "page_integrity": "index.html is derived from the snapshotted template and bounded review.json; manifest does not self-reference."}
    write_json(out / "manifest.json", manifest)
    print(f"Ready: {out}/index.html", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=ROOT / "outputs/reviews/structure-evaluation-03")
    parser.add_argument("--comparison", type=Path, default=ROOT / "outputs/reviews/local-structure-comparison-02")
    parser.add_argument("--listening", type=Path, default=ROOT / "outputs/reviews/listening-examples-01")
    parser.add_argument("--feedback", type=Path, default=ROOT / "benchmark/feedback/listening-examples-01.json")
    parser.add_argument("--audio-review", type=Path, default=ROOT / "outputs/reviews/structure-review-03")
    parser.add_argument("--out", type=Path, required=True)
    build(**vars(parser.parse_args()))
