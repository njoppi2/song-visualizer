#!/usr/bin/env python3
"""Run the predeclared, one-shot section-method transfer screen.

This runner intentionally calls the existing full-song ``compute_story`` method
without listener references or separated stems.  It writes each result before
joining the fixed reference windows used only for the final screen report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import librosa
import numpy as np

from songviz.story import compute_story


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 22_050
HOP_LENGTH = 512
FRAME_LENGTH = 2_048


@dataclass(frozen=True)
class TransferCase:
    key: str
    source_relative_path: str
    source_sha256: str
    window_start_s: float
    window_end_s: float
    decision_use: str


CASES = (
    TransferCase(
        "agnes",
        "songs/Agnes - MILK.flac",
        "ac23d00df4892e03b8df1696927d1b066bcc147122112b7154239ea3dbb43fbb",
        104.294263,
        106.294263,
        "positive",
    ),
    TransferCase(
        "castlecomer",
        "songs/Castlecomer - Move.flac",
        "0fcc06d48fa5def9c5472ac2a25146d7fbaf885a1c5c3ef4dd3893139f4d6555",
        92.633333333,
        95.633333333,
        "positive",
    ),
    TransferCase(
        "arctic_monkeys",
        "songs/Arctic Monkeys - Do I Wanna Know_.flac",
        "f1133a00b470c7732fea1bf784c1809b438e01070e05456aa416c9913be25d4b",
        140.197052,
        142.197052,
        "event_scope_only",
    ),
    TransferCase(
        "feel_good_inc",
        "songs/Gorillaz - Feel Good Inc (featuring De La Soul).flac",
        "657af9333edafbb6cf8cc573446650773d1169b0117c784106d1895cf802cb44",
        16.0,
        26.0,
        "negative",
    ),
)

# Transcribed from the preregistration's descriptive table.  These are never
# passed to compute_story and are joined only after all story files are saved.
FGI_HUMAN_INTERNAL_BOUNDARIES_S = (
    5.350, 6.157, 30.467, 33.445, 61.368, 64.855, 78.890, 92.816,
    95.398, 124.295, 137.999, 144.178, 158.591, 165.534, 187.441,
    189.926, 203.759, 217.973,
)
DIAGNOSTIC_CUTOFFS_S = (0.5, 1.0, 3.0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def boundaries_in_window(boundaries_s: Iterable[float], start_s: float, end_s: float) -> list[float]:
    """Return final boundaries inside the closed, preregistered time window."""
    if not start_s <= end_s:
        raise ValueError("window start must not exceed window end")
    return [float(value) for value in boundaries_s if start_s <= float(value) <= end_s]


def screen_gate(window_boundaries: Mapping[str, Iterable[float]]) -> dict[str, Any]:
    """Apply the fixed two-positive/one-negative acceptance rule only."""
    positives = {
        key: len(list(window_boundaries[key])) >= 1
        for key in ("agnes", "castlecomer")
    }
    negative_empty = len(list(window_boundaries["feel_good_inc"])) == 0
    return {
        "status": "pass" if all(positives.values()) and negative_empty else "fail",
        "positive_requirements": positives,
        "negative_requirement": {"feel_good_inc_no_boundary": negative_empty},
        "arctic_monkeys_not_scored": True,
    }


def fgi_distance_diagnostics(
    final_boundaries_s: Iterable[float],
    human_boundaries_s: Iterable[float] = FGI_HUMAN_INTERNAL_BOUNDARIES_S,
) -> dict[str, Any]:
    """Describe nearest final-boundary distances without using them for scoring."""
    predicted = [float(value) for value in final_boundaries_s]
    rows: list[dict[str, float | None]] = []
    for human in human_boundaries_s:
        human_value = float(human)
        if predicted:
            nearest = min(predicted, key=lambda value: abs(value - human_value))
            rows.append({
                "human_boundary_s": human_value,
                "nearest_final_boundary_s": nearest,
                "signed_offset_s": nearest - human_value,
                "absolute_distance_s": abs(nearest - human_value),
            })
        else:
            rows.append({
                "human_boundary_s": human_value,
                "nearest_final_boundary_s": None,
                "signed_offset_s": None,
                "absolute_distance_s": None,
            })
    return {
        "kind": "descriptive_in_sample_diagnostic_not_a_scoring_tolerance",
        "cutoffs_s": list(DIAGNOSTIC_CUTOFFS_S),
        "human_boundaries": rows,
        "human_boundaries_with_nearest_final_boundary_within_cutoff": {
            str(cutoff): sum(
                row["absolute_distance_s"] is not None and row["absolute_distance_s"] <= cutoff
                for row in rows
            )
            for cutoff in DIAGNOSTIC_CUTOFFS_S
        },
    }


def final_internal_boundaries(story: Mapping[str, Any]) -> list[float]:
    """Extract exactly the scoreable final starts: ``story['sections'][1:]``."""
    sections = story.get("sections")
    if not isinstance(sections, list) or not sections:
        raise ValueError("compute_story returned no final sections")
    starts: list[float] = []
    previous = -float("inf")
    for section in sections:
        if not isinstance(section, Mapping) or "start_s" not in section:
            raise ValueError("compute_story returned a section without start_s")
        start = float(section["start_s"])
        if not np.isfinite(start) or start < previous:
            raise ValueError("compute_story returned invalid section starts")
        starts.append(start)
        previous = start
    return starts[1:]


def code_provenance() -> dict[str, str]:
    paths = {
        "runner": Path(__file__).resolve(),
        "story_py": ROOT / "songviz/story.py",
        "structure_grid_py": ROOT / "songviz/structure_grid.py",
    }
    return {key: sha256_file(path) for key, path in paths.items()}


def versions() -> dict[str, str]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "librosa": librosa.__version__,
        "numpy": np.__version__,
    }


def _story_summary(case: TransferCase, story: Mapping[str, Any], elapsed_s: float) -> dict[str, Any]:
    final_boundaries = final_internal_boundaries(story)
    duration = float(story["meta"]["duration_s"])
    if not duration > 0:
        raise ValueError("compute_story returned a nonpositive duration")
    meta = story.get("meta", {})
    events = story.get("events", {})
    return {
        "source_relative_path": case.source_relative_path,
        "source_sha256": case.source_sha256,
        "window": {
            "start_s": case.window_start_s,
            "end_s": case.window_end_s,
            "decision_use": case.decision_use,
        },
        "duration_s": duration,
        "runtime_s": elapsed_s,
        "section_method": meta.get("section_method"),
        "section_error": meta.get("section_error"),
        "beat_grid": meta.get("beat_grid"),
        "final_internal_boundaries_s": final_boundaries,
        "final_internal_boundary_count": len(final_boundaries),
        "final_internal_boundary_count_per_minute": len(final_boundaries) / (duration / 60.0),
        "final_boundaries_in_fixed_window_s": boundaries_in_window(
            final_boundaries, case.window_start_s, case.window_end_s
        ),
        "final_sections_descriptive": story["sections"],
        "discarded_boundary_candidates_descriptive": events.get("boundary_candidates", []),
    }


def _report_text(result: Mapping[str, Any]) -> str:
    lines = [
        "# Predeclared current-section-method transfer screen",
        "",
        "This is a one-shot transfer screen using the existing `compute_story` section output.",
        "Only final internal starts from `story['sections'][1:]` are scored.",
        "The Arctic window is reported but excluded from the gate. The Feel Good Inc human-boundary",
        "distances are descriptive in-sample diagnostics, not scoring tolerances or standard metrics.",
        "",
        f"## Gate: {result['gate']['status']}",
        "",
    ]
    for case in CASES:
        item = result["songs"][case.key]
        lines.extend([
            f"## {case.key}",
            "",
            f"- Fixed window: {case.window_start_s:.9f}–{case.window_end_s:.9f}s ({case.decision_use})",
            f"- Final internal boundaries: {item['final_internal_boundaries_s']}",
            f"- Boundaries in fixed window: {item['final_boundaries_in_fixed_window_s']}",
            f"- Count/minute: {item['final_internal_boundary_count_per_minute']:.6f}",
            f"- Section method/error: {item['section_method']} / {item['section_error']}",
            "",
        ])
    diagnostic = result["feel_good_inc_human_boundary_diagnostics"]
    lines.extend([
        "## Feel Good Inc descriptive human-boundary distances",
        "",
        f"- Counts within descriptive cutoffs: {diagnostic['human_boundaries_with_nearest_final_boundary_within_cutoff']}",
        "- These values do not affect the gate.",
        "",
    ])
    return "\n".join(lines)


def run(output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing destination: {output_dir}")
    output_dir.mkdir(parents=True)
    stories_dir = output_dir / "stories"
    stories_dir.mkdir()
    log_path = output_dir / "run.log"

    def log(message: str) -> None:
        line = f"{datetime.now(timezone.utc).isoformat()} {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    started = time.perf_counter()
    try:
        log("starting fixed four-source run")
        songs: dict[str, Any] = {}
        for case in CASES:
            source = ROOT / case.source_relative_path
            actual_hash = sha256_file(source)
            if actual_hash != case.source_sha256:
                raise ValueError(f"source SHA-256 mismatch for {case.key}: {actual_hash}")
            log(f"loading {case.key} source_sha256={actual_hash}")
            y, sr = librosa.load(source, sr=SAMPLE_RATE, mono=True)
            if sr != SAMPLE_RATE:
                raise ValueError(f"librosa did not produce {SAMPLE_RATE} Hz for {case.key}: {sr}")
            case_started = time.perf_counter()
            # Deliberately no reference data, external beat grid, stems, or other stem input.
            story = compute_story(
                y, SAMPLE_RATE, hop_length=HOP_LENGTH, frame_length=FRAME_LENGTH,
                stems=None, other_y=None, beat_times_s=None,
            )
            elapsed = time.perf_counter() - case_started
            story_path = stories_dir / f"{case.key}.story.json"
            write_json(story_path, story)
            songs[case.key] = _story_summary(case, story, elapsed)
            log(f"saved {story_path.relative_to(output_dir)} runtime_s={elapsed:.6f}")

        # All predictions now exist on disk.  Join fixed references only here.
        windows = {key: item["final_boundaries_in_fixed_window_s"] for key, item in songs.items()}
        result = {
            "schema_version": 1,
            "run_kind": "predeclared_current_section_method_transfer_screen",
            "fixed_call": {
                "loader": "librosa.load(source, sr=22050, mono=True)",
                "compute_story": {
                    "sample_rate": SAMPLE_RATE,
                    "hop_length": HOP_LENGTH,
                    "frame_length": FRAME_LENGTH,
                    "stems": None,
                    "other_y": None,
                    "beat_times_s": None,
                },
            },
            "songs": songs,
            "gate": screen_gate(windows),
            "feel_good_inc_human_boundary_diagnostics": fgi_distance_diagnostics(
                songs["feel_good_inc"]["final_internal_boundaries_s"]
            ),
            "runtime_s": time.perf_counter() - started,
        }
        write_json(output_dir / "result.json", result)
        (output_dir / "report.md").write_text(_report_text(result), encoding="utf-8")
        manifest = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_hashes": {case.key: case.source_sha256 for case in CASES},
            "code_sha256": code_provenance(),
            "versions": versions(),
            "fixed_windows": {
                case.key: {"start_s": case.window_start_s, "end_s": case.window_end_s,
                           "decision_use": case.decision_use}
                for case in CASES
            },
            "reference_join_rule": "saved all story outputs before joining any fixed windows or human boundaries",
            "artifact_sha256": {
                path.relative_to(output_dir).as_posix(): sha256_file(path)
                for path in sorted(stories_dir.glob("*.json")) + [output_dir / "result.json", output_dir / "report.md"]
            },
        }
        write_json(output_dir / "manifest.json", manifest)
        log(f"finished status={result['gate']['status']} runtime_s={result['runtime_s']:.6f}")
        return result
    except Exception as exc:
        error = {
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(output_dir / "error.json", error)
        log(f"failed error_type={type(exc).__name__} message={exc}")
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/reviews/ssm-section-transfer-01",
        help="new destination; existing directories are refused",
    )
    args = parser.parse_args()
    run(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
