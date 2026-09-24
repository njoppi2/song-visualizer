#!/usr/bin/env python3
"""Run a fixed, source-bound Arctic arrangement diagnostic.

This is deliberately a diagnostic runner, not a new detector.  It extracts the
same full-track CQT/RMS representation as the existing reserve extractor, then
applies the unchanged control, separate-channel, and sustained-activity
policies.  The saved prediction view is restricted to anchors with common
plus/minus-eight-beat support inside the registered listening bounds.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import sys
import time
from typing import Any

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.extract_arrangement_reserve import STEMS, extract
from songviz.local_structure import LocalStructureConfig, detect_local_structure
from songviz.local_structure_variants import LocalStructureVariantConfig, detect_local_structure_variant


SOURCE = "songs/Arctic Monkeys - Do I Wanna Know_.flac"
ANALYSIS = "outputs/Arctic Monkeys - Do I Wanna Know_/analysis/analysis.json"
STEMS_META = "outputs/Arctic Monkeys - Do I Wanna Know_/stems/stems.json"
STEM_PATHS = {name: f"outputs/Arctic Monkeys - Do I Wanna Know_/stems/{name}.wav" for name in STEMS}
REQUIRED_INPUTS = {SOURCE, ANALYSIS, STEMS_META, *STEM_PATHS.values()}
COMMON_SUPPORT_BEATS = 8


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def record(repo: Path, path: Path) -> dict[str, Any]:
    return {"path": str(path.relative_to(repo)), "bytes": path.stat().st_size, "sha256": digest(path)}


def resolve_repo_path(repo: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Registration input paths must be nonempty repository-relative strings")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Registration input path must be repository relative")
    resolved = (repo / path).resolve()
    if resolved != repo and repo not in resolved.parents:
        raise ValueError("Registration input path escapes repository")
    return resolved


def load_registration(repo: Path, registration: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load and hash-verify the exact registration before any extraction."""
    try:
        value = json.loads(registration.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid registration: {registration}") from exc
    if not isinstance(value, dict):
        raise ValueError("Registration must be a JSON object")
    inputs = value.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("Registration inputs must be a nonempty list")
    bounds = value.get("bounds_s")
    if (not isinstance(bounds, list) or len(bounds) != 2
            or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in bounds)):
        raise ValueError("Registration bounds_s must be [start, end]")
    start, end = (float(bounds[0]), float(bounds[1]))
    if not np.isfinite([start, end]).all() or not start < end:
        raise ValueError("Registration bounds_s must be finite and increasing")
    human = value.get("human_song_time_s")
    if isinstance(human, bool) or not isinstance(human, (int, float)) or not np.isfinite(float(human)):
        raise ValueError("Registration human_song_time_s must be finite")
    if not start <= float(human) <= end:
        raise ValueError("Registration human_song_time_s must be inside bounds_s")
    seen: set[str] = set()
    verified: list[dict[str, Any]] = []
    for item in inputs:
        if (not isinstance(item, dict) or not {"path", "sha256"} <= set(item)
                or set(item) - {"path", "sha256", "bytes"}):
            raise ValueError("Each registration input requires path and sha256, with optional bytes")
        path_value, expected = item["path"], item["sha256"]
        if not isinstance(expected, str) or len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
            raise ValueError("Registration input sha256 must be lowercase hexadecimal")
        if path_value in seen:
            raise ValueError("Registration inputs must not repeat a path")
        seen.add(path_value)
        path = resolve_repo_path(repo, path_value)
        if not path.is_file():
            raise ValueError(f"Missing registered input: {path_value}")
        if "bytes" in item and path.stat().st_size != item["bytes"]:
            raise ValueError(f"Byte count mismatch: {path_value}")
        actual = digest(path)
        if actual != expected:
            raise ValueError(f"Fingerprint mismatch: {path_value} ({actual}, expected {expected})")
        verified.append(record(repo, path))
    missing = REQUIRED_INPUTS - seen
    if missing:
        raise ValueError(f"Registration inputs must bind the source/cache/stems (missing={sorted(missing)})")
    return value, verified


def validate_cache(repo: Path, bounds: tuple[float, float], source_sha256: str) -> tuple[dict[str, Path], np.ndarray, float]:
    """Validate source association, duration, and grid before extracting features."""
    metadata = json.loads((repo / STEMS_META).read_text())
    if metadata.get("input", {}).get("path") != SOURCE or metadata.get("input", {}).get("sha256") != source_sha256:
        raise ValueError("Stem metadata does not bind the registered source")
    analysis = json.loads((repo / ANALYSIS).read_text())
    beats = np.asarray(analysis.get("beats", {}).get("beat_times_s"), dtype=np.float64)
    duration = float(analysis.get("meta", {}).get("duration_s"))
    if not np.isfinite(duration) or abs(sf.info(repo / SOURCE).duration - duration) > .05:
        raise ValueError("Source duration differs from cached analysis")
    if (beats.ndim != 1 or beats.size < 2 or not np.isfinite(beats).all() or np.any(np.diff(beats) <= 0)
            or beats[0] < 0 or beats[-1] > duration):
        raise ValueError("Cached analysis has an invalid beat grid")
    if bounds[0] < 0 or bounds[1] > duration:
        raise ValueError("Registered bounds are outside the cached source duration")
    stem_paths = {name: repo / value for name, value in STEM_PATHS.items()}
    for name, path in stem_paths.items():
        if abs(sf.info(path).duration - duration) > .05:
            raise ValueError(f"Stem duration differs from cached analysis: {name}")
    return stem_paths, beats, duration


def eligible_mask(times_s: list[float], bounds: tuple[float, float]) -> list[bool]:
    """Use the common largest (eight-beat) support for every reported curve."""
    return [index >= COMMON_SUPPORT_BEATS and index + COMMON_SUPPORT_BEATS < len(times_s)
            and times_s[index - COMMON_SUPPORT_BEATS] >= bounds[0]
            and times_s[index + COMMON_SUPPORT_BEATS] <= bounds[1]
            for index in range(len(times_s))]


def support_inside(candidate: dict[str, Any], bounds: tuple[float, float]) -> bool:
    return (candidate.get("support_start_s") is not None and candidate.get("support_end_s") is not None
            and candidate["support_start_s"] >= bounds[0] and candidate["support_end_s"] <= bounds[1])


def visible_result(result: dict[str, Any], bounds: tuple[float, float]) -> dict[str, Any]:
    """Mask curve values and retain candidates only when their complete support fits."""
    visible = {key: value for key, value in result.items()
               if key not in {"curves", "channel_curves", "activity_curves", "changes", "transitions"}}
    mask = eligible_mask(result["times_s"], bounds)
    visible["eligible_anchor_indices"] = [index for index, allowed in enumerate(mask) if allowed]
    visible["reported_support_rule"] = "common +/-8 beat support inside registered bounds"
    for source_key, curve_keys in (("curves", ("pattern_change", "arrangement_change", "combined_change")),
                                   ("channel_curves", ("pattern_change", "arrangement_change")),
                                   ("activity_curves", ("activity_change",))):
        curves = []
        for curve in result.get(source_key, []):
            copied = dict(curve)
            for key in curve_keys:
                copied[key] = [value if mask[index] else None for index, value in enumerate(curve[key])]
            # Evidence is also an anchor series and must not reveal out-of-bounds anchors.
            for key in ("stem_evidence", "evidence"):
                if key in copied:
                    if isinstance(copied[key], dict):
                        copied[key] = {stem: [value if mask[index] else None
                                              for index, value in enumerate(series)]
                                       for stem, series in copied[key].items()}
                    else:
                        copied[key] = [value if mask[index] else None for index, value in enumerate(copied[key])]
            curves.append(copied)
        if curves:
            visible[source_key] = curves
    visible["changes"] = [candidate for candidate in result["changes"]
                          if support_inside(candidate, bounds) and mask[candidate["beat_index"]]]
    visible["transitions"] = [candidate for candidate in result["transitions"]
                              if support_inside(candidate, bounds) and mask[candidate["start_beat"]]]
    return visible


def nearest_human_evidence(control: dict[str, Any], separate: dict[str, Any], sustained: dict[str, Any], human_s: float) -> dict[str, Any]:
    times = control["times_s"]
    index = min(range(len(times)), key=lambda item: (abs(times[item] - human_s), item))
    contrasts = []
    for position, curve in enumerate(control["curves"]):
        scale = curve["scale_beats"]
        contrasts.append({"scale_beats": scale, "anchor_index": index, "anchor_s": times[index],
                          "support_start_s": times[index - scale] if index >= scale else None,
                          "support_end_s": times[index + scale] if index + scale < len(times) else None,
                          "pattern_change": curve["pattern_change"][index],
                          "arrangement_change": curve["arrangement_change"][index],
                          "combined_change": curve["combined_change"][index], "threshold": curve["threshold"],
                          "per_stem": separate["channel_curves"][position]["stem_evidence"][index]})
    activity = [{"scale_beats": curve["scale_beats"], "anchor_index": index, "anchor_s": times[index],
                 "activity_change": curve["activity_change"][index], "threshold": curve["threshold"],
                 "support_fraction_required": curve["support_fraction_required"],
                 "per_stem": curve["evidence"][index]}
                for curve in sustained["activity_curves"]]
    return {"requested_song_time_s": human_s, "nearest_anchor_index": index, "nearest_anchor_s": times[index],
            "control_and_separate_channel_per_stem_contrasts": contrasts,
            "sustained_activity_per_stem_evidence": activity,
            "meaning": "descriptive guide comparison only; support is feature input context, not event extent"}


def _candidate_points(result: dict[str, Any]) -> list[float]:
    return [candidate["time_s"] for candidate in result["changes"] + result["transitions"]]


def plot(output: Path, policies: list[tuple[str, dict[str, Any]]], bounds: tuple[float, float], guide: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(6, 1, figsize=(13, 14), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1, 3, 1, 3, 1]})
    colors = {2: "#0b7285", 4: "#1971c2", 8: "#6741d9"}
    for row, (name, result) in enumerate(policies):
        curve_ax, candidate_ax = axes[row * 2], axes[row * 2 + 1]
        for curve in result.get("curves", []):
            scale = curve["scale_beats"]
            curve_ax.plot(result["times_s"], curve["pattern_change"], color=colors.get(scale, "#343a40"),
                          label=f"pattern {scale} beats")
            curve_ax.plot(result["times_s"], curve["arrangement_change"], color=colors.get(scale, "#d9480f"),
                          alpha=.6, linestyle="--", label=f"arrangement {scale} beats")
        # Variants retain auxiliary activity arrays even when that policy does
        # not use them to propose candidates.
        for curve in result.get("activity_curves", []) if name == "sustained_activity" else []:
            curve_ax.plot(result["times_s"], curve["activity_change"], color="#2b8a3e", linestyle=":",
                          label=f"activity {curve['scale_beats']} beats")
        for axis in (curve_ax, candidate_ax):
            axis.axvline(guide, color="#212529", linestyle=":", linewidth=1.1)
            axis.set_xlim(*bounds)
            axis.grid(alpha=.2)
        curve_ax.set_ylim(0, 1)
        curve_ax.set_ylabel("contrast")
        curve_ax.set_title(name, loc="left", fontsize=10)
        curve_ax.legend(ncol=3, fontsize=7, loc="upper right")
        times = _candidate_points(result)
        candidate_ax.scatter(times, [0.5] * len(times), marker="|", s=180, color="#7b2cbf")
        candidate_ax.set_ylim(0, 1)
        candidate_ax.set_yticks([.5], ["candidate anchors"])
    axes[-1].set_xlabel("song seconds; dotted black = approximate human guide")
    fig.suptitle("Fixed local-structure diagnostic; supports are feature context, not event extent")
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(output / "arrangement-curves-and-candidates.png", dpi=150)
    fig.savefig(output / "arrangement-curves-and-candidates.svg", metadata={"Date": None})
    plt.close(fig)


def run(repo: Path, registration_path: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    registration, inputs = load_registration(repo, registration_path)
    bounds = (float(registration["bounds_s"][0]), float(registration["bounds_s"][1]))
    source_record = next(item for item in inputs if item["path"] == SOURCE)
    stem_paths, beats, duration = validate_cache(repo, bounds, source_record["sha256"])
    started = time.monotonic()
    features, rms, timing = extract(stem_paths, beats)
    control = detect_local_structure(features, rms, beats)
    separate = detect_local_structure_variant(features, rms, beats, variant="separate_channels")
    sustained = detect_local_structure_variant(features, rms, beats, variant="sustained_activity")
    visible = {"control": visible_result(control, bounds),
               "separate_channels": visible_result(separate, bounds),
               "sustained_activity": visible_result(sustained, bounds)}
    guide = float(registration["human_song_time_s"])
    evidence = nearest_human_evidence(visible["control"], visible["separate_channels"],
                                      visible["sustained_activity"], guide)

    output.mkdir(parents=True)
    snapshots = output / "snapshots"
    snapshots.mkdir()
    for source, name in ((Path(__file__), "evaluate_arctic_arrangement.py"),
                         (repo / "experiments/extract_arrangement_reserve.py", "extract_arrangement_reserve.py"),
                         (repo / "songviz/local_structure.py", "local_structure.py"),
                         (repo / "songviz/local_structure_variants.py", "local_structure_variants.py"),
                         (registration_path, "registration.json")):
        shutil.copy2(source, snapshots / name)
    # Save full-track reusable features before the display-only filtered results.
    np.savez_compressed(output / "features.npz", beat_times_s=beats,
                        **{f"{name}_features": features[name] for name in STEMS},
                        **{f"{name}_rms": rms[name] for name in STEMS})
    timing = dict(timing) | {"full_track_duration_s": duration,
                             "registered_bounds_s": {"start_s": bounds[0], "end_s": bounds[1]}}
    write_json(output / "timing.json", timing)
    records = {"schema_version": 1, "kind": "songviz-arctic-fixed-arrangement-diagnostic",
               "status": "fixed-method-diagnostic", "registration": {"bounds_s": list(bounds), "human_song_time_s": guide},
               "method": {"control": "songviz.local_structure.detect_local_structure default config",
                          "separate_channels": "songviz.local_structure_variants.detect_local_structure_variant(separate_channels) default config",
                          "sustained_activity": "songviz.local_structure_variants.detect_local_structure_variant(sustained_activity) default config",
                          "control_config": asdict(LocalStructureConfig()),
                          "variant_config": asdict(LocalStructureVariantConfig())},
               "limitations": ["No accuracy, hit, miss, section-identity, or importance claim is computed.",
                               "Human timing is an approximate guide, not exact ground truth.",
                               "Support describes detector input context, not event extent.",
                               "No held-out claim is made; historical source exposure is unknown."],
               "eligible_component_curves_and_candidates": visible,
               "nearest_human_guide_anchor": evidence}
    write_json(output / "records.json", records)
    plot(output, list(visible.items()), bounds, guide)
    elapsed = time.monotonic() - started
    outputs = [record(repo, path) for path in sorted(output.rglob("*")) if path.is_file()]
    manifest = {"schema_version": 1, "kind": records["kind"], "status": records["status"],
                "inputs": inputs, "source_fingerprint": source_record,
                "registration_fingerprint": record(repo, registration_path),
                "output_fingerprints": outputs,
                "runtime": {"device": "cpu", "elapsed_s": elapsed,
                            "maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                            "thread_env": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")}},
                "versions": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
                             "soundfile": sf.__version__, "librosa": version("librosa"), "scipy": version("scipy")},
                "created_at": datetime.now(timezone.utc).isoformat(),
                "registered_bounds_s": list(bounds), "human_song_time_s": guide}
    write_json(output / "manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    repo = arguments.repo.resolve()
    registration_file = arguments.registration.resolve()
    if repo != registration_file and repo not in registration_file.parents:
        raise ValueError("Registration must be inside --repo")
    run(repo, registration_file, arguments.output.resolve())


if __name__ == "__main__":
    main()
