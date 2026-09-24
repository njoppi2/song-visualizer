#!/usr/bin/env python3
"""Saved-feature diagnostic for local-structure contrast components.

This is a seen, exploratory diagnostic.  It does not change detector defaults,
thresholds, production policy, or the existing benchmark.  Its sole purpose is
to preserve the default detector's pattern and arrangement curves separately,
then compare its frozen control with the already-implemented separate-channel
variant around the four development excerpts and the now-seen Agnes reserve.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from songviz.local_structure import detect_local_structure
from songviz.local_structure_variants import detect_local_structure_variant


DEVELOPMENT = "outputs/reviews/structure-evaluation-03"
RESERVE_PACKAGE = "outputs/reviews/arrangement-continuity-reserve-features-01"
LOCAL_CONTROL = "outputs/reviews/local-structure-02"
ARRANGEMENT = "outputs/reviews/arrangement-continuity-01"
BENCHMARK = "outputs/reviews/development-benchmark-02/benchmark.json"
FEEDBACK = "outputs/reviews/arrangement-continuity-listening-01/human-feedback.json"
RESERVE = (93.294263, 117.294263)
HUMAN_POINT = 105.294263

PINS = {
    f"{DEVELOPMENT}/manifest.json": "f7d3290b1ed33f7f66a469dedd6bd7714280dc5d3667023d9461ae4a3137192d",
    f"{DEVELOPMENT}/features.npz": "6cc0a2215a16cb420c02dd6c52fca4ea782fbbad61dfb1909e4b07b82714ac7e",
    f"{RESERVE_PACKAGE}/manifest.json": "8a46795f3d4890b6fa694d5be8b992d4f47093e50adf44e95394496880b682a5",
    f"{RESERVE_PACKAGE}/features.npz": "6366a5fedce68d2a7fcd5649b38263298c532081f5ce3f9f62b596564af1a965",
    f"{LOCAL_CONTROL}/manifest.json": "61e3aa311531dfe536da51164be1fa3e5865520a21b96238815a0a1c33f21d8e",
    f"{LOCAL_CONTROL}/predictions.json": "eedc48cc0cb10c1e8d6ba6e2f434315ef642d29aa514157f729abafeaf659f08",
    f"{ARRANGEMENT}/manifest.json": "f077d3025280a6aab5eb3330ba9ed226d718e26b396442ac6dac00c5f43a4a70",
    BENCHMARK: "223e19591f21dcb009a801447094f8754a3e1e4cd947647afe4334839b0eac79",
    FEEDBACK: "1f358b6681b6d6af9fefb1ef2acb7188d2835c2843a5f2e1f25d9ed0de78561f",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def resolve(repo: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo / path


def verify(path: Path, expected: str) -> None:
    actual = digest(path)
    if actual != expected:
        raise ValueError(f"Fingerprint mismatch: {path} ({actual}, expected {expected})")


def verify_records(repo: Path, package: Path, manifest: dict, fields: tuple[str, ...]) -> None:
    """Verify recorded source/input/output links without trusting cache names."""
    seen: set[Path] = set()
    for field in fields:
        records = manifest.get(field, [])
        if isinstance(records, dict):
            # The frozen comparison stores output names relative to its package,
            # but source names relative to the repository.
            base = package if field == "outputs" else repo
            records = [{"path": base / name, "sha256": value} for name, value in records.items()]
        for record in records:
            if not isinstance(record, dict) or "path" not in record or "sha256" not in record:
                raise ValueError(f"Malformed {field} record in manifest")
            path = resolve(repo, record["path"])
            if path not in seen:
                verify(path, record["sha256"])
                seen.add(path)


def load_features(path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        names = sorted(key[:-9] for key in data.files if key.endswith("_features"))
        expected = {"beat_times_s"} | {f"{name}_{kind}" for name in names for kind in ("features", "rms")}
        if not names or set(data.files) != expected:
            raise ValueError(f"Unexpected saved-feature schema: {path}")
        return ({name: data[f"{name}_features"] for name in names},
                {name: data[f"{name}_rms"] for name in names}, data["beat_times_s"])


def inside_support(candidate: dict, bounds: tuple[float, float]) -> bool:
    return (candidate.get("support_start_s") is not None and candidate.get("support_end_s") is not None
            and candidate["support_start_s"] >= bounds[0] and candidate["support_end_s"] <= bounds[1])


def filtered_candidates(result: dict, bounds: tuple[float, float] | None = None) -> list[dict]:
    candidates = result["changes"] + result["transitions"]
    if bounds is not None:
        candidates = [candidate for candidate in candidates if inside_support(candidate, bounds)]
    return candidates


def excerpt_candidates(result: dict, bounds: tuple[float, float]) -> list[dict]:
    """Development excerpts select anchors and retain their untrimmed support."""
    return [candidate for candidate in result["changes"] + result["transitions"]
            if bounds[0] <= candidate["time_s"] <= bounds[1]]


def reserve_visible(result: dict) -> dict:
    """Mask non-reserve values while retaining thresholds computed over the full track."""
    visible = {key: value for key, value in result.items()
               if key not in {"channel_curves", "activity_curves", "changes", "transitions", "curves"}}
    curves = []
    maximum_scale = max(curve["scale_beats"] for curve in result["curves"])
    eligible = [index >= maximum_scale and index + maximum_scale < len(result["times_s"])
                and result["times_s"][index - maximum_scale] >= RESERVE[0]
                and result["times_s"][index + maximum_scale] <= RESERVE[1]
                for index in range(len(result["times_s"]))]
    eligible_indices = {index for index, allowed in enumerate(eligible) if allowed}
    for curve in result["curves"]:
        copied = dict(curve)
        for key in ("pattern_change", "arrangement_change", "combined_change"):
            copied[key] = [value if eligible[index] else None for index, value in enumerate(curve[key])]
        curves.append(copied)
    visible["curves"] = curves
    visible["changes"] = [candidate for candidate in result["changes"]
                          if inside_support(candidate, RESERVE) and candidate["beat_index"] in eligible_indices]
    visible["transitions"] = [candidate for candidate in result["transitions"]
                              if inside_support(candidate, RESERVE) and candidate.get("start_beat") in eligible_indices]
    return visible


def nearest_curve_scores(result: dict, requested_s: float, channel_curves: list[dict] | None = None) -> list[dict]:
    times = result["times_s"]
    index = min(range(len(times)), key=lambda item: (abs(times[item] - requested_s), item))
    return [{"scale_beats": curve["scale_beats"], "requested_s": requested_s,
             "anchor_index": index, "anchor_s": times[index],
             "support_start_s": times[index - curve["scale_beats"]] if index >= curve["scale_beats"] else None,
             "support_end_s": (times[index + curve["scale_beats"]]
                               if index + curve["scale_beats"] < len(times) else None),
             "pattern_change": curve["pattern_change"][index],
             "arrangement_change": curve["arrangement_change"][index],
             "combined_change": curve["combined_change"][index], "threshold": curve["threshold"],
             "per_stem": (channel_curves[position]["stem_evidence"][index]
                          if channel_curves is not None else None)}
            for position, curve in enumerate(result["curves"])]


def extrema(result: dict, bounds: tuple[float, float]) -> dict:
    answer: dict[str, dict] = {}
    for curve in result["curves"]:
        rows = [(index, value) for index, value in enumerate(curve["pattern_change"])
                if value is not None and bounds[0] <= result["times_s"][index] <= bounds[1]]
        arrangement = [(index, value) for index, value in enumerate(curve["arrangement_change"])
                       if value is not None and bounds[0] <= result["times_s"][index] <= bounds[1]]
        def maximum(values: list[tuple[int, float]]) -> dict | None:
            if not values:
                return None
            score = max(value for _, value in values)
            # All ties are retained explicitly rather than presenting one as unique.
            tied = [index for index, value in values if value == score]
            return {"score": score, "anchor_indices": tied,
                    "anchor_times_s": [result["times_s"][index] for index in tied]}
        answer[str(curve["scale_beats"])] = {"pattern_change": maximum(rows),
                                               "arrangement_change": maximum(arrangement)}
    return answer


def level_anchors(predictions: dict, bounds: tuple[float, float]) -> list[float]:
    return [row["anchor_s"] for row in predictions["anchors"] if row is not None and row["level_change"]
            and bounds[0] <= row["anchor_s"] <= bounds[1]]


def case_record(name: str, bounds: tuple[float, float], fixed_anchor_s: float | None,
                control: dict, separate: dict, frozen_levels: dict, nearest_scores: list[dict] | None = None) -> dict:
    probes = ([] if fixed_anchor_s is None else nearest_curve_scores(
        control, fixed_anchor_s, separate.get("channel_curves"))) if nearest_scores is None else nearest_scores
    return {"name": name, "bounds_s": {"start_s": bounds[0], "end_s": bounds[1]},
            "fixed_anchor_s": fixed_anchor_s, "control_nearest_anchor_scores": probes,
            "control_extrema": extrema(control, bounds), "separate_channel_extrema": extrema(separate, bounds),
            "control_candidates": excerpt_candidates(control, bounds),
            "separate_channel_candidates": excerpt_candidates(separate, bounds),
            "frozen_level_only_anchors_s": level_anchors(frozen_levels, bounds)}


def plot(output: Path, panels: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(5, 1, figsize=(13, 16), sharey=True)
    colors = {2: ("#0b7285", "#d9480f"), 4: ("#1971c2", "#e8590c"), 8: ("#6741d9", "#c92a2a")}
    for ax, panel in zip(axes, panels):
        result, start, end = panel["control"], *panel["bounds"]
        for curve in result["curves"]:
            scale = curve["scale_beats"]
            rows = [(time, pattern, arrangement) for time, pattern, arrangement in zip(
                result["times_s"], curve["pattern_change"], curve["arrangement_change"])
                if start <= time <= end]
            if rows:
                times, pattern, arrangement = zip(*rows)
                ax.plot(times, pattern, color=colors[scale][0], label=f"pattern {scale} beats")
                ax.plot(times, arrangement, color=colors[scale][1], linestyle="--", label=f"arrangement {scale} beats")
        for time in panel["levels"]:
            ax.axvline(time, color="#5f3dc4", alpha=.5, linewidth=.8)
        if panel.get("guide") is not None:
            ax.axvline(panel["guide"], color="#212529", linestyle=":", linewidth=1.2)
        ax.set_xlim(start, end)
        ax.set_ylim(0, 1)
        ax.set_title(panel["name"], loc="left", fontsize=10)
        ax.grid(alpha=.2)
        ax.set_ylabel("contrast")
    axes[0].legend(ncol=3, fontsize=8, loc="upper right")
    axes[-1].set_xlabel("song seconds")
    fig.suptitle("Default local-structure component curves; purple lines: frozen level-only anchors\n"
                 "Pattern is mean log-CQT contrast, not melody or section identity.")
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(output / "component-curves.png", dpi=150)
    fig.savefig(output / "component-curves.svg", metadata={"Date": None})
    plt.close(fig)


def run(repo: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    for name, expected in PINS.items():
        verify(repo / name, expected)
    manifests = {name: json.loads((repo / name / "manifest.json").read_text())
                 for name in (DEVELOPMENT, RESERVE_PACKAGE, LOCAL_CONTROL, ARRANGEMENT)}
    for name, manifest in manifests.items():
        verify_records(repo, repo / name, manifest, ("sources", "inputs", "outputs", "input_snapshots"))
    if manifests[RESERVE_PACKAGE].get("reserve_s") != {"start_s": RESERVE[0], "end_s": RESERVE[1]}:
        raise ValueError("Reserve bounds differ from frozen package")
    benchmark = json.loads((repo / BENCHMARK).read_text())
    feedback = json.loads((repo / FEEDBACK).read_text())
    if feedback.get("observation", {}).get("song_time_s_approx") != HUMAN_POINT:
        raise ValueError("Human approximate guide differs from preserved feedback")

    development_inputs = load_features(repo / DEVELOPMENT / "features.npz")
    reserve_inputs = load_features(repo / RESERVE_PACKAGE / "features.npz")
    control = detect_local_structure(*development_inputs)
    frozen_control = json.loads((repo / LOCAL_CONTROL / "predictions.json").read_text())
    if json.loads(json.dumps(control, allow_nan=False)) != frozen_control:
        raise ValueError("Current default control does not reproduce frozen local-structure-02 predictions")
    separate = detect_local_structure_variant(*development_inputs, variant="separate_channels")
    raw_reserve_control = detect_local_structure(*reserve_inputs)
    raw_reserve_separate = detect_local_structure_variant(*reserve_inputs, variant="separate_channels")
    reserve_nearest = nearest_curve_scores(raw_reserve_control, HUMAN_POINT, raw_reserve_separate["channel_curves"])
    reserve_control = reserve_visible(raw_reserve_control)
    reserve_separate = reserve_visible(raw_reserve_separate)
    frozen_development = json.loads((repo / ARRANGEMENT / "development-predictions.json").read_text())
    frozen_reserve = json.loads((repo / ARRANGEMENT / "reserve-predictions.json").read_text())

    cases = [case_record(case["evaluation_reference_id"],
                         (case["context"]["excerpt_bounds_s"]["start_s"], case["context"]["excerpt_bounds_s"]["end_s"]),
                         case["context"]["fixed_anchor_s"], control, separate, frozen_development)
             for case in benchmark["cases"]]
    reserve = case_record("Agnes — now-seen exploratory diagnostic",
                          RESERVE, HUMAN_POINT, reserve_control, reserve_separate, frozen_reserve,
                          nearest_scores=reserve_nearest)
    reserve["control_candidates"] = filtered_candidates(reserve_control, RESERVE)
    reserve["separate_channel_candidates"] = filtered_candidates(reserve_separate, RESERVE)
    reserve["human_approximate_guide"] = {"song_time_s": HUMAN_POINT,
        "meaning": "guide only; not exact ground truth or event extent"}
    records = {"kind": "songviz-transition-component-diagnostic", "status": "seen-exploratory-diagnostic",
               "method": {"control": "songviz.local_structure.detect_local_structure default config",
                          "comparison": "songviz.local_structure_variants.detect_local_structure_variant(separate_channels) default config"},
               "limitations": ["No held-out claim. Focus bands prompt inspection; they are not event extents.",
                 "Feature support windows are not event extents. Nearest anchors are descriptive, not hit/miss scoring.",
                 "Pattern values are mean log-CQT contrast and are not perfectly level-invariant; they are not melody or section identity."],
               "control_reproduces_frozen_local_structure_02": True,
               "development": {"control": control, "separate_channels": separate, "cases": cases},
               "reserve": {"control": reserve_control, "separate_channels": reserve_separate, "case": reserve,
                 "reported_candidates_require_full_support_inside_reserve": True},
               "full_track_candidate_counts": {"development_control": len(control["changes"]) + len(control["transitions"]),
                    "development_separate_channels": len(separate["changes"]) + len(separate["transitions"]),
                    "Agnes_control_reported_reserve_only": len(reserve["control_candidates"]),
                    "Agnes_separate_channels_reported_reserve_only": len(reserve["separate_channel_candidates"])}}
    output.mkdir(parents=True)
    shutil.copy2(Path(__file__), output / "diagnose_transition_components.py")
    shutil.copy2(repo / "songviz/local_structure.py", output / "local_structure.py")
    shutil.copy2(repo / "songviz/local_structure_variants.py", output / "local_structure_variants.py")
    shutil.copy2(repo / "docs/32_transition_components.md", output / "design.md")
    save(output / "records.json", records)
    panels = [{"name": item["name"], "bounds": (item["bounds_s"]["start_s"], item["bounds_s"]["end_s"]),
               "control": control, "levels": item["frozen_level_only_anchors_s"], "guide": item["fixed_anchor_s"]}
              for item in cases]
    panels.append({"name": reserve["name"], "bounds": RESERVE, "control": reserve_control,
                   "levels": reserve["frozen_level_only_anchors_s"], "guide": HUMAN_POINT})
    plot(output, panels)
    save(output / "manifest.json", {"kind": records["kind"], "status": records["status"], "sources": PINS,
        "snapshots": {p.name: digest(p) for p in output.iterdir() if p.is_file()}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.repo.resolve(), arguments.output.resolve())
