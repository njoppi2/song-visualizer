#!/usr/bin/env python3
"""Run the fixed two-arm experiment; join references only after saving predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arrangement_continuity import compute_arrangement_continuity

PINS = {
    "outputs/reviews/structure-evaluation-03/manifest.json": "f7d3290b1ed33f7f66a469dedd6bd7714280dc5d3667023d9461ae4a3137192d",
    "outputs/reviews/structure-evaluation-03/features.npz": "6cc0a2215a16cb420c02dd6c52fca4ea782fbbad61dfb1909e4b07b82714ac7e",
    "outputs/reviews/structure-evaluation-03/timing.json": "835498972abc87ea6b972611ac18f57d70be298a7897b3e511405cbe18d4e660",
    "outputs/reviews/arrangement-continuity-registration-01/protocol.md": "06bc4bb0aef7658871ef59b4b68a601b3ed92c08175f905763f230bc54ca2307",
}
BENCHMARK = "outputs/reviews/development-benchmark-02/benchmark.json"
BENCHMARK_HASH = "223e19591f21dcb009a801447094f8754a3e1e4cd947647afe4334839b0eac79"
RESERVE = (93.294263, 117.294263)
RESERVE_MANIFEST_HASH = "8a46795f3d4890b6fa694d5be8b992d4f47093e50adf44e95394496880b682a5"
STEMS = ("bass", "drums", "other", "vocals")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(path, expected):
    if digest(path) != expected:
        raise ValueError(f"Fingerprint mismatch: {path}")


def save(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def predict(path):
    with np.load(path, allow_pickle=False) as data:
        return compute_arrangement_continuity(
            {name: data[f"{name}_features"] for name in STEMS},
            {name: data[f"{name}_rms"] for name in STEMS}, data["beat_times_s"])


def in_reserve(row):
    return (row is not None and row["support_union_start_s"] >= RESERVE[0]
            and row["support_union_end_s"] <= RESERVE[1])


def summarize(rows):
    return {
        "eligible_anchors": len(rows),
        "level_only_anchors": sum(row["level_change"] for row in rows),
        "continuity_supported_anchors": sum(row["continuity_support"] for row in rows),
        "arrangement_candidate_anchors": sum(row["arrangement_candidate"] for row in rows),
        "anchors": rows,
    }


def plot(output, development, reserve, cases):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(5, 1, figsize=(11, 11))
    panels = [(c["evaluation_reference_id"], c["context"]["excerpt_bounds_s"],
               development["anchors"], c["context"]["focus_bounds_s"]) for c in cases]
    panels.append(("Agnes — prospective reserve (unscored)",
                   {"start_s": RESERVE[0], "end_s": RESERVE[1]}, reserve["anchors"], None))
    for ax, (name, bounds, records, focus) in zip(axes, panels):
        rows = [r for r in records if r is not None and bounds["start_s"] <= r["anchor_s"] <= bounds["end_s"]]
        for y, key, color in [(1, "level_change", "#747a88"), (0, "arrangement_candidate", "#057e79")]:
            yes = [r["anchor_s"] for r in rows if r[key]]
            no = [r["anchor_s"] for r in rows if not r[key]]
            ax.scatter(no, [y] * len(no), marker="|", color="#d0d3d9", s=25)
            ax.scatter(yes, [y] * len(yes), color=color, s=25)
        if focus:
            ax.axvspan(focus["start_s"], focus["end_s"], alpha=.1, color="#dca530")
        ax.set(xlim=(bounds["start_s"], bounds["end_s"]), ylim=(-.5, 1.5),
               yticks=[0, 1], yticklabels=["+ Continuity gate", "Level-only"], xlabel="Song seconds")
        ax.set_title(name, loc="left")
        ax.grid(axis="x", alpha=.2)
    fig.suptitle("Fixed arrangement-candidate comparison\nDots: qualifying anchors; pale ticks: eligible rejected anchors; shading: existing focus bands.")
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(output / "comparison.png", dpi=150)
    fig.savefig(output / "comparison.svg", metadata={"Date": None})
    plt.close(fig)


def run(repo, output, reserve_package):
    if output.exists():
        raise FileExistsError(output)
    for name, pin in PINS.items():
        verify(repo / name, pin)
    development_manifest = json.loads((repo / next(iter(PINS))).read_text())
    # Verify actual source media records underpinning the accepted development cache.
    for record in development_manifest["sources"]:
        p = Path(record["path"])
        if p.suffix.lower() in (".flac", ".wav") and ("/stems/" in str(p) or p.suffix.lower() == ".flac"):
            verify(p if p.is_absolute() else repo / p, record["sha256"])
    verify(reserve_package / "manifest.json", RESERVE_MANIFEST_HASH)
    reserve_manifest = json.loads((reserve_package / "manifest.json").read_text())
    if reserve_manifest.get("status") != "complete-feature-extraction-only":
        raise ValueError("Reserve extraction must be complete")
    if reserve_manifest.get("reserve_s") != {"start_s": RESERVE[0], "end_s": RESERVE[1]}:
        raise ValueError("Reserve bounds differ from frozen protocol")
    for record in reserve_manifest["outputs"] + reserve_manifest["inputs"] + reserve_manifest["sources"]:
        path = Path(record["path"])
        verify(path if path.is_absolute() else repo / path, record["sha256"])
    output.mkdir(parents=True)
    shutil.copy2(Path(__file__), output / "run_arrangement_continuity.py")
    shutil.copy2(Path(__file__).with_name("arrangement_continuity.py"), output / "arrangement_continuity.py")
    shutil.copy2(repo / "outputs/reviews/arrangement-continuity-registration-01/protocol.md", output / "protocol.md")
    shutil.copy2(repo / "outputs/reviews/structure-evaluation-03/features.npz", output / "development-features.npz")
    shutil.copy2(reserve_package / "features.npz", output / "reserve-features.npz")
    registration = {"protocol_sha256": digest(output / "protocol.md"),
                    "module_sha256": digest(output / "arrangement_continuity.py"),
                    "runner_sha256": digest(output / "run_arrangement_continuity.py"),
                    "reserve_manifest_sha256": digest(reserve_package / "manifest.json"),
                    "reference_loaded": False, "reserve_bounds_s": list(RESERVE)}
    save(output / "registration.json", registration)
    development = predict(output / "development-features.npz")
    reserve = predict(output / "reserve-features.npz")
    reserve["anchors"] = [r if in_reserve(r) else None for r in reserve["anchors"]]
    used = {r["anchor_index"] for r in reserve["anchors"] if r is not None}
    used |= {k + 1 for k in list(used)}
    for scale in reserve["scales"]:
        scale["samples"] = [r if k in used else None for k, r in enumerate(scale["samples"])]
    # Prediction bytes are committed before the evaluator opens any reference.
    save(output / "development-predictions.json", development)
    save(output / "reserve-predictions.json", reserve)
    prediction_pins = {name: digest(output / name) for name in ("development-predictions.json", "reserve-predictions.json")}
    save(output / "predictions-frozen.json", prediction_pins)

    verify(repo / BENCHMARK, BENCHMARK_HASH)
    benchmark = json.loads((repo / BENCHMARK).read_text())
    cases = []
    for c in benchmark["cases"]:
        joined = {"case_id": c["case_id"], "name": c["evaluation_reference_id"], "reference_context": c["context"]}
        for name, bounds_key in [("focus", "focus_bounds_s"), ("excerpt", "excerpt_bounds_s")]:
            bounds = c["context"][bounds_key]
            rows = [r for r in development["anchors"] if r is not None and bounds["start_s"] <= r["anchor_s"] <= bounds["end_s"]]
            joined[name] = summarize(rows)
        cases.append(joined)
    no_change = next(c for c in cases if c["name"] == "within-passage")
    drums = next(c for c in cases if c["name"] == "drum-entry")
    drum_hit = any(r["arrangement_candidate"] and {"stem": "drums", "direction": "increase"} in r["changed_layers"] for r in drums["focus"]["anchors"])
    quiet = (no_change["focus"]["eligible_anchors"] > 0
             and no_change["focus"]["arrangement_candidate_anchors"] == 0)
    evaluation = {"cases": cases, "narrow_development_criteria": {
        "drum_increase_with_continuity": drum_hit, "no_within_passage_candidate": quiet,
        "both_met": drum_hit and quiet, "candidate_count_lower_than_level_only": no_change["focus"]["level_only_anchors"] > no_change["focus"]["arrangement_candidate_anchors"]},
        "reserve": summarize([r for r in reserve["anchors"] if r is not None]),
        "reference_sha256": BENCHMARK_HASH,
        "semantic_benchmark_credit": "Unchanged: no automatic identity, significance, vocal-behavior or transition-extent pass.",
        "reserve_evaluation": "No listener labels; no accuracy/generalization claim. No scores outside the reserve are retained."}
    save(output / "evaluation.json", evaluation)
    plot(output, development, reserve, benchmark["cases"])
    lines = ["# Fixed arrangement/continuity comparison", "", "| Development case | Eligible focus anchors | Level-only proposals | With continuity gate |", "| --- | ---: | ---: | ---: |"]
    for c in cases:
        f = c["focus"]
        lines.append(f"| {c['name']} | {f['eligible_anchors']} | {f['level_only_anchors']} | {f['arrangement_candidate_anchors']} |")
    lines += ["", f"Drum-increase criterion: **{drum_hit}**. Within-passage zero-proposal criterion: **{quiet}**.",
              "These are fixed, seen-development checks. Matching zeroes are not an improvement. All eligible/rejected anchors and their source supports are in evaluation.json.",
              "", f"Reserved Agnes passage: {evaluation['reserve']['eligible_anchors']} eligible anchors; {evaluation['reserve']['level_only_anchors']} level-only and {evaluation['reserve']['arrangement_candidate_anchors']} continuity-gated proposals. It has no listener labels and is unscored.",
              "", "[Comparison plot](comparison.png). Source windows overlap; proposal counts are not independent events. Continuity support means stable ordered acoustic layers, not established musical identity or importance. No production policy is promoted."]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    for name, pin in prediction_pins.items():
        verify(output / name, pin)
    save(output / "manifest.json", {"kind": "songviz-arrangement-continuity-comparison", "status": "complete",
         "sources": PINS, "benchmark_sha256": BENCHMARK_HASH,
         "reserve_manifest_sha256": registration["reserve_manifest_sha256"], "numpy": np.__version__,
         "outputs": {p.name: digest(p) for p in output.iterdir() if p.is_file()}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reserve-package", type=Path, required=True)
    a = parser.parse_args()
    run(a.repo.resolve(), a.output.resolve(), a.reserve_package.resolve())
