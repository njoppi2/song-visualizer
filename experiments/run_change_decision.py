#!/usr/bin/env python3
"""Run the single frozen doc-33 screen on saved features; join references last."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.change_decision import compute_change_decision

CACHES = {
    "development": ("outputs/reviews/structure-evaluation-03", "f7d3290b1ed33f7f66a469dedd6bd7714280dc5d3667023d9461ae4a3137192d", "6cc0a2215a16cb420c02dd6c52fca4ea782fbbad61dfb1909e4b07b82714ac7e"),
    "agnes": ("outputs/reviews/arrangement-continuity-reserve-features-01", "8a46795f3d4890b6fa694d5be8b992d4f47093e50adf44e95394496880b682a5", "6366a5fedce68d2a7fcd5649b38263298c532081f5ce3f9f62b596564af1a965"),
    "arctic": ("outputs/reviews/transition-components-arctic-arrangement-01", "29935b555b8d5d22111ff05c8ebbe2a5a270de4529c01e572901dafbb7c78d2e", "0c474eaad61528ed8f1bfac6e18a9900cb3b2ee60254cc96878b745381074c64"),
}
RESERVES = {"agnes": (93.294263, 117.294263), "arctic": (124.197052154195, 148.197052154195)}
BENCHMARK = "outputs/reviews/development-benchmark-02/benchmark.json"
FEEDBACK = {
    "agnes": "outputs/reviews/arrangement-continuity-listening-01/human-feedback.json",
    "arctic": "outputs/reviews/transition-components-arctic-intake-01/human-feedback.json",
}
PINS = {
    BENCHMARK: "223e19591f21dcb009a801447094f8754a3e1e4cd947647afe4334839b0eac79",
    FEEDBACK["agnes"]: "1f358b6681b6d6af9fefb1ef2acb7188d2835c2843a5f2e1f25d9ed0de78561f",
    FEEDBACK["arctic"]: "7f7580eaccdf04308299d47b1b3ee3212850d6580a3d04aecf3c3ae538d40950",
    "outputs/reviews/arrangement-continuity-01/development-predictions.json": "25ea51d02897211ea1b17abfcf090b2b2c030a6a5837676d5bba5cf5bb558e18",
    "outputs/reviews/arrangement-continuity-01/reserve-predictions.json": "5b65171d5c40b71c6665053c2f0791b8ab9a996ddd293fb6074222674838c98d",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def mask_reserve(result, bounds):
    result["anchors"] = [r if r is not None and r["support_start_s"] >= bounds[0]
                         and r["support_end_s"] <= bounds[1] else None for r in result["anchors"]]
    used = {r["anchor_index"] for r in result["anchors"] if r is not None}
    used |= {k + 1 for k in list(used)}
    for scale in result["scales"]:
        scale["samples"] = [r if k in used else None for k, r in enumerate(scale["samples"])]


def counts(rows):
    return {"eligible_anchors": len(rows), "level_anchors": sum(bool(r["level_changes"]) for r in rows),
            "pattern_anchors": sum(bool(r["pattern_changes"]) for r in rows),
            "pattern_unknown_anchors": sum(r["pattern_signal"] is None for r in rows)}


def describe_case(name, track, bounds, target, reference, prediction):
    bt = np.asarray(prediction["beat_times_s"])
    center = int(np.argmin(np.abs(bt - target)))
    indices = list(range(max(0, center - 2), min(len(bt), center + 3)))
    rows = [r for r in prediction["anchors"] if r is not None and bounds[0] <= r["anchor_s"] <= bounds[1]]
    probe = [prediction["anchors"][k] for k in indices if prediction["anchors"][k] is not None]
    return {"name": name, "track": track, "bounds_s": list(bounds), "reference_anchor_s": target,
            "reference": reference, "probe_indices": indices, "probe": counts(probe),
            "probe_rows": probe, "excerpt": counts(rows), "excerpt_rows": rows,
            "probe_samples": [{"scale_beats": s["scale_beats"], "samples": [s["samples"][k] for k in indices]}
                              for s in prediction["scales"]],
            "musical_acceptance": "unresolved", "acceptance_reason": "Acoustic signals do not assert the musical distinction; unknown is not pass."}


def plot(output, cases):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(6, 1, figsize=(11, 10))
    for ax, c in zip(axes, cases):
        rows = c["excerpt_rows"]
        for y, key, color in [(0, "level_changes", "#777777"), (1, "pattern_changes", "#087f8c")]:
            ax.scatter([r["anchor_s"] for r in rows], [y] * len(rows), color="#dddddd", marker="|", s=25)
            yes = [r["anchor_s"] for r in rows if r[key]]
            ax.scatter(yes, [y] * len(yes), color=color, s=22)
        ax.axvline(c["reference_anchor_s"], color="#b37900", linestyle=":")
        ax.set(xlim=c["bounds_s"], ylim=(-.4, 1.4), yticks=[0, 1], yticklabels=["Level", "Pattern"], xlabel="Song seconds")
        ax.set_title(c["name"], loc="left", fontsize=10)
    fig.suptitle("Fixed distribution-contrast screen: overlapping anchors, not events\nPale ticks: eligible anchors (including unknown pattern); dotted line: reference guide")
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(output / "comparison.png", dpi=150)
    fig.savefig(output / "comparison.svg", metadata={"Date": None})
    plt.close(fig)


def run(repo, output):
    if output.exists():
        raise FileExistsError(output)
    verified = {}

    def verify(path, expected):
        path = path if path.is_absolute() else repo / path
        if str(path) not in verified:
            verified[str(path)] = digest(path)
        actual = verified[str(path)]
        if actual != expected:
            raise ValueError(f"Fingerprint mismatch: {path}")

    for name, expected in PINS.items():
        verify(Path(name), expected)
    for folder, manifest_hash, feature_hash in CACHES.values():
        package = repo / folder
        verify(package / "manifest.json", manifest_hash)
        verify(package / "features.npz", feature_hash)
        m = json.loads((package / "manifest.json").read_text())
        for field in ("sources", "inputs", "outputs", "input_snapshots", "output_fingerprints"):
            records = m.get(field, [])
            if isinstance(records, dict):
                base = package if field == "outputs" else repo
                records = [{"path": str(base / n), "sha256": h} for n, h in records.items()]
            for record in records:
                verify(Path(record["path"]), record["sha256"])
    output.mkdir(parents=True)
    for source in (Path(__file__), repo / "experiments/change_decision.py", repo / "songviz/local_structure.py",
                   repo / "songviz/local_structure_variants.py", repo / "docs/33_change_decision_screen.md"):
        shutil.copy2(source, output / source.name)
    save(output / "registration.json", {"config_source": "change_decision.py", "design_sha256": digest(output / "33_change_decision_screen.md"),
          "code_sha256": digest(output / "change_decision.py"), "scope": "Six seen references; fixed engineering screen, no semantic classifier", "inputs": verified})
    predictions = {}
    for track, (folder, _, _) in CACHES.items():
        with np.load(repo / folder / "features.npz", allow_pickle=False) as z:
            stems = ("bass", "drums", "other", "vocals")
            predictions[track] = compute_change_decision({s: z[s + "_features"] for s in stems},
                                                        {s: z[s + "_rms"] for s in stems}, z["beat_times_s"])
        if track in RESERVES:
            mask_reserve(predictions[track], RESERVES[track])
        save(output / f"{track}-predictions.json", predictions[track])
    save(output / "predictions-frozen.json", {t: digest(output / f"{t}-predictions.json") for t in predictions})

    # Join references only after candidate bytes are saved. Analyst exposure is declared.
    benchmark = json.loads((repo / BENCHMARK).read_text())
    cases = []
    for c in benchmark["cases"]:
        b = c["context"]["excerpt_bounds_s"]
        cases.append(describe_case(c["evaluation_reference_id"], "development", (b["start_s"], b["end_s"]),
                                  c["context"]["fixed_anchor_s"], {"feedback": c["listener_feedback"], "rubric": c["rubric"]}, predictions["development"]))
    for track in RESERVES:
        reference = json.loads((repo / FEEDBACK[track]).read_text())
        target = reference["observation"]["song_time_s_approx"]
        cases.append(describe_case(track, track, RESERVES[track], target, reference, predictions[track]))
    parity = []
    for track, filename in [("development", "development"), ("agnes", "reserve")]:
        old = json.loads((repo / f"outputs/reviews/arrangement-continuity-01/{filename}-predictions.json").read_text())
        for k, r in enumerate(old["anchors"]):
            if r is None:
                continue
            new = predictions[track]["anchors"][k]
            if new is None or bool(new["level_changes"]) != r["level_change"] or new["level_changes"] != r["changed_layers"]:
                parity.append({"track": track, "anchor_index": k})
    by_name = {c["name"]: c for c in cases}
    def increase(case, stem=None):
        return any(change["direction"] == "increase" and (stem is None or change["stem"] == stem)
                   for r in case["probe_rows"] for change in r["level_changes"])
    criteria = {"level_parity": not parity, "drum_increase_retained": increase(by_name["drum-entry"], "drums"),
                "arctic_increase_retained": increase(by_name["arctic"]),
                "agnes_pattern_added": by_name["agnes"]["probe"]["pattern_anchors"] > 0,
                "nonchange_probe_no_pattern_added": by_name["within-passage"]["probe"]["pattern_anchors"] == 0}
    evaluation = {"narrow_screen": criteria, "narrow_screen_pass": all(criteria.values()), "level_parity_mismatches": parity,
                  "cases": cases, "semantic_acceptance": "unresolved for all six cases; no semantic predictions", "no_promotion": True}
    save(output / "evaluation.json", evaluation)
    plot(output, cases)
    lines = ["# Fixed change-decision screen", "", f"Narrow engineering screen passes: {all(criteria.values())}.",
             "All musical acceptance remains unresolved. No production promotion or threshold tuning.", "",
             "| Case | Eligible excerpt anchors | Level | Pattern | Unknown pattern | Probe level / pattern |",
             "| --- | ---: | ---: | ---: | ---: | --- |"]
    for c in cases:
        e, p = c["excerpt"], c["probe"]
        lines.append(f"| {c['name']} | {e['eligible_anchors']} | {e['level_anchors']} | {e['pattern_anchors']} | {e['pattern_unknown_anchors']} | {p['level_anchors']} / {p['pattern_anchors']} |")
    (output / "report.md").write_text("\n".join(lines) + "\n")
    save(output / "manifest.json", {"inputs": verified, "outputs": {p.name: digest(p) for p in output.iterdir() if p.is_file()}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.repo.resolve(), args.output.resolve())
