"""Package saved MuQ frame features into a provenance-bound diagnostic review.

This script deliberately does not load a model or audio.  Its only model input is
the frame ``.npz`` emitted by the runtime worker, so a review cannot accidentally
turn a synthetic/unit-test array into a claim that MuQ ran.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from songviz.ingest import sha256_file

SOURCE_SHA256 = "657af9333edafbb6cf8cc573446650773d1169b0117c784106d1895cf802cb44"
FEEDBACK_SHA256 = "f69b14f0343d0ebe4af12d37f66e88c76aa5d8e7673f7529345333589d5c47b6"
MUQ_CHECKPOINT_ID = "OpenMuQ/MuQ-large-msd-iter"
MUQ_CHECKPOINT_REVISION = "0562a57814f6f8bbd9fdea0a25921a2fce1a841a"
MUQ_WEIGHTS_SHA256 = "273febab2be02872c37d2c37e48a9d6c52c1c9392f3eeeabd498efa281ccb7a6"
MUQ_CONFIG_SHA256 = "237335ee27d8fb951ce778701a12a79e06c51ae636dd786f97e45f51ce532543"
BASELINE_KINDS = {
    "structure": "songviz-structural-development-evaluation",
    "role": "songviz-role-context-review",
    "listening": "songviz-listening-examples",
}
REQUIRED_FRAME_KEYS = ("embeddings", "frame_times_s", "support_start_s", "support_end_s")


def record(path: Path) -> dict:
    path = path.resolve()
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def verify_record(row: dict) -> Path:
    if not isinstance(row, dict) or not isinstance(row.get("path"), str) or not isinstance(row.get("sha256"), str):
        raise ValueError("Malformed fingerprint record")
    path = Path(row["path"])
    path = path if path.is_absolute() else ROOT / path
    if not path.is_file() or sha256_file(path) != row["sha256"]:
        raise ValueError(f"Changed or missing fingerprinted input: {path}")
    return path.resolve()


def verified_package(package: Path, expected_kind: str) -> tuple[dict, list[Path]]:
    manifest_path = package / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Missing baseline manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("kind") != expected_kind:
        raise ValueError(f"Unexpected baseline package kind in {manifest_path}")
    consumed = [manifest_path.resolve()]
    # Frozen-package validation must cover each declared record, not merely its
    # manifest.  Older packages may have sources/snapshots/outputs only.
    for group in ("sources", "input_snapshots", "outputs"):
        rows = manifest.get(group)
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"Missing {group} in {manifest_path}")
        consumed.extend(verify_record(row) for row in rows)
    return manifest, consumed


def require_declared(manifest: dict, path: Path) -> None:
    """A consumed baseline file must be uniquely fingerprinted by its package."""
    found = []
    for group in ("sources", "input_snapshots", "outputs"):
        for row in manifest.get(group, []):
            try:
                if verify_record(row) == path.resolve():
                    found.append(row)
            except ValueError:
                raise
    if len(found) != 1:
        raise ValueError(f"Baseline manifest does not uniquely declare {path}")


def _bound_source(metadata: dict, *, label: str) -> None:
    source = metadata.get("source") if isinstance(metadata.get("source"), dict) else metadata
    values = [source.get(key) for key in ("source_audio_sha256", "audio_sha256", "sha256") if isinstance(source, dict)]
    if SOURCE_SHA256 not in values:
        raise ValueError(f"{label} is not bound to the required original source SHA-256")


def load_beats(structure: Path) -> np.ndarray:
    features = structure / "features.npz"
    if not features.is_file():
        raise ValueError("Structure baseline is missing features.npz")
    with np.load(features, allow_pickle=False) as saved:
        beats = np.asarray(saved["beat_times_s"], dtype=float)
    if beats.ndim != 1 or beats.size < 2 or not np.isfinite(beats).all() or np.any(np.diff(beats) <= 0):
        raise ValueError("Baseline beat clock must be finite and strictly increasing")
    return beats


def load_runtime(runtime: Path) -> tuple[dict, dict[str, np.ndarray], list[Path]]:
    manifest_path = runtime / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Runtime output requires manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("kind") != "songviz-muq-runtime" or manifest.get("schema_version") != 1:
        raise ValueError("Runtime manifest is not a songviz-muq-runtime v1 artifact")
    checkpoint, config, code = manifest.get("checkpoint"), manifest.get("config"), manifest.get("code")
    source = manifest.get("source")
    if (not isinstance(source, dict) or not isinstance(checkpoint, dict) or not checkpoint.get("id") or not checkpoint.get("revision") or not checkpoint.get("sha256")
            or not isinstance(config, dict) or not config.get("sha256")
            or not isinstance(code, dict) or not code.get("sha256")):
        raise ValueError("Runtime manifest lacks pinned checkpoint/config/code fingerprints")
    if (source.get("source_audio_sha256") != SOURCE_SHA256 or source.get("sha256") != SOURCE_SHA256
            or checkpoint.get("id") != MUQ_CHECKPOINT_ID or checkpoint.get("revision") != MUQ_CHECKPOINT_REVISION
            or checkpoint.get("sha256") != MUQ_WEIGHTS_SHA256 or config.get("sha256") != MUQ_CONFIG_SHA256):
        raise ValueError("Runtime manifest does not match pinned source/model/config")
    provenance_paths = [verify_record(row) for row in (source, checkpoint, config, code)]
    clock = manifest.get("clock")
    if (not isinstance(clock, dict) or clock.get("source_seconds") is not True
            or clock.get("frame_hz") != 25.0 or clock.get("chunk_seconds") != 10
            or clock.get("hop_seconds") != 5 or clock.get("ownership") != "closest eligible chunk center; earlier chunk wins exact ties"):
        raise ValueError("Runtime manifest must declare source_seconds clock")
    extraction = manifest.get("extraction")
    if (not isinstance(extraction, dict) or extraction.get("encoder_input_sample_rate") != 24000
            or extraction.get("dtype") != "float32" or extraction.get("eval") is not True
            or extraction.get("selected_layer") != "last_hidden_state" or extraction.get("source_cut_before_resample") is not True
            or extraction.get("mono") != "arithmetic mean of stereo float32 PCM"):
        raise ValueError("Runtime manifest does not satisfy the pinned MuQ extraction protocol")
    frames = manifest.get("frames")
    if not isinstance(frames, dict):
        raise ValueError("Runtime manifest lacks frames binding")
    frame_path = runtime / frames.get("path", "frames.npz")
    if not frame_path.is_file() or sha256_file(frame_path) != frames.get("sha256"):
        raise ValueError("Runtime frames.npz is missing or does not match manifest")
    if frames.get("source_audio_sha256") != SOURCE_SHA256:
        raise ValueError("Runtime frames are not bound to required source")
    with np.load(frame_path, allow_pickle=False) as data:
        if set(data.files) != set(REQUIRED_FRAME_KEYS):
            raise ValueError("frames.npz must contain exactly embeddings and frame/support clocks")
        if data["embeddings"].dtype != np.float32 or any(data[key].dtype != np.float64 for key in REQUIRED_FRAME_KEYS[1:]):
            raise ValueError("frames.npz does not preserve required fp32 embedding/fp64 clock dtypes")
        arrays = {key: np.asarray(data[key], dtype=float) for key in REQUIRED_FRAME_KEYS}
    emb = arrays["embeddings"]
    n = emb.shape[0] if emb.ndim == 2 else -1
    if n < 1 or emb.shape[1] < 1 or any(arrays[key].shape != (n,) for key in REQUIRED_FRAME_KEYS[1:]):
        raise ValueError("frames.npz dimensions do not align")
    if (not np.isfinite(emb).all() or any(not np.isfinite(arrays[key]).all() for key in REQUIRED_FRAME_KEYS[1:])
            or np.any(np.diff(arrays["frame_times_s"]) <= 0)
            or np.any(arrays["support_start_s"] > arrays["frame_times_s"])
            or np.any(arrays["frame_times_s"] > arrays["support_end_s"])):
        raise ValueError("frames.npz contains invalid source-time/support values")
    if frames.get("shape") != [n, emb.shape[1]] or frames.get("finite") is not True or emb.shape[1] != 1024:
        raise ValueError("Runtime frame metadata does not bind the actual array coverage")
    if (source.get("sample_rate") != 44100 or source.get("channels") != 2
            or not isinstance(source.get("frames"), int) or source["frames"] < 1
            or source.get("duration_s") != source["frames"] / source["sample_rate"]):
        raise ValueError("Runtime source clock is not the pinned stereo 44.1kHz source extent")
    from experiments.probe_music_representation import chunk_geometry
    expected_chunks = chunk_geometry(source["frames"], source["sample_rate"], clock.get("offset_seconds"))
    if manifest.get("chunks") != expected_chunks:
        raise ValueError("Runtime chunks do not match the pinned ownership geometry")
    expected = [(chunk["start_s"] + index / 25, chunk["support_start_s"], chunk["support_end_s"])
                for chunk in expected_chunks for index in chunk["retained_frame_indices"]]
    if len(expected) != n or not np.array_equal(arrays["frame_times_s"], np.array([row[0] for row in expected], dtype=float)) or not np.array_equal(arrays["support_start_s"], np.array([row[1] for row in expected], dtype=float)) or not np.array_equal(arrays["support_end_s"], np.array([row[2] for row in expected], dtype=float)):
        raise ValueError("Runtime frames do not exactly reconstruct from retained chunk ownership")
    libraries = manifest.get("library_sources")
    snapshots = manifest.get("input_snapshots")
    if not isinstance(libraries, list) or not libraries or not isinstance(snapshots, list) or not snapshots:
        raise ValueError("Runtime manifest lacks library/code input fingerprint coverage")
    extra_paths = [verify_record(row) for row in [*libraries, *snapshots]]
    return manifest, arrays, [manifest_path.resolve(), frame_path.resolve(), *provenance_paths, *extra_paths]


def _runtime_results(arrays: dict[str, np.ndarray], beats: np.ndarray) -> dict:
    from songviz.music_representation import (compare_ordered_recurrence,
                                               compute_local_contrasts,
                                               pool_frame_embeddings)
    pooled = pool_frame_embeddings(arrays["embeddings"], arrays["frame_times_s"], arrays["support_start_s"], arrays["support_end_s"], beats)
    local = compute_local_contrasts(pooled, scales=(2, 4, 8))
    recurrence = compare_ordered_recurrence(pooled, scales=(16, 32), stride_beats=4)
    # ``pooled`` also intentionally carries NumPy work arrays for the numerical
    # functions.  Save only its JSON-safe evidence view.
    return {"pooling": {"method": pooled["method"], "samples": pooled["samples"]}, "local": local, "recurrence": recurrence,
            "method": "cosine distances on pooled pinned MuQ frame embeddings; no calibrated threshold or score rescaling"}


def _fixed_cases(listening: Path, local: dict) -> list[dict]:
    review = json.loads((listening / "review.json").read_text())
    _bound_source(review, label="Listening review")
    cases = []
    for example in review.get("examples", []):
        if not isinstance(example, dict) or not isinstance(example.get("id"), str):
            raise ValueError("Malformed frozen listening example")
        midpoint = (float(example["focus_start_s"]) + float(example["focus_end_s"])) / 2
        times = local["times_s"]
        anchor = min(range(len(times)), key=lambda i: (abs(times[i] - midpoint), i))
        per_scale = []
        for curve in local["curves"]:
            samples = curve["samples"]
            per_scale.append({"scale_beats": curve["scale_beats"], "anchor_samples": [
                samples[anchor + offset] if 0 <= anchor + offset < len(samples) else None
                for offset in (-2, -1, 0, 1, 2)]})
        cases.append({"id": example["id"], "excerpt_bounds_s": {"start_s": example["start_s"], "end_s": example["end_s"]},
                      "focus_bounds_s": {"start_s": example["focus_start_s"], "end_s": example["focus_end_s"]},
                      "fixed_anchor_index": anchor, "fixed_anchor_s": times[anchor],
                      "neighbor_offsets_beats": [-2, -1, 0, 1, 2], "per_scale": per_scale})
    if len(cases) != 4:
        raise ValueError("Expected exactly four frozen listening cases")
    return cases


def baseline_join(structure: Path, feedback: Path) -> dict:
    """Retain explicit positive identity evidence without inventing negatives."""
    reference = json.loads((structure / "reference.json").read_text())
    pairs = []
    for layer in reference.get("layers", []):
        spans = layer.get("spans", []) if isinstance(layer, dict) else []
        for left_index, left in enumerate(spans):
            motif = left.get("motif") if isinstance(left, dict) else None
            if not isinstance(motif, str) or not motif:
                continue
            for right in spans[left_index + 1:]:
                if right.get("motif") == motif and float(left["end_s"]) <= float(right["start_s"]):
                    pairs.append({"layer_id": layer.get("id"), "motif": motif,
                                  "a": {key: left.get(key) for key in ("start_s", "end_s", "label", "motif")},
                                  "b": {key: right.get(key) for key in ("start_s", "end_s", "label", "motif")},
                                  "label": "explicit positive return; adjacent same-motif variation remains eligible",
                                  "negative_label": None})
    return {"structure_baseline": record(structure / "evaluation.json"),
            "recurrence_baseline": record(structure / "recurrence.json"),
            "raw_feedback": json.loads(feedback.read_text()), "positive_returns": pairs,
            "policy": "Only same explicit nonempty motif names are positive returns. Different names and unlabeled times are unknown, never negatives."}


def role_baseline_join(role: Path) -> dict:
    """Keep frozen stem-context measurements beside learned mix distances."""
    data = json.loads((role / "evaluation.json").read_text())
    rows = []
    for case in data.get("cases", []):
        rows.append({"id": case.get("id"), "fixed_anchor_index": case.get("fixed_anchor_index"),
                     "fixed_anchor_s": case.get("fixed_anchor_s"), "per_scale": case.get("per_scale")})
    if len(rows) != 4:
        raise ValueError("Role baseline must preserve four fixed listening cases")
    return {"source": record(role / "evaluation.json"), "cases": rows,
            "comparison_limit": "These are stem-aware RMS/activity/share/spectral descriptor differences, unlike mix-only embedding cosine distance. They are joined for inspection, not subtracted or ranked across units."}


def local_differences(base: dict, shifted: dict) -> dict:
    """Pair complete local curves by declared scale/index; never select cases."""
    second = {row["scale_beats"]: row for row in shifted["curves"]}
    curves = []
    for row in base["curves"]:
        other = second.get(row["scale_beats"])
        if other is None or len(other["samples"]) != len(row["samples"]):
            raise ValueError("Shifted local curves do not align to the base beat clock")
        samples = []
        for a, b in zip(row["samples"], other["samples"]):
            if a is None or b is None or a["anchor_index"] != b["anchor_index"]:
                samples.append(None)
            else:
                av, bv = a["cosine_distance"], b["cosine_distance"]
                samples.append({"anchor_index": a["anchor_index"], "anchor_s": a["anchor_s"],
                                "base_cosine_distance": av, "shifted_cosine_distance": bv,
                                "difference": None if av is None or bv is None else float(bv - av)})
        curves.append({"scale_beats": row["scale_beats"], "samples": samples})
    return {"method": "shifted_minus_base_cosine_distance_v1", "curves": curves}


def beat_pool_differences(base: dict, shifted: dict) -> dict:
    """Full beat-wise base/shift cosine distances on fixed nominal beat bins."""
    a, b = base["samples"], shifted["samples"]
    if len(a) != len(b):
        raise ValueError("Shifted pooled beat count differs from base")
    rows = []
    for left, right in zip(a, b):
        if (left["beat"] != right["beat"] or left["nominal_start_s"] != right["nominal_start_s"]
                or left["nominal_end_s"] != right["nominal_end_s"]):
            raise ValueError("Shifted pooled beat endpoints differ from base")
        va, vb = left["vector"], right["vector"]
        distance = None
        if va is not None and vb is not None:
            xa, xb = np.asarray(va, dtype=float), np.asarray(vb, dtype=float)
            if np.linalg.norm(xa) > 1e-12 and np.linalg.norm(xb) > 1e-12:
                distance = float(1 - np.clip(np.dot(xa, xb) / (np.linalg.norm(xa) * np.linalg.norm(xb)), -1, 1))
        rows.append({"beat": left["beat"], "nominal_start_s": left["nominal_start_s"], "nominal_end_s": left["nominal_end_s"],
                     "base_frame_count": left["frame_count"], "shifted_frame_count": right["frame_count"],
                     "base_encoder_support": {"start_s": left["encoder_support_start_s"], "end_s": left["encoder_support_end_s"]},
                     "shifted_encoder_support": {"start_s": right["encoder_support_start_s"], "end_s": right["encoder_support_end_s"]},
                     "cosine_distance": distance})
    return {"method": "base_vs_shifted_pooled_beat_cosine_distance_v1", "samples": rows,
            "limitations": "Same nominal beat endpoints; different frame centers/counts and encoder contexts are retained, so this is placement sensitivity rather than a pure seam test."}


def plot_svg(local: dict, cases: list[dict], path: Path, shifted: dict | None = None) -> None:
    """Small standalone plot: all local values, with frozen-case anchor marks."""
    width, height, margin = 1000, 440, 62
    times = local["times_s"]; end = max(times[-1], 1.0)
    all_values = [s["cosine_distance"] for obj in [local, shifted] if obj
                  for c in obj["curves"] for s in c["samples"] if s and s["cosine_distance"] is not None]
    ymax = max(.1, np.ceil(max(all_values, default=.1) * 10) / 10)
    colors = ("#1769aa", "#d65f00", "#2e8b57")
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>', '<style>text{font:12px sans-serif}.axis{stroke:#555}.mark{stroke:#999;stroke-dasharray:3 3}</style>',
             f'<line class="axis" x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}"/>',
             f'<text x="{margin}" y="20">Local cosine distance: solid = base, dashed = shifted. Raw units; display axis zoomed.</text>']
    for value in np.linspace(0, ymax, 5):
        y = height - margin - (height - 2 * margin) * value / ymax
        lines += [f'<line x1="{margin}" x2="{width-margin}" y1="{y}" y2="{y}" stroke="#ddd"/>',
                  f'<text x="8" y="{y+4}">{value:.3f}</text>']
    for t in range(30, int(end - 20), 30):
        x = margin + (width - 2 * margin) * t / end
        lines.append(f'<text x="{x}" y="{height-16}">{t}s</text>')
    lines.append('<text x="745" y="42" fill="#666">Vertical lines: fixed listening anchors</text>')
    for case in cases:
        x = margin + (width - 2 * margin) * case["fixed_anchor_s"] / end
        lines.append(f'<line class="mark" x1="{x:.2f}" y1="30" x2="{x:.2f}" y2="{height-margin}"/>')
    def draw(curve: dict, color: str, dash: str = "") -> None:
        segment: list[str] = []
        for sample in curve["samples"] + [None]:
            if sample is None or sample.get("cosine_distance") is None:
                if segment:
                    lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.4" {dash} points="{" ".join(segment)}"/>')
                segment = []
                continue
            x = margin + (width - 2 * margin) * sample["anchor_s"] / end
            y = height - margin - (height - 2 * margin) * sample["cosine_distance"] / ymax
            segment.append(f"{x:.2f},{y:.2f}")
    shifted_by_scale = {row["scale_beats"]: row for row in shifted["curves"]} if shifted else {}
    for index, (curve, color) in enumerate(zip(local["curves"], colors)):
        draw(curve, color)
        if curve["scale_beats"] in shifted_by_scale:
            draw(shifted_by_scale[curve["scale_beats"]], color, 'stroke-dasharray="4 3"')
        lines.append(f'<text x="{margin+index*240}" y="42" fill="{color}">{curve["scale_beats"]} beats per side</text>')
    lines += [f'<text x="{margin}" y="{height-16}">0s</text>', f'<text x="{width-margin-35}" y="{height-16}">{end:.0f}s</text>', '</svg>']
    path.write_text("\n".join(lines) + "\n")


def report_markdown(data: dict) -> str:
    lines = [
        "# MuQ music-representation review", "",
        "This package describes learned embedding distances from the original mix. It does not identify sections, musical roles, vocal function, importance, or calibrated similarity.", "",
        "## Important comparison limit", "",
        "The learned evidence is from the full mix, while the frozen log-CQT/RMS baseline includes separated stems. Any difference can therefore reflect both representation and mix/stem support; it cannot be attributed to MuQ alone.", "",
        "## Evaluation", "",
        "All four frozen listening cases retain fixed anchors plus five neighboring beat anchors at every local scale. Curves and ordered recurrence are retained in JSON; descriptive rank summaries do not select a best scale or threshold.", "",
        "## History", "",
        "Nominal historical comparisons require non-overlapping prior pooled windows. Encoder-independent history additionally requires the prior full encoder support to end before the target encoder support starts. Missing history remains unknown. Both measurements use offline context.", "",
        "## Provenance", "",
        f"Source SHA-256: `{SOURCE_SHA256}`. Runtime, checkpoint/config, baseline manifests, raw feedback and numerical source snapshots are fingerprinted in `manifest.json`.", "",
        "## Fixed four-case sensitivity", "",
        "| Case | Scale (beats) | five-anchor valid distances: min–max |", "| --- | ---: | --- |"]
    for case in data["guided_cases"]:
        for scale in case["per_scale"]:
            values = [sample["cosine_distance"] for sample in scale["anchor_samples"] if sample and sample["cosine_distance"] is not None]
            text = "unavailable" if not values else f"{min(values):.6f}–{max(values):.6f} ({len(values)}/5 valid)"
            lines.append(f"| {case['id']} | {scale['scale_beats']} | {text} |")
    lines += ["", "These are descriptive ranges across every fixed neighbor, not a choice of a best anchor, scale, threshold, or interpretation.", ""]
    evaluation = data.get("evaluation", {})
    def fmt(value):
        return "unknown" if value is None else f"{value:.4f}"
    def median(rows, field):
        values = [r[field] for r in rows if r.get(field) is not None]
        return float(np.median(values)) if values else None
    shifted_curves = {c['scale_beats']: c['samples'] for c in data.get('shifted_pass', {}).get('results', {}).get('local', {}).get('curves', [])}
    lines += ["## Fixed central anchors and baseline context", "",
              "Midrank is a descriptive percentile among all supported anchors at the same scale, not a probability of a meaningful change. RMS changes are amplitude differences; share changes below are percentage points.", "",
              "| Case | Beats/side | MuQ base | Shifted | Base midrank % | Stem | RMS change | Share change (pp) | Activity before / after |",
              "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |"]
    stems = {'drum-entry': 'drums', 'within-passage': 'other', 'verse-ending': 'vocals', 'transition-extent': 'drums'}
    for case in evaluation.get('guided_cases', []):
        for scale in case['scales']:
            row = scale['anchors'][2]
            stem = stems[case['id']]
            base = row['baseline_stems'][stem]
            shifted = shifted_curves.get(scale['scale_beats'])
            sv = shifted[row['anchor_index']]['cosine_distance'] if shifted else None
            share = base['changes']['signed_rms_power_share_difference']
            lines.append(f"| {case['id']} | {scale['scale_beats']} | {fmt(row['muq_cosine_distance'])} | {fmt(sv)} | {fmt(row['muq_whole_curve_midrank_percentile'])} | {stem} | {fmt(base['changes']['signed_rms_difference'])} | {fmt(None if share is None else share*100)} | {fmt(base['left']['active_fraction'])} / {fmt(base['right']['active_fraction'])} |")
    lines += ["", "The displayed stem is fixed by the existing listening question (drums, other, vocals, drums); every stem and all five neighboring anchors remain in evaluation.json.", "",
              "## Explicit positive returns", "",
              "All eligible pairs across separate occurrences of each explicit motif are included. These cosine/pattern measurements have different definitions and are not comparable accuracy scores. Different motifs are not negative labels.", "",
              "| Motif | Beats | Earlier / later occurrence start (s) | Eligible pairs | MuQ median cosine | Shifted median cosine | Baseline median pattern | Baseline median arrangement |",
              "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    for summary in evaluation.get('positive_return_summary', []):
        rows = [r for r in evaluation['positive_return_pairs'] if r['scale_beats']==summary['scale_beats'] and r['layer_id']==summary['layer_id'] and r['motif']==summary['motif'] and r['a_occurrence']['start_s']==summary['a_run_start_s'] and r['b_occurrence']['start_s']==summary['b_run_start_s']]
        lines.append(f"| {summary['motif']} | {summary['scale_beats']} | {summary['a_run_start_s']:.2f} / {summary['b_run_start_s']:.2f} | {summary['eligible_pair_count']} | {fmt(median(rows,'muq_cosine_similarity'))} | {fmt(median(rows,'shifted_muq_cosine_similarity'))} | {fmt(median(rows,'baseline_pattern_similarity'))} | {fmt(median(rows,'baseline_arrangement_similarity'))} |")
    if data.get('shifted_pass'):
        lines += ["", "## Placement sensitivity", "",
                  "The shift changes both encoder context and frame-grid phase (20 ms). This is not a pure seam test. First boundary chunks overlap deliberately to preserve source coverage.", "",
                  "| Beats/side | Paired anchors | Median absolute change | Maximum absolute change | Base/shift Pearson correlation |",
                  "| ---: | ---: | ---: | ---: | ---: |"]
        for curve in data['shifted_pass']['local_differences']['curves']:
            rows = [r for r in curve['samples'] if r and r['difference'] is not None]
            diffs = np.abs([r['difference'] for r in rows])
            a, b = [r['base_cosine_distance'] for r in rows], [r['shifted_cosine_distance'] for r in rows]
            corr = float(np.corrcoef(a,b)[0,1]) if len(rows)>1 and np.std(a)>0 and np.std(b)>0 else None
            lines.append(f"| {curve['scale_beats']} | {len(rows)} | {fmt(float(np.median(diffs)) if len(rows) else None)} | {fmt(float(np.max(diffs)) if len(rows) else None)} | {fmt(corr)} |")
        beat_rows = data['shifted_pass']['beat_pool_differences']['samples']
        lines += ["", f"Base/shift pooled-beat cosine distance: median {fmt(median(beat_rows,'cosine_distance'))}; all {len(beat_rows)} beat comparisons are saved.", ""]
    lines += ["Full per-anchor MuQ/stem baseline joins and all eligible return rows are in `evaluation.json`. Nominal historical recurrence permits nonoverlapping nominal spans; the separately reported strict encoder-context history further requires earlier complete encoder support before target support, so it may be unavailable more often.", ""]
    return "\n".join(lines)


def build(*, runtime: Path, out: Path, structure: Path = ROOT / "outputs/reviews/structure-evaluation-03",
          role: Path = ROOT / "outputs/reviews/role-context-02", listening: Path = ROOT / "outputs/reviews/listening-examples-01",
          shifted_runtime: Path | None = None) -> None:
    runtime, out, structure, role, listening = (p.resolve() for p in (runtime, out, structure, role, listening))
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    structure_manifest, structure_inputs = verified_package(structure, BASELINE_KINDS["structure"])
    role_manifest, role_inputs = verified_package(role, BASELINE_KINDS["role"])
    listening_manifest, listening_inputs = verified_package(listening, BASELINE_KINDS["listening"])
    require_declared(structure_manifest, structure / "features.npz")
    require_declared(structure_manifest, structure / "reference.json")
    require_declared(listening_manifest, listening / "review.json")
    require_declared(role_manifest, role / "role-context.json")
    require_declared(role_manifest, role / "evaluation.json")
    require_declared(structure_manifest, structure / "recurrence.json")
    require_declared(structure_manifest, structure / "evaluation.json")
    _bound_source(json.loads((structure / "reference.json").read_text()), label="Structure baseline")
    beats = load_beats(structure)
    runtime_manifest, arrays, runtime_inputs = load_runtime(runtime)
    if runtime_manifest["clock"].get("offset_seconds") != 0:
        raise ValueError("Base runtime must use zero chunk-placement offset")
    if arrays["support_start_s"].min() > beats[0] or arrays["support_end_s"].max() < beats[-1]:
        raise ValueError("Runtime frame supports do not cover the complete baseline beat clock")
    result = _runtime_results(arrays, beats)
    missing_beats = [sample["beat"] for sample in result["pooling"]["samples"] if sample["vector"] is None]
    if missing_beats:
        raise ValueError(f"Runtime frame centers leave {len(missing_beats)} unsupported baseline beat bins")
    feedback = ROOT / "benchmark/feedback/listening-examples-01.json"
    if not feedback.is_file() or sha256_file(feedback) != FEEDBACK_SHA256:
        raise ValueError("Frozen listening raw feedback is missing or changed")
    data = {"schema_version": 1, "kind": "songviz-music-representation-review", "source_audio_sha256": SOURCE_SHA256,
            "runtime": runtime_manifest, "beat_times_s": beats.tolist(), "results": result,
            "guided_cases": _fixed_cases(listening, result["local"]),
            "comparison_limit": "Mix-only MuQ evidence versus stem-aware log-CQT/RMS baseline is representation-plus-support confounded.",
            "raw_feedback": record(feedback)}
    data["baseline_join"] = baseline_join(structure, feedback)
    data["role_baseline_join"] = role_baseline_join(role)
    from songviz.music_representation import evaluate_representation
    baseline_recurrence = json.loads((structure / "recurrence.json").read_text())
    role_context = json.loads((role / "role-context.json").read_text())
    role_evaluation = json.loads((role / "evaluation.json").read_text())
    reference = json.loads((structure / "reference.json").read_text())
    if role_context['times_s'] != beats.tolist() or runtime_manifest['source']['duration_s'] != reference['source']['duration_s']:
        raise ValueError("Source duration or role baseline clock disagrees with the structure baseline")
    for base_scale, muq_scale in zip(baseline_recurrence['results'], result['recurrence']['results']):
        if base_scale['scale_beats'] != muq_scale['scale_beats'] or len(base_scale['spans']) != len(muq_scale['spans']):
            raise ValueError("Baseline recurrence scales/coverage differ")
        for a,b in zip(base_scale['spans'], muq_scale['spans']):
            if any(a[k] != b[k] for k in ('start_beat','end_beat','start_s','end_s')):
                raise ValueError("Baseline recurrence spans differ from model spans")
    raw_feedback = json.loads(feedback.read_text())
    if shifted_runtime is not None:
        shifted_manifest, shifted_arrays, shifted_inputs = load_runtime(shifted_runtime.resolve())
        if shifted_manifest.get("checkpoint") != runtime_manifest.get("checkpoint") or shifted_manifest.get("config") != runtime_manifest.get("config"):
            raise ValueError("Shifted runtime must use the exact same checkpoint and config")
        if shifted_manifest["clock"].get("offset_seconds") != 2.5:
            raise ValueError("Shifted runtime must use 2.5-second chunk-placement offset")
        if shifted_arrays["support_start_s"].min() > beats[0] or shifted_arrays["support_end_s"].max() < beats[-1]:
            raise ValueError("Shifted runtime frame supports do not cover the complete baseline beat clock")
        shifted = _runtime_results(shifted_arrays, beats)
        if any(sample["vector"] is None for sample in shifted["pooling"]["samples"]):
            raise ValueError("Shifted runtime frame centers leave unsupported baseline beat bins")
        data["shifted_pass"] = {"runtime": shifted_manifest, "results": shifted,
                                "beat_pool_differences": beat_pool_differences(result["pooling"], shifted["pooling"]),
                                "local_differences": local_differences(result["local"], shifted["local"]),
                                "policy": "Full unselected local/recurrence outputs are retained; differences are descriptive and not cherry-picked. This probes chunk/context plus frame-grid placement sensitivity, not a pure encoder-seam effect; beat endpoints are fixed while frame counts can differ."}
        runtime_inputs.extend(shifted_inputs)
        data["evaluation"] = evaluate_representation(result, baseline_recurrence, role_context, role_evaluation, reference, raw_feedback, shifted_result=shifted)
    else:
        data["evaluation"] = evaluate_representation(result, baseline_recurrence, role_context, role_evaluation, reference, raw_feedback)
    dependencies = {p: record(p) for p in [*structure_inputs, *role_inputs, *listening_inputs, *runtime_inputs, feedback, ROOT / "experiments/build_music_representation_review.py", ROOT / "experiments/probe_music_representation.py", ROOT / "songviz/music_representation.py"]}
    if any(not p.is_file() for p in dependencies):
        raise FileNotFoundError("Builder or numeric source missing")
    # Keep the package replayable without duplicating original audio, stems, or
    # checkpoint weights.  Those remain verified external dependencies.
    # The saved frame artifact is the replay input and is copied even if large;
    # model weights/audio/stems are not.  Other large cache-like NPZ files stay
    # external, fingerprinted baseline dependencies.
    snapshots = {p: fp for p, fp in dependencies.items()
                 if (p.suffix in {".json", ".py"} and p.stat().st_size < 10_000_000)
                 or (p.suffix == ".npz" and (p.stat().st_size < 10_000_000 or p in runtime_inputs))}
    out.mkdir(parents=True)
    for path in snapshots:
        target = out / "inputs" / path.relative_to(ROOT) if ROOT in path.parents else out / "inputs" / "external" / path.name
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
    write_json(out / "review.json", data)
    write_json(out / "local.json", result["local"])
    write_json(out / "recurrence.json", result["recurrence"])
    write_json(out / "baseline-joins.json", data["baseline_join"])
    write_json(out / "role-baseline-joins.json", data["role_baseline_join"])
    write_json(out / "evaluation.json", data["evaluation"])
    if "shifted_pass" in data:
        write_json(out / "shifted-local-differences.json", data["shifted_pass"]["local_differences"])
        write_json(out / "shifted-beat-pool-differences.json", data["shifted_pass"]["beat_pool_differences"])
    plot_svg(result["local"], data["guided_cases"], out / "local-curves.svg",
             data.get("shifted_pass", {}).get("results", {}).get("local"))
    (out / "report.md").write_text(report_markdown(data))
    # Recheck every consumed dependency immediately before finalizing.
    for path, fp in dependencies.items():
        if sha256_file(path) != fp["sha256"]:
            raise ValueError(f"Input changed during build: {path}")
    manifest = {"schema_version": 1, "kind": data["kind"], "created_utc": datetime.now(timezone.utc).isoformat(),
                "source_audio_sha256": SOURCE_SHA256, "dependencies": list(dependencies.values()),
                "input_snapshots": [record(p) for p in sorted((out / "inputs").rglob("*")) if p.is_file()],
                "outputs": [record(p) for p in sorted(out.iterdir()) if p.is_file()],
                "python_version": platform.python_version(), "replay": "Run this builder with the saved runtime directory into a new output directory; compare local.json and recurrence.json numerically.",
                "limitations": data["comparison_limit"]}
    write_json(out / "manifest.json", manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--structure", type=Path, default=ROOT / "outputs/reviews/structure-evaluation-03")
    parser.add_argument("--role", type=Path, default=ROOT / "outputs/reviews/role-context-02")
    parser.add_argument("--listening", type=Path, default=ROOT / "outputs/reviews/listening-examples-01")
    parser.add_argument("--shifted-runtime", type=Path)
    build(**vars(parser.parse_args()))
