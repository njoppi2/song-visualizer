"""Create the no-run, provenance-bound semantic-probe preregistration package.

By default this builder only binds *declared existing files* by SHA-256.  Its
explicit ``--write-clips`` mode additionally creates the frozen local cuts; it
never uploads audio or runs a model.  A package is immutable by convention: an
existing output directory is always refused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
DESIGN_PATH = ROOT / "docs" / "27_semantic_experiment_design.md"
ORDER_SEED = 20260914
PACKAGE_KIND = "songviz-semantic-preregistration"
SCHEMA_VERSION = 1

STAGE_A = (
    "Listen to this audio clip. Describe what you hear and how it develops over\n"
    "time. Give clip-relative times in seconds for anything that starts, stops or\n"
    "changes. If you are unsure, say so instead of guessing."
)
STAGE_B = (
    "1. Is there a moment where the arrangement changes (parts entering, leaving,\n"
    "   becoming fuller or thinner)? If yes: when, and which parts. If no, say \"no change\".\n"
    "2. Do any voices appear? For each: how is it produced (for example singing,\n"
    "   rapping, speaking, or another vocal sound), and does that manner change? When?\n"
    "3. Does any part move between leading and supporting roles? When, and which?\n"
    "4. Does the clip feel like one continuing musical part, or a move from one\n"
    "   part to another? If a move, when?\n"
    "Answer \"unsure\" where appropriate."
)

# This is intentionally a closed list: changing an id, interval, source recipe,
# or duration is a protocol deviation rather than an implicit new experiment.
REQUIRED_CLIPS = (
    {"id": "drum-entry.core", "source_kind": "original_mix", "start_s": 74, "end_s": 85, "duration_s": 11},
    {"id": "within-passage.core", "source_kind": "original_mix", "start_s": 16, "end_s": 26, "duration_s": 10},
    {"id": "verse-ending.core", "source_kind": "original_mix", "start_s": 119, "end_s": 132, "duration_s": 13},
    {"id": "transition-extent.core", "source_kind": "original_mix", "start_s": 57, "end_s": 70, "duration_s": 13},
    {"id": "verse-ending.pre", "source_kind": "original_mix", "start_s": 115, "end_s": 132, "duration_s": 17},
    {"id": "verse-ending.post", "source_kind": "original_mix", "start_s": 119, "end_s": 136, "duration_s": 17},
    {"id": "verse-ending.resum", "source_kind": "stem_sum", "stem_names": ["bass", "drums", "other", "vocals"], "start_s": 119, "end_s": 132, "duration_s": 13},
    {"id": "verse-ending.novocals", "source_kind": "stem_sum", "stem_names": ["bass", "drums", "other"], "start_s": 119, "end_s": 132, "duration_s": 13},
)
EXPECTED_IDS = frozenset(row["id"] for row in REQUIRED_CLIPS)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"Input is not a regular file: {resolved}")
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha256_file(resolved)}


def validate_protocol(clips: tuple[dict[str, Any], ...] = REQUIRED_CLIPS) -> None:
    if len(clips) != 8 or set(row["id"] for row in clips) != EXPECTED_IDS:
        raise ValueError("Protocol must define exactly the eight required R clip ids")
    for clip in clips:
        start, end, duration = clip["start_s"], clip["end_s"], clip["duration_s"]
        if not isinstance(clip["id"], str) or not clip["id"] or end - start != duration or duration <= 0:
            raise ValueError(f"Invalid declared interval for {clip.get('id')!r}")
        if clip["source_kind"] == "stem_sum" and not clip.get("stem_names"):
            raise ValueError(f"Stem recipe missing for {clip['id']}")
        if clip["source_kind"] == "external_control" and clip.get("replacement_source_start_s") != start:
            raise ValueError(f"External-control source alignment missing for {clip['id']}")


def effective_clips(*, novocals_replacement: Path | None) -> tuple[dict[str, Any], ...]:
    """Return the frozen R set, optionally substituting only the failed ablation.

    The replacement is a 13-second, already-aligned candidate instrumental
    excerpt.  Its original-song alignment remains 119–132s while its file-local
    read range is 0–13s; both facts are made explicit in the manifest.
    """
    if novocals_replacement is None:
        return REQUIRED_CLIPS
    rows: list[dict[str, Any]] = []
    for row in REQUIRED_CLIPS:
        if row["id"] != "verse-ending.novocals":
            rows.append(row)
            continue
        rows.append({
            "id": row["id"], "source_kind": "external_control",
            "control_name": "kim-mel-band-roformer-other",
            "start_s": row["start_s"], "end_s": row["end_s"], "duration_s": row["duration_s"],
            "replacement_source_start_s": row["start_s"], "replacement_file_start_s": 0,
            "replacement_input": str(novocals_replacement.resolve(strict=True)),
        })
    result = tuple(rows)
    validate_protocol(result)
    return result


def ordered_clips(clips: tuple[dict[str, Any], ...] = REQUIRED_CLIPS) -> list[dict[str, Any]]:
    """Return the frozen ASCII-id random permutation, numbered for future files."""
    validate_protocol(clips)
    ids = sorted(EXPECTED_IDS)
    order = random.Random(ORDER_SEED).sample(ids, len(ids))
    by_id = {row["id"]: row for row in clips}
    return [{"file_name": f"clip_{index:02d}.wav", "clip_id": clip_id, **by_id[clip_id]}
            for index, clip_id in enumerate(order, start=1)]


def scoring_template(clips: tuple[dict[str, Any], ...] = REQUIRED_CLIPS) -> dict[str, Any]:
    """Structured blank records; this is not an automatic scorer."""
    claim_fields = [
        "clip", "stage", "question", "span", "proposition", "polarity", "claim_type",
        "clip_duration_s", "clip_start_s", "clip_end_s", "time_status", "source_start_s",
        "source_end_s", "hedged", "target_id", "grade", "is_contradiction", "proxy_conflict",
    ]
    return {
        "kind": "semantic-probe-scoring-template",
        "schema_version": 1,
        "instruction": "Blank scorer records only. Populate independently before seeing the id mapping, as doc 27 §4.3 requires.",
        "claim_record_fields": claim_fields,
        "allowed_polarity": ["affirmed", "negated", "uncertain"],
        "allowed_time_status": ["valid", "invalid", "untimed"],
        "t5_grades": ["full", "full-A", "partial", "unscoreable"],
        "two_scorers_required": True,
        "stage_aggregation": "Stage A and Stage B form one answer after blind segmentation and mapping.",
        "records": [{"clip_file": row["file_name"], "stage_a_claims": [], "stage_b_claims": []} for row in ordered_clips(clips)],
    }


def package_schema(*, clips_written: bool) -> dict[str, Any]:
    return {
        "kind": PACKAGE_KIND,
        "schema_version": SCHEMA_VERSION,
        "required_files": ["manifest.json", "schema.json", "order_mapping.json", "prompts.json", "scoring_template.json", "manipulation_check_template.json"],
        "clip_mode": "eight local PCM_24 WAV clips written" if clips_written else "no clips written",
        "prohibitions": [
            "This builder never uploads audio or runs a model.",
            "Do not overwrite this package; create a new output directory for a new experiment.",
        ],
        "later_execution_requirements": [
            "Verify every source fingerprint before any later audio/model action.",
            "Use the recorded order mapping and fresh model context per clip.",
            "Record raw/retry replies and both independent scorer records without modifying this package.",
        ],
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _audio_info(path: Path, *, label: str, required_frames: int) -> sf.SoundFile:
    """Validate format/support before creating any output package directory."""
    info = sf.info(path)
    if info.samplerate != 44_100 or info.channels != 2:
        raise ValueError(f"{label} must be 44.1kHz stereo (got {info.samplerate}Hz/{info.channels}ch)")
    if info.frames < required_frames:
        raise ValueError(f"{label} lacks required frame support ({info.frames} < {required_frames})")
    return info


def _read_frames(path: Path, *, start_frame: int, frames: int) -> np.ndarray:
    with sf.SoundFile(path, "r") as audio:
        audio.seek(start_frame)
        samples = audio.read(frames, dtype="float64", always_2d=True)
    if samples.shape != (frames, 2):
        raise ValueError(f"Could not read exact requested frame range from {path}")
    return samples


def _write_clips(output_dir: Path, *, original_mix: Path, stems: dict[str, Path], clips: tuple[dict[str, Any], ...], novocals_replacement: Path | None) -> list[dict[str, Any]]:
    """Write only the frozen R clips, using integer 44.1kHz source-frame bounds."""
    clips_dir = output_dir / "clips"
    clips_dir.mkdir()
    records = []
    for row in ordered_clips(clips):
        start_frame = row["start_s"] * 44_100
        frames = row["duration_s"] * 44_100
        if row["source_kind"] == "original_mix":
            samples = _read_frames(original_mix, start_frame=start_frame, frames=frames)
        elif row["source_kind"] == "stem_sum":
            parts = [_read_frames(stems[name], start_frame=start_frame, frames=frames) for name in row["stem_names"]]
            samples = np.add.reduce(parts)
            peak = float(np.max(np.abs(samples))) if samples.size else 0.0
            if not np.isfinite(samples).all() or peak > 1.0:
                raise ValueError(f"Stem sum for {row['clip_id']} would clip (peak {peak!r})")
        elif row["source_kind"] == "external_control":
            if novocals_replacement is None:
                raise ValueError("External-control row requires a replacement input")
            samples = _read_frames(novocals_replacement, start_frame=0, frames=frames)
        else:
            raise ValueError(f"Unsupported clip source kind: {row['source_kind']}")
        path = clips_dir / row["file_name"]
        # PCM_24 is deliberate: all clips use the same uncompressed, portable
        # representation, while frame boundaries remain the exact integer bounds.
        sf.write(path, samples, 44_100, subtype="PCM_24", format="WAV")
        info = sf.info(path)
        if info.samplerate != 44_100 or info.channels != 2 or info.frames != frames or info.subtype != "PCM_24":
            raise ValueError(f"Generated clip format mismatch for {row['clip_id']}")
        records.append({
            "clip_id": row["clip_id"], "file_name": row["file_name"], "path": f"clips/{row['file_name']}",
            "start_frame": start_frame, "frames": frames, "sample_rate": 44_100, "channels": 2,
            "subtype": "PCM_24", "sha256": sha256_file(path), "bytes": path.stat().st_size,
        })
    return records


def build(output_dir: Path, *, original_mix: Path, bass: Path, drums: Path, other: Path, vocals: Path, write_clips: bool = False, novocals_replacement: Path | None = None, replacement_control_record: Path | None = None, design_document: Path | None = None) -> Path:
    """Write a fresh preregistration package after fingerprinting all declared inputs."""
    clips = effective_clips(novocals_replacement=novocals_replacement)
    validate_protocol(clips)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing package: {output_dir}")
    inputs = {
        "original_mix": fingerprint(original_mix),
        "stems": {"bass": fingerprint(bass), "drums": fingerprint(drums), "other": fingerprint(other), "vocals": fingerprint(vocals)},
    }
    if (novocals_replacement is None) != (replacement_control_record is None):
        raise ValueError("Replacement audio and its completed control record must be supplied together")
    if novocals_replacement is not None and replacement_control_record is not None:
        inputs["novocals_replacement"] = fingerprint(novocals_replacement)
        inputs["replacement_control_record"] = fingerprint(replacement_control_record)
    design = fingerprint(design_document or DESIGN_PATH)
    code = fingerprint(Path(__file__))
    stems = {"bass": bass.resolve(), "drums": drums.resolve(), "other": other.resolve(), "vocals": vocals.resolve()}
    if write_clips:
        required_frames = max(row["end_s"] for row in clips) * 44_100
        _audio_info(original_mix, label="Original mix", required_frames=required_frames)
        for name, path in stems.items():
            _audio_info(path, label=f"Stem {name}", required_frames=required_frames)
        if novocals_replacement is not None:
            replacement_duration = next(row["duration_s"] for row in clips if row["id"] == "verse-ending.novocals")
            replacement_frames = replacement_duration * 44_100
            info = _audio_info(novocals_replacement, label="Replacement no-vocals control", required_frames=replacement_frames)
            if info.frames != replacement_frames:
                raise ValueError(f"Replacement no-vocals control must be exactly {replacement_duration} seconds")
    output_dir.mkdir(parents=False)
    order = ordered_clips(clips)
    clip_records = _write_clips(output_dir, original_mix=original_mix, stems=stems, clips=clips, novocals_replacement=novocals_replacement) if write_clips else []
    write_json(output_dir / "schema.json", package_schema(clips_written=write_clips))
    write_json(output_dir / "order_mapping.json", {"order_seed": ORDER_SEED, "algorithm": "random.Random(seed).sample(sorted(ids), len(ids))", "clips": order})
    write_json(output_dir / "prompts.json", {
        "stage_a": STAGE_A, "stage_b": STAGE_B,
        "decoding": {"do_sample": False, "max_new_tokens_per_stage": 400, "framework_seed": 0},
        "retry": {"maximum_per_clip": 1, "fresh_context": True, "stage_a_append": "Write each observation as `[start–end s] description` or `[time s] description`."},
    })
    write_json(output_dir / "scoring_template.json", scoring_template(clips))
    write_json(output_dir / "manipulation_check_template.json", {
        "instruction": "Before any model output is read, bind the matched resum-positive and no-vocals control records. Valid only if resum=yes and novocals=no.",
        "replacement_control_record": (fingerprint(replacement_control_record) if replacement_control_record else None),
        "responses": ({"verse-ending.resum": "See completed replacement control record.", "verse-ending.novocals": "See completed replacement control record."}
                      if replacement_control_record else {"verse-ending.resum": None, "verse-ending.novocals": None}),
    })
    manifest = {
        "kind": PACKAGE_KIND, "schema_version": SCHEMA_VERSION,
        "protocol": {"design_document": design, "builder": code, "order_seed": ORDER_SEED, "required_r_clip_count": 8},
        "declared_inputs": inputs,
        "clips": list(clips),
        "generated_clips": clip_records,
        "package_rule": "G0 is not complete until this package, clip hashes, order mapping, and manipulation-check order are recorded before model output.",
        "audio_action": ("Generated exactly eight local sample-exact frame cuts; no audio was uploaded or run by a model."
                         if write_clips else "No clips were cut, decoded, mixed, copied, uploaded, or run by this builder."),
    }
    write_json(output_dir / "manifest.json", manifest)
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--original-mix", type=Path, required=True)
    parser.add_argument("--bass-stem", type=Path, required=True)
    parser.add_argument("--drums-stem", type=Path, required=True)
    parser.add_argument("--other-stem", type=Path, required=True)
    parser.add_argument("--vocals-stem", type=Path, required=True)
    parser.add_argument("--write-clips", action="store_true", help="Explicitly write the eight local PCM_24 WAV clips after source validation.")
    parser.add_argument("--novocals-replacement", type=Path, help="Exact 13-second replacement instrumental control for verse-ending.novocals.")
    parser.add_argument("--replacement-control-record", type=Path, help="Completed blind listener record bound to --novocals-replacement.")
    parser.add_argument("--design-document", type=Path, default=DESIGN_PATH, help="Frozen protocol document or explicit amendment to fingerprint.")
    args = parser.parse_args(argv)
    build(args.output_dir, original_mix=args.original_mix, bass=args.bass_stem, drums=args.drums_stem, other=args.other_stem, vocals=args.vocals_stem, write_clips=args.write_clips, novocals_replacement=args.novocals_replacement, replacement_control_record=args.replacement_control_record, design_document=args.design_document)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
