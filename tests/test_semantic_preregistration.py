from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from experiments import build_semantic_preregistration as builder


def _inputs(tmp_path: Path) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in ("mix", "bass", "drums", "other", "vocals"):
        path = tmp_path / f"{name}.wav"
        path.write_bytes(f"not-audio-{name}".encode())
        paths[name] = path
    return paths


def _build(tmp_path: Path) -> Path:
    inputs = _inputs(tmp_path)
    design = tmp_path / "design.md"
    design.write_text("frozen design")
    original = builder.DESIGN_PATH
    builder.DESIGN_PATH = design
    try:
        return builder.build(tmp_path / "package", original_mix=inputs["mix"], bass=inputs["bass"], drums=inputs["drums"], other=inputs["other"], vocals=inputs["vocals"])
    finally:
        builder.DESIGN_PATH = original


def test_builds_exact_eight_clip_no_audio_preregistration(tmp_path: Path) -> None:
    package = _build(tmp_path)
    names = {path.name for path in package.iterdir()}
    assert names == {"manifest.json", "schema.json", "order_mapping.json", "prompts.json", "scoring_template.json", "manipulation_check_template.json"}
    manifest = json.loads((package / "manifest.json").read_text())
    assert manifest["kind"] == builder.PACKAGE_KIND
    assert manifest["audio_action"].startswith("No clips were cut")
    assert {row["id"] for row in manifest["clips"]} == builder.EXPECTED_IDS
    assert [row["duration_s"] for row in manifest["clips"]] == [11, 10, 13, 13, 17, 17, 13, 13]
    assert manifest["declared_inputs"]["original_mix"]["bytes"] > 0
    assert set(manifest["declared_inputs"]["stems"]) == {"bass", "drums", "other", "vocals"}


def test_order_is_seeded_ascii_permutation_and_file_mapping(tmp_path: Path) -> None:
    package = _build(tmp_path)
    order = json.loads((package / "order_mapping.json").read_text())
    assert order["order_seed"] == 20260914
    assert [row["file_name"] for row in order["clips"]] == [f"clip_{n:02d}.wav" for n in range(1, 9)]
    assert [row["clip_id"] for row in order["clips"]] == [
        "verse-ending.resum", "verse-ending.pre", "verse-ending.post", "verse-ending.novocals",
        "verse-ending.core", "within-passage.core", "drum-entry.core", "transition-extent.core",
    ]
    assert {row["clip_id"] for row in order["clips"]} == builder.EXPECTED_IDS


def test_prompts_and_scoring_template_are_frozen(tmp_path: Path) -> None:
    package = _build(tmp_path)
    prompts = json.loads((package / "prompts.json").read_text())
    assert prompts["decoding"] == {"do_sample": False, "framework_seed": 0, "max_new_tokens_per_stage": 400}
    assert "laughter" not in prompts["stage_b"].lower()
    template = json.loads((package / "scoring_template.json").read_text())
    assert template["two_scorers_required"] is True
    assert set(template["allowed_polarity"]) == {"affirmed", "negated", "uncertain"}
    assert len(template["records"]) == 8
    assert "proxy_conflict" in template["claim_record_fields"]


def test_refuses_existing_output_and_missing_inputs(tmp_path: Path) -> None:
    package = _build(tmp_path)
    inputs = _inputs(tmp_path / "second")
    with pytest.raises(FileExistsError, match="overwrite"):
        builder.build(package, original_mix=inputs["mix"], bass=inputs["bass"], drums=inputs["drums"], other=inputs["other"], vocals=inputs["vocals"])
    with pytest.raises(FileNotFoundError):
        builder.build(tmp_path / "missing", original_mix=tmp_path / "missing.wav", bass=inputs["bass"], drums=inputs["drums"], other=inputs["other"], vocals=inputs["vocals"])


def _tiny_audio_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, Path], tuple[dict, ...]]:
    """Use a short protocol preserving ids/recipes, never real song audio."""
    protocol = tuple({**row, "start_s": index, "end_s": index + 1, "duration_s": 1}
                     for index, row in enumerate(builder.REQUIRED_CLIPS))
    monkeypatch.setattr(builder, "REQUIRED_CLIPS", protocol)
    monkeypatch.setattr(builder, "EXPECTED_IDS", frozenset(row["id"] for row in protocol))
    frames = 8 * 44_100
    ramp = np.arange(frames, dtype=np.float64)[:, None] / (frames * 8)
    mix = np.hstack((ramp, -ramp))
    values = {"bass": 0.01, "drums": 0.02, "other": 0.03, "vocals": 0.04}
    paths: dict[str, Path] = {}
    mix_path = tmp_path / "mix.wav"
    sf.write(mix_path, mix, 44_100, subtype="PCM_24")
    paths["mix"] = mix_path
    for name, value in values.items():
        path = tmp_path / f"{name}.wav"
        sf.write(path, np.full((frames, 2), value, dtype=np.float64), 44_100, subtype="PCM_24")
        paths[name] = path
    return paths, protocol


def test_write_clips_creates_eight_hashed_sample_exact_pcm24_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths, protocol = _tiny_audio_inputs(tmp_path, monkeypatch)
    design = tmp_path / "design.md"
    design.write_text("frozen design")
    monkeypatch.setattr(builder, "DESIGN_PATH", design)
    package = builder.build(tmp_path / "package", original_mix=paths["mix"], bass=paths["bass"], drums=paths["drums"], other=paths["other"], vocals=paths["vocals"], write_clips=True)
    manifest = json.loads((package / "manifest.json").read_text())
    records = manifest["generated_clips"]
    assert len(records) == 8
    assert manifest["audio_action"].startswith("Generated exactly eight")
    assert {row["clip_id"] for row in records} == {row["id"] for row in protocol}
    for row in records:
        path = package / row["path"]
        info = sf.info(path)
        assert (info.samplerate, info.channels, info.frames, info.subtype) == (44_100, 2, 44_100, "PCM_24")
        assert row["sha256"] == builder.sha256_file(path)
    by_id = {row["clip_id"]: row for row in records}
    original_samples, _ = sf.read(package / by_id["drum-entry.core"]["path"], dtype="float64", always_2d=True)
    expected_mix, _ = sf.read(paths["mix"], start=0, frames=44_100, dtype="float64", always_2d=True)
    assert np.array_equal(original_samples, expected_mix)
    resumed, _ = sf.read(package / by_id["verse-ending.resum"]["path"], dtype="float64", always_2d=True)
    assert np.allclose(resumed, 0.10, atol=2e-7)
    novocals, _ = sf.read(package / by_id["verse-ending.novocals"]["path"], dtype="float64", always_2d=True)
    assert np.allclose(novocals, 0.06, atol=2e-7)


def test_write_clips_rejects_mismatched_stem_support_and_clipping(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths, _ = _tiny_audio_inputs(tmp_path, monkeypatch)
    design = tmp_path / "design.md"
    design.write_text("frozen design")
    monkeypatch.setattr(builder, "DESIGN_PATH", design)
    sf.write(paths["vocals"], np.zeros((8 * 44_100, 2)), 48_000, subtype="PCM_24")
    with pytest.raises(ValueError, match="44.1kHz stereo"):
        builder.build(tmp_path / "wrong-rate", original_mix=paths["mix"], bass=paths["bass"], drums=paths["drums"], other=paths["other"], vocals=paths["vocals"], write_clips=True)
    sf.write(paths["vocals"], np.zeros((44_100, 2)), 44_100, subtype="PCM_24")
    with pytest.raises(ValueError, match="lacks required frame support"):
        builder.build(tmp_path / "short", original_mix=paths["mix"], bass=paths["bass"], drums=paths["drums"], other=paths["other"], vocals=paths["vocals"], write_clips=True)
    # Restore support and force the frozen four-stem recipe beyond unity.
    sf.write(paths["vocals"], np.full((8 * 44_100, 2), 0.99), 44_100, subtype="PCM_24")
    with pytest.raises(ValueError, match="would clip"):
        builder.build(tmp_path / "clip", original_mix=paths["mix"], bass=paths["bass"], drums=paths["drums"], other=paths["other"], vocals=paths["vocals"], write_clips=True)


def test_replacement_control_writes_only_the_novocals_clip_from_exact_candidate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths, protocol = _tiny_audio_inputs(tmp_path, monkeypatch)
    design = tmp_path / "design.md"
    design.write_text("frozen design")
    monkeypatch.setattr(builder, "DESIGN_PATH", design)
    candidate = tmp_path / "candidate.wav"
    sf.write(candidate, np.full((44_100, 2), 0.07), 44_100, subtype="PCM_24")
    record = tmp_path / "passed-control.json"
    record.write_text('{"outcome":{"valid":true}}\n')
    package = builder.build(
        tmp_path / "package", original_mix=paths["mix"], bass=paths["bass"], drums=paths["drums"],
        other=paths["other"], vocals=paths["vocals"], write_clips=True,
        novocals_replacement=candidate, replacement_control_record=record,
    )
    manifest = json.loads((package / "manifest.json").read_text())
    novocals = next(row for row in manifest["clips"] if row["id"] == "verse-ending.novocals")
    assert novocals["source_kind"] == "external_control"
    assert novocals["replacement_source_start_s"] == novocals["start_s"]
    assert manifest["declared_inputs"]["novocals_replacement"]["sha256"] == builder.sha256_file(candidate)
    generated = {row["clip_id"]: row for row in manifest["generated_clips"]}
    replacement_samples, _ = sf.read(package / generated["verse-ending.novocals"]["path"], dtype="float64", always_2d=True)
    assert np.allclose(replacement_samples, 0.07, atol=2e-7)
    resum_samples, _ = sf.read(package / generated["verse-ending.resum"]["path"], dtype="float64", always_2d=True)
    assert np.allclose(resum_samples, 0.10, atol=2e-7)


def test_replacement_audio_and_record_are_an_atomic_pair(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    candidate = tmp_path / "candidate.wav"
    candidate.write_bytes(b"candidate")
    with pytest.raises(ValueError, match="supplied together"):
        builder.build(tmp_path / "package", original_mix=inputs["mix"], bass=inputs["bass"], drums=inputs["drums"], other=inputs["other"], vocals=inputs["vocals"], novocals_replacement=candidate)
