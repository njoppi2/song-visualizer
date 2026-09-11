import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import experiments.build_directed_review as builder
from experiments.build_review import fingerprint
from songviz.direction import make_plan


def test_stem_energy_uses_stereo_power_for_antiphase_signal(tmp_path: Path) -> None:
    """Opposite stereo channels must add as energy, not cancel to silence."""
    wav = tmp_path / "antiphase.wav"
    mono = np.tile(np.array([1.0, -1.0], dtype=np.float32), 10)
    sf.write(wav, np.column_stack((mono, -mono)), 100)

    times, values, metadata = builder.stem_energy(wav)

    assert times == [0.0, 0.1, 0.2]
    assert all(0.0 <= value <= 1.0 for value in values)
    assert max(values) > 0.0
    assert metadata["timestamps"] == "completed-block end; initial zero"
    assert metadata["hop_s"] == pytest.approx(0.1)


def test_stem_energy_timestamps_and_normalized_bounds_include_partial_block(tmp_path: Path) -> None:
    wav = tmp_path / "partial.wav"
    samples = np.ones((25, 2), dtype=np.float32)
    sf.write(wav, samples, 100)

    times, values, metadata = builder.stem_energy(wav)

    assert times == [0.0, 0.1, 0.2, 0.25]
    assert len(times) == len(values)
    assert values[0] == 0.0
    assert all(0.0 <= value <= 1.0 for value in values)
    assert metadata["normalization_rms_p95"] == pytest.approx(1.0, rel=1e-4)


def _signals() -> dict:
    return {
        "beat_times_s": [0.0, 1.0, 2.0],
        "hits": [
            {"component": "kick", "t": 0.25, "velocity": 0.8},
            {"component": "snare", "t": 2.25, "velocity": 0.7},
        ],
        "energy": {
            "times_s": [0.0, 1.0, 2.0, 3.0],
            "bass": [0.2, 0.3, 0.4, 0.2],
            "vocals": [0.1, 0.7, 0.8, 0.2],
            "other": [0.2, 0.2, 0.3, 0.4],
        },
    }


def _write_replay_package(tmp_path: Path) -> tuple[Path, dict, dict]:
    package = tmp_path / "replay"
    package.mkdir()
    signals = _signals()
    plan = make_plan(signals, 1.0, 3.0)
    (package / "signals.json").write_text(json.dumps(signals, indent=2) + "\n")
    (package / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    sf.write(package / "original.wav", np.zeros((20, 2), dtype=np.float32), 10)
    manifest = {
        "schema_version": 1,
        "origins": {"fixture": True},
        "outputs": [fingerprint(package / name) for name in ("signals.json", "plan.json", "original.wav")],
    }
    (package / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return package, signals, plan


def test_replay_inputs_loads_saved_fixture_without_prepare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package, signals, plan = _write_replay_package(tmp_path)
    monkeypatch.setattr(builder, "prepare", lambda *args, **kwargs: pytest.fail("prepare must not run during replay"))

    loaded_signals, loaded_plan, pcm, sr, origins = builder.replay_inputs(package)

    assert loaded_signals == signals
    assert loaded_plan == plan
    assert pcm.shape == (20, 2)
    assert sr == 10
    assert origins["fixture"] is True
    assert origins["replayed_from"]["sha256"] == builder.fingerprint(package / "manifest.json")["sha256"]


@pytest.mark.parametrize("filename", ["signals.json", "plan.json", "original.wav"])
def test_replay_inputs_rejects_changed_saved_evidence(tmp_path: Path, filename: str) -> None:
    package, _, _ = _write_replay_package(tmp_path)
    path = package / filename
    if filename.endswith(".json"):
        payload = json.loads(path.read_text())
        if filename == "signals.json":
            payload["beat_times_s"].append(2.5)
        else:
            payload["uncertainty"] = "changed"
        path.write_text(json.dumps(payload) + "\n")
    else:
        path.write_bytes(path.read_bytes() + b"changed")

    with pytest.raises(ValueError, match="changed"):
        builder.replay_inputs(package)


def test_replay_inputs_rejects_override_with_wrong_audio_interval(tmp_path: Path) -> None:
    package, _, plan = _write_replay_package(tmp_path)
    override = tmp_path / "override-plan.json"
    edited = dict(plan)
    edited["end_s"] = 2.5
    edited["segments"] = [dict(edited["segments"][0], end_s=2.5)]
    override.write_text(json.dumps(edited) + "\n")

    with pytest.raises(ValueError, match="audio excerpt interval"):
        builder.replay_inputs(package, override)


def test_build_refuses_overwrite_before_prepare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "existing"
    out.mkdir()
    monkeypatch.setattr(builder, "prepare", lambda *args, **kwargs: pytest.fail("prepare ran before overwrite check"))

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        builder.build(out, review=tmp_path / "review")
