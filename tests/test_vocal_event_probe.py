import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from experiments.probe_vocal_events import (HOP, RATE, SCORE_COUNT, WINDOW, consecutive,
    fixed_gate, infer, prepare_inputs, run_screen, support_mask, validate_runtime, validate_waveform, windows)
import experiments.probe_vocal_events as probe


ROOT = Path(__file__).resolve().parents[1]


def scores(n): return np.zeros((n, SCORE_COUNT), dtype=np.float32)


def test_window_geometry_offsets_end_and_tail():
    spans, tail = windows(WINDOW + 2 * HOP + 99)
    assert spans == [(0, WINDOW), (HOP, HOP + WINDOW), (2 * HOP, 2 * HOP + WINDOW)]
    assert tail == 99
    assert windows(WINDOW - 1) == ([], WINDOW - 1)
    assert [(len(windows(seconds * RATE)[0]), windows(seconds * RATE)[1]) for seconds in (10, 11, 13)] == [(19, 6160), (21, 6800), (26, 400)]
    target_spans, _ = windows(13 * RATE)
    assert support_mask(target_spans, 119, (119, 123.72)).sum() == 8
    assert support_mask(target_spans, 119, (125.45, 132)).sum() == 12


def test_support_excludes_straddlers_and_gap_breaks_consecutive():
    spans = [(0, WINDOW), (HOP, HOP + WINDOW), (2 * HOP, 2 * HOP + WINDOW)]
    mask = support_mask(spans, 119.0, (119.0, 120.0))
    assert mask.tolist() == [True, False, False]  # end exactly at 119.975; later straddles/end exceeds
    assert not consecutive(np.array([True, True, True]), np.array([True, False, True]))


def test_fixed_gate_positive_negative_and_abstaining_are_synthetic_only():
    # Early and late supports with a gap; values are fabricated scores.
    spans = [(i * HOP, i * HOP + WINDOW) for i in range(25)]
    sample_scores = scores(len(spans))
    early = support_mask(spans, 119.0, (119, 123.72)); late = support_mask(spans, 119.0, (125.45, 132))
    sample_scores[early, 0] = .8; sample_scores[late, 13] = .8
    yes = fixed_gate(sample_scores, spans, 119.0)
    assert yes["narrow_behavior_support"] and yes["status"] == "evaluated"
    no = fixed_gate(scores(len(spans)), spans, 119.0)
    assert not no["narrow_behavior_support"] and no["status"] == "evaluated"
    abstain = fixed_gate(scores(1), [(0, WINDOW)], 119.0)
    assert abstain["status"].startswith("abstain") and abstain["late_laughter_positive_fraction"] is None


def test_halfgain_is_exact_float32_and_rejects_bad_input():
    source = np.array([.1, -.7, 1.0], dtype=np.float32)
    half = np.float32(.5) * source
    assert half.dtype == np.float32
    np.testing.assert_array_equal(half, np.array([.05, -.35, .5], dtype=np.float32))
    with pytest.raises(ValueError): validate_waveform(np.array([np.nan], dtype=np.float32))
    with pytest.raises(ValueError): validate_waveform(np.array([1.01], dtype=np.float32))


def test_runtime_pin_tamper_is_rejected_with_self_contained_fixture(tmp_path, monkeypatch):
    runtime, model_dir = tmp_path / "runtime", tmp_path / "model"
    runtime.mkdir(); model_dir.mkdir()
    names = ["Speech", *[f"class-{index}" for index in range(1, SCORE_COUNT)]]
    names[13], names[24], names[31] = "Laughter", "Singing", "Rapping"
    csv_text = "index,mid,display_name\n" + "".join(f'{i},/m/{i},{name}\n' for i, name in enumerate(names))
    class_map = model_dir / "yamnet_class_map.csv"; class_map.write_text(csv_text)
    model = model_dir / "yamnet.tflite"
    with zipfile.ZipFile(model, "w") as archive:
        archive.writestr("yamnet_label_list.txt", "\n".join(names) + "\n")
    synthetic = {"song_or_stem_opened": False, "interpreter": {"input_details": [{"shape": [WINDOW]}], "output_details": [{"shape": [1, SCORE_COUNT]}]}}
    result = runtime / "synthetic-result.json"; result.write_text(json.dumps(synthetic))
    upstream = runtime / "upstream.json"; upstream.write_text(json.dumps({"model": {"sha256": hashlib.sha256(model.read_bytes()).hexdigest()}}))
    manifest = runtime / "manifest.json"
    manifest.write_text(json.dumps({"kind": "songviz-yamnet-local-runtime-screen", "synthetic_execution": {"result_sha256": hashlib.sha256(result.read_bytes()).hexdigest()}}))
    monkeypatch.setattr(probe, "MODEL_SHA256", hashlib.sha256(model.read_bytes()).hexdigest())
    monkeypatch.setattr(probe, "MODEL_BYTES", model.stat().st_size)
    monkeypatch.setattr(probe, "CLASS_MAP_SHA256", hashlib.sha256(class_map.read_bytes()).hexdigest())
    monkeypatch.setattr(probe, "RUNTIME_MANIFEST_SHA256", hashlib.sha256(manifest.read_bytes()).hexdigest())
    monkeypatch.setattr(probe, "RUNTIME_UPSTREAM_SHA256", hashlib.sha256(upstream.read_bytes()).hexdigest())
    validate_runtime(runtime, model_dir)
    model.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="model"):
        validate_runtime(runtime, model_dir)


class DummyInterpreter:
    def __init__(self, row): self.row = row
    def get_input_details(self): return [{"index": 3}]
    def get_output_details(self): return [{"index": 7}]
    def set_tensor(self, index, value): assert index == 3 and value.shape == (WINDOW,)
    def invoke(self): pass
    def get_tensor(self, index): return self.row.reshape(1, -1)


@pytest.mark.parametrize("bad", [np.full(SCORE_COUNT, np.nan, dtype=np.float32), np.full(SCORE_COUNT, 1.1, dtype=np.float32)])
def test_infer_rejects_bad_scores_for_any_condition(bad):
    with pytest.raises(ValueError, match="score"):
        infer(DummyInterpreter(bad), np.zeros(WINDOW, dtype=np.float32), [(0, WINDOW)])


def test_recipe_order_offsets_and_exact_halfgain_without_audio(monkeypatch):
    paths = {name: Path(f"/unused/{name}.wav") for name in ("original", "demucs_vocals", "roformer_vocals", "roformer_other")}
    monkeypatch.setattr(probe, "verify_sources", lambda repo: {**paths, "hashes": {"fixture": "yes"}})
    calls = []
    def fake_load(path, start, end):
        calls.append((path.name, start, end))
        return np.full(round((end - start) * RATE), len(calls) / 10, dtype=np.float32)
    monkeypatch.setattr(probe, "load_resampled", fake_load)
    prepared, hashes = prepare_inputs(Path("/unused"))
    assert [item["input_id"] for item in prepared] == [f"input-{i:02d}" for i in range(1, 9)]
    assert [item["clip_start_s"] for item in prepared] == [16, 74, 119, 57, 119, 119, 119, 119]
    assert calls[-2:] == [("roformer_vocals.wav", 0.0, 13.0), ("roformer_other.wav", 0.0, 13.0)]
    np.testing.assert_array_equal(prepared[7]["samples"], np.float32(.5) * prepared[2]["samples"])
    assert hashes == {"fixture": "yes"}


def test_overwrite_is_rejected_before_audio_or_model_work(tmp_path):
    destination = tmp_path / "already-there"; destination.mkdir()
    with pytest.raises(FileExistsError):
        run_screen(ROOT, ROOT / "missing-runtime", ROOT / "missing-protocol", destination)
