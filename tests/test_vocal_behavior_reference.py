import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from experiments.build_vocal_behavior_reference import build, extract_pcm24_excerpt


ROOT = Path(__file__).resolve().parents[1]


def make_source(path, rate=8000, channels=2):
    # int32 values are aligned to PCM_24 exactly, making source/output parity observable.
    frames = rate * 20
    values = (np.arange(frames * channels, dtype=np.int32).reshape(frames, channels) % 65536 - 32768) << 8
    sf.write(path, values, rate, subtype="PCM_24", format="WAV")
    return hashlib.sha256(path.read_bytes()).hexdigest(), values


def test_extract_native_pcm24_and_exact_decoded_frames(tmp_path):
    source = tmp_path / "source.wav"; expected, samples = make_source(source, rate=8000, channels=1)
    target = tmp_path / "excerpt.wav"
    record = extract_pcm24_excerpt(source, target, expected_hash=expected, start_s=3, end_s=7)
    assert (record["sample_rate"], record["channels"], record["frames"], record["subtype"]) == (8000, 1, 32000, "PCM_24")
    written, rate = sf.read(target, dtype="int32", always_2d=True)
    assert rate == 8000
    np.testing.assert_array_equal(written, samples[24000:56000])


def test_build_binds_audio_and_refuses_bad_source_or_existing_output(tmp_path):
    source = tmp_path / "source.wav"; expected, _ = make_source(source)
    output = tmp_path / "package"
    build(ROOT, output, source=source, expected_source_hash=expected, start_s=3, end_s=7)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["source"]["sha256"] == expected
    assert manifest["excerpt"]["start_song_s"] == 3
    assert manifest["identity"]["builder_sha256"] == hashlib.sha256((ROOT / "experiments/build_vocal_behavior_reference.py").read_bytes()).hexdigest()
    assert manifest["identity"]["template_sha256"] == hashlib.sha256((ROOT / "experiments/templates/vocal_behavior_reference.html").read_bytes()).hexdigest()
    page = (output / "index.html").read_text()
    assert "excerpt.wav" in page and '"start_song_s": 3' in page
    assert 'id="start" type="number" step="0.01" required' in page
    assert "labels.length>0" in page and "Number.isFinite" in page
    assert "textContent" in page and "retry-audio" in page and "source.addEventListener('error',audioFailed)" in page
    with pytest.raises(FileExistsError):
        build(ROOT, output, source=source, expected_source_hash=expected, start_s=3, end_s=7)
    with pytest.raises(ValueError, match="FLAC hash"):
        build(ROOT, tmp_path / "bad", source=source, expected_source_hash="0" * 64, start_s=3, end_s=7)


def test_existing_destination_is_refused_before_source_read(tmp_path):
    output = tmp_path / "exists"; output.mkdir()
    with pytest.raises(FileExistsError):
        build(ROOT, output, source=tmp_path / "does-not-exist.wav", expected_source_hash="0" * 64)
