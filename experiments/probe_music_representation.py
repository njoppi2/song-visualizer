#!/usr/bin/env python3
"""Pinned MuQ CPU feasibility probe and immutable full-song frame extraction."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import resource
import shutil
import sys
import time
from pathlib import Path


TARGET_SAMPLE_RATE = 24_000
MODEL_ID = "OpenMuQ/MuQ-large-msd-iter"
MODEL_REVISION = "0562a57814f6f8bbd9fdea0a25921a2fce1a841a"
WEIGHTS_SHA256 = "273febab2be02872c37d2c37e48a9d6c52c1c9392f3eeeabd498efa281ccb7a6"
CONFIG_SHA256 = "237335ee27d8fb951ce778701a12a79e06c51ae636dd786f97e45f51ce532543"
SOURCE_SHA256 = "657af9333edafbb6cf8cc573446650773d1169b0117c784106d1895cf802cb44"
OWNERSHIP = "closest eligible chunk center; earlier chunk wins exact ties"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_sample_bounds(start_s: float, duration_s: float, sample_rate: int) -> tuple[int, int]:
    """Return exact source sample bounds, rejecting off-grid source times."""
    if not math.isfinite(start_s) or not math.isfinite(duration_s) or start_s < 0 or duration_s <= 0 or sample_rate <= 0:
        raise ValueError("expected nonnegative start and positive finite duration/sample rate")
    start = round(start_s * sample_rate)
    end = round((start_s + duration_s) * sample_rate)
    if abs(start / sample_rate - start_s) > 1e-9:
        raise ValueError("start_s is not on the source sample clock")
    if abs(end / sample_rate - (start_s + duration_s)) > 1e-9:
        raise ValueError("end time is not on the source sample clock")
    if end <= start:
        raise ValueError("duration must include at least one source sample")
    return start, end


def nominal_frame_times_s(start_s: float, frame_count: int) -> list[float]:
    """MuQ's 240-sample mel hop and 4x padded conv clock: 25 Hz from chunk start."""
    return [start_s + index / 25 for index in range(frame_count)]


def verify_model(model_dir: Path) -> None:
    if sha256(model_dir / "config.json") != CONFIG_SHA256:
        raise ValueError("MuQ config does not match the pinned checkpoint")
    if sha256(model_dir / "model.safetensors") != WEIGHTS_SHA256:
        raise ValueError("MuQ weights do not match the pinned checkpoint")


def verify_architecture(model) -> float:
    strides = [block.conv1.stride[-1] for block in model.model.conv.conv]
    c = model.config
    if (c.hop_length, c.encoder_depth, c.encoder_dim, strides) != (240, 12, 1024, [2, 2]):
        raise ValueError(f"unexpected MuQ architecture: {c.hop_length=}, {c.encoder_depth=}, {c.encoder_dim=}, {strides=}")
    return TARGET_SAMPLE_RATE / c.hop_length / (strides[0] * strides[1])


def run_probe(audio: Path, model_dir: Path, start_s: float, duration_s: float, threads: int = 4) -> dict:
    import muq
    import numpy as np
    import soundfile as sf
    import torch

    verify_model(model_dir)
    info = sf.info(audio)
    if info.channels != 2:
        raise ValueError(f"expected stereo source mix, got {info.channels} channels")
    start_frame, end_frame = source_sample_bounds(start_s, duration_s, info.samplerate)
    if end_frame > info.frames:
        raise ValueError("requested source support ends after audio")
    with sf.SoundFile(audio) as handle:
        handle.seek(start_frame)
        source_pcm = handle.read(end_frame - start_frame, dtype="float32", always_2d=True)
    mono = source_pcm.mean(axis=1, dtype=np.float32)
    torch.set_num_threads(threads)
    wave = torch.from_numpy(mono).unsqueeze(0)
    wave_24k = __import__("torchaudio").functional.resample(
        wave, info.samplerate, TARGET_SAMPLE_RATE
    )
    # The source PCM cut occurs before resampling: no audio outside [start_s, end_s]
    # is fed to the encoder.  The resampler's internal filter/padding is not a source
    # support extension; the encoder itself attends to the entire retained chunk.
    rss_before_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    model = muq.MuQ.from_pretrained(str(model_dir)).cpu().eval()
    frame_clock_hz = verify_architecture(model)
    rss_after_load_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    started = time.perf_counter()
    with torch.inference_mode():
        output = model(wave_24k.float(), output_hidden_states=True)
    elapsed_s = time.perf_counter() - started
    rss_after_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    features = output.last_hidden_state.detach().cpu().float().numpy()[0]
    if frame_clock_hz != 25:
        raise ValueError(f"unexpected derived frame clock {frame_clock_hz}")
    times = nominal_frame_times_s(start_s, features.shape[0])
    return {
        "schema_version": 1,
        "kind": "songviz-muq-runtime-probe",
        "source": {
            "path": str(audio), "sha256": sha256(audio), "sample_rate": info.samplerate,
            "channels": info.channels, "source_frame_bounds": [start_frame, end_frame],
            "support_start_s": start_frame / info.samplerate,
            "support_end_s": end_frame / info.samplerate,
        },
        "model": {
            "id": MODEL_ID, "revision": MODEL_REVISION, "directory": str(model_dir),
            "config_sha256": CONFIG_SHA256, "weights_sha256": WEIGHTS_SHA256,
        },
        "extraction": {
            "device": "cpu", "dtype": "float32", "source_cut_before_resample": True,
            "source_to_model_resample": f"torchaudio.functional.resample {info.samplerate}Hz to {TARGET_SAMPLE_RATE}Hz",
            "encoder_input_samples": int(wave_24k.shape[-1]), "encoder_input_sample_rate": TARGET_SAMPLE_RATE,
            "frame_clock_hz": frame_clock_hz, "frame_clock_definition": "chunk_start + frame_index / 25; 24kHz / 240 hop / 2 / 2 conv strides",
            "feature_shape": list(features.shape), "hidden_state_count": len(output.hidden_states),
            "finite": bool(np.isfinite(features).all()), "frame_times_s": times,
            "support_start_s": [start_frame / info.samplerate] * features.shape[0],
            "support_end_s": [end_frame / info.samplerate] * features.shape[0],
        },
        "measurement": {
            "inference_elapsed_s": elapsed_s, "maxrss_before_kib": rss_before_kib,
            "maxrss_after_load_kib": rss_after_load_kib, "maxrss_after_kib": rss_after_kib,
            "torch_threads": threads, "python": sys.version, "platform": platform.platform(),
            "torch": torch.__version__, "torch_cuda_available": torch.cuda.is_available(),
        },
    }


def chunk_geometry(source_frames: int, source_rate: int, offset_s: float) -> list[dict]:
    """Derive complete coverage and frame ownership without rounded global keys."""
    if offset_s not in (0.0, 2.5) or source_frames <= 0 or source_rate <= 0:
        raise ValueError("expected positive source extent and base 0 or shifted 2.5 seconds")
    duration = source_frames / source_rate
    starts = [0.0] if offset_s else []
    start = offset_s
    while start < duration:
        starts.append(start)
        start += 5.0
    chunks = []
    for start in starts:
        a, b = source_sample_bounds(start, min(10.0, duration - start), source_rate)
        samples = ((b - a) * TARGET_SAMPLE_RATE + source_rate - 1) // source_rate
        # Centered STFT has floor(N/240)+1 frames; MuQ removes the last one.
        # Two padded stride-2 convolutions each take ceil(length/2).
        count = (samples // 240 + 3) // 4
        chunks.append({"start_s": a / source_rate, "center_s": (a + b) / (2 * source_rate),
                       "support_start_s": a / source_rate, "support_end_s": b / source_rate,
                       "source_start_frame": a, "source_end_frame": b,
                       "encoder_input_samples": samples, "frame_count": count})
    for i, chunk in enumerate(chunks):
        left = 0.0 if i == 0 else (chunks[i - 1]["center_s"] + chunk["center_s"]) / 2
        right = duration if i + 1 == len(chunks) else (chunk["center_s"] + chunks[i + 1]["center_s"]) / 2
        indices = []
        for j in range(chunk["frame_count"]):
            t = chunk["start_s"] + j / 25
            if t >= chunk["support_end_s"]:
                continue
            # Truncated edge chunks can start after the center midpoint. Only
            # chunks that actually contain t may own it, otherwise a gap appears.
            eligible = [(abs(t - other["center_s"]), k) for k, other in enumerate(chunks)
                        if other["support_start_s"] <= t < other["support_end_s"]]
            if min(eligible)[1] == i:
                indices.append(j)
        chunk.update(center_midpoint_left_s=left, center_midpoint_right_s=right,
                     retained_frame_indices=indices, retained_frame_count=len(indices))
    return chunks


def fingerprint(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size}


def run_full(audio: Path, model_dir: Path, output_dir: Path, offset_s: float, threads: int) -> None:
    """Extract final MuQ features in deterministic 10s / 5s source-clock chunks."""
    import muq
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio

    audio, model_dir, output_dir = audio.resolve(), model_dir.resolve(), output_dir.resolve()
    if output_dir.exists():
        raise ValueError(f"refusing to overwrite existing output directory: {output_dir}")
    verify_model(model_dir)
    source_hash = sha256(audio)
    if source_hash != SOURCE_SHA256:
        raise ValueError("source FLAC does not match the pinned source hash")
    info = sf.info(audio)
    if info.channels != 2 or info.samplerate != 44100 or threads < 1:
        raise ValueError("expected pinned stereo 44.1kHz source and positive thread count")
    chunks = chunk_geometry(info.frames, info.samplerate, offset_s)
    code_record = fingerprint(Path(__file__))
    model_records = [fingerprint(model_dir / "model.safetensors"), fingerprint(model_dir / "config.json")]
    library_records = [fingerprint(p) for p in sorted(Path(muq.__file__).parent.rglob("*.py"))]
    versions = {name: importlib.metadata.version(name) for name in
                ("muq", "torch", "torchaudio", "transformers", "huggingface-hub", "numpy", "soundfile", "scipy", "safetensors", "einops", "easydict")}
    torch.set_num_threads(threads)
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    model = muq.MuQ.from_pretrained(str(model_dir)).cpu().eval()
    frame_clock = verify_architecture(model)
    duration = info.frames / info.samplerate
    rows = []
    elapsed = []
    with sf.SoundFile(audio) as handle, torch.inference_mode():
        for i, chunk in enumerate(chunks):
            a, b = chunk["source_start_frame"], chunk["source_end_frame"]
            handle.seek(a)
            pcm = handle.read(b - a, dtype="float32", always_2d=True)
            mono = pcm.mean(axis=1, dtype=np.float32)
            wave = torchaudio.functional.resample(torch.from_numpy(mono).unsqueeze(0), info.samplerate, TARGET_SAMPLE_RATE)
            if wave.shape[-1] != chunk["encoder_input_samples"]:
                raise ValueError("resampled input length differs from source-clock geometry")
            begun = time.perf_counter()
            out = model(wave.float(), output_hidden_states=True).last_hidden_state.detach().cpu().float().numpy()[0]
            elapsed.append(time.perf_counter() - begun)
            if out.shape != (chunk["frame_count"], 1024) or not np.isfinite(out).all():
                raise ValueError("actual model output disagrees with frame geometry or is nonfinite")
            for j in chunk["retained_frame_indices"]:
                rows.append((chunk["start_s"] + j / frame_clock, chunk["support_start_s"], chunk["support_end_s"], out[j].copy()))
            if (i + 1) % 10 == 0 or i + 1 == len(chunks):
                print(f"offset={offset_s}: {i+1}/{len(chunks)} chunks; inference={sum(elapsed):.2f}s", flush=True)
    embeddings = np.stack([r[3] for r in rows]).astype("float32", copy=False)
    times = np.array([r[0] for r in rows], dtype="float64")
    support_a = np.array([r[1] for r in rows], dtype="float64")
    support_b = np.array([r[2] for r in rows], dtype="float64")
    if not np.isfinite(embeddings).all() or embeddings.shape[1] != 1024:
        raise ValueError("non-finite or unexpected MuQ final hidden features")
    if times[0] != 0 or np.any(np.diff(times) <= 0) or np.max(np.diff(times)) > .060001 or duration - times[-1] > .040001:
        raise ValueError("frame ownership leaves unexpected coverage gaps")
    for row in [code_record, *model_records, *library_records, {"path": str(audio), "sha256": source_hash}]:
        if sha256(Path(row["path"])) != row["sha256"]:
            raise ValueError("an input changed during extraction")
    output_dir.mkdir(parents=True)
    snapshot_dir = output_dir / "inputs"
    snapshot_dir.mkdir()
    shutil.copy2(Path(__file__), snapshot_dir / Path(__file__).name)
    shutil.copy2(model_dir / "config.json", snapshot_dir / "model-config.json")
    for row in library_records:
        source_path = Path(row["path"])
        target = snapshot_dir / "muq" / source_path.relative_to(Path(muq.__file__).parent)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target)
    frames = output_dir / "frames.npz"
    np.savez_compressed(frames, embeddings=embeddings, frame_times_s=times,
                        support_start_s=support_a, support_end_s=support_b)
    manifest = {
        "schema_version": 1, "kind": "songviz-muq-runtime", "source": {**fingerprint(audio), "source_audio_sha256": source_hash,
                        "sample_rate": info.samplerate, "channels": info.channels, "frames": info.frames, "duration_s": duration},
        "checkpoint": {**model_records[0], "id": MODEL_ID, "revision": MODEL_REVISION},
        "config": {**model_records[1], "encoder_depth": 12, "encoder_dim": 1024, "hop_length": 240},
        "extraction": {"encoder_input_sample_rate": TARGET_SAMPLE_RATE, "dtype": "float32", "eval": True,
                       "selected_layer": "last_hidden_state", "source_cut_before_resample": True, "mono": "arithmetic mean of stereo float32 PCM", "source_frame_count": info.frames},
        "code": fingerprint(snapshot_dir / Path(__file__).name),
        "clock": {"source_seconds": True, "frame_hz": frame_clock, "chunk_seconds": 10, "hop_seconds": 5, "offset_seconds": offset_s, "ownership": OWNERSHIP},
        "chunks": chunks,
        "frames": {"path": "frames.npz", "sha256": sha256(frames), "source_audio_sha256": source_hash, "shape": list(embeddings.shape), "finite": True},
        "runtime": {"device": "cpu", "dtype": "float32", "torch_threads": threads, "chunk_inference_seconds": elapsed, "maxrss_before_kib": rss_before, "maxrss_after_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
        "versions": {"python": sys.version, **versions},
        "library_sources": library_records,
        "input_snapshots": [fingerprint(p) for p in sorted(snapshot_dir.rglob("*")) if p.is_file()],
        "support": "Each selected frame retains the full source PCM chunk supplied to the attention encoder; PCM is cut before per-chunk 44.1kHz-to-24kHz torchaudio resampling.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--start-s", type=float)
    parser.add_argument("--duration-s", type=float)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--full-output-dir", type=Path)
    parser.add_argument("--offset-s", type=float, default=0.0)
    args = parser.parse_args()
    if args.full_output_dir:
        run_full(args.audio, args.model_dir, args.full_output_dir, args.offset_s, args.threads)
        return
    if args.output is None or args.start_s is None or args.duration_s is None:
        raise SystemExit("--output, --start-s and --duration-s are required unless --full-output-dir is used")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    result = run_probe(args.audio, args.model_dir, args.start_s, args.duration_s, args.threads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"feature_shape": result["extraction"]["feature_shape"], **result["measurement"]}, sort_keys=True))


if __name__ == "__main__":
    main()
