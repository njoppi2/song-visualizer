"""Build short, reproducible review clips from cached analysis without ML runs.

Run from the repo root:
  .songviz/venv/bin/python experiments/build_review.py --out outputs/reviews/restart-01

Creates a new directory only. Keeps full-song timestamps for visual rendering,
sample-exact source audio cuts, diagnostic video, stem audio, raw-event clicks,
input snapshots, checksums, and a local HTML page with downloadable feedback.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import html
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from songviz.ingest import sha256_file
from songviz.paths import output_dir_for_audio
from songviz.render import RenderConfig, Visualizer, _render_mp4_with_visualizer


def run(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def fingerprint(path: Path) -> dict:
    return {"path": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path), "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()}


def cut_audio(path: Path, start: float, end: float) -> tuple[np.ndarray, int]:
    with sf.SoundFile(path) as f:
        lo, hi = round(start * f.samplerate), round(end * f.samplerate)
        if not 0 <= lo < hi <= len(f):
            raise ValueError(f"Excerpt outside {path}: {start}..{end}")
        f.seek(lo)
        return f.read(hi - lo, dtype="float32", always_2d=True), f.samplerate


def rms_curve(samples: np.ndarray, sr: int, start: float) -> tuple[np.ndarray, np.ndarray]:
    hop = max(1, round(sr * 0.05))
    # Retain energy in each channel before averaging (avoid stereo cancellation).
    chunks = [samples[i:i + hop] for i in range(0, len(samples), hop)]
    values = np.array([np.sqrt(np.mean(chunk.astype(np.float64) ** 2)) for chunk in chunks])
    times = start + np.arange(len(chunks)) * hop / sr
    return times, values


def drum_clicks(reduced: dict, start: float, end: float, sr: int = 22050) -> np.ndarray:
    """Audition raw event TIMES. No beat snapping or filtering; not a drum resynthesis."""
    buf = np.zeros(round((end - start) * sr), dtype=np.float32)
    frequencies = {"kick": 150, "snare": 600, "hh": 1600, "toms": 300, "ride": 2100, "crash": 2600}
    for hit in reduced.get("drums", {}).get("hits", []):
        offset = round((float(hit["t"]) - start) * sr)
        length = round(0.04 * sr)
        if offset >= len(buf) or offset + length <= 0:
            continue
        t = np.arange(length) / sr
        tone = np.sin(2 * np.pi * frequencies.get(hit["component"], 900) * t) * np.exp(-t * 120)
        lo, hi = max(0, offset), min(len(buf), offset + length)
        buf[lo:hi] += tone[lo - offset:hi - offset] * float(hit.get("velocity", 0.5))
    peak = np.max(np.abs(buf)) if buf.size else 0
    if peak > 0:
        buf *= 0.8 / peak
    return buf


class ReviewVisualizer:
    """Current mix renderer plus excerpt-local signals on the same source clock."""

    def __init__(self, analysis: dict, reduced: dict, cfg: RenderConfig,
                 start: float, end: float, curves: dict):
        self.start, self.end, self.cfg = start, end, cfg
        self.base = Visualizer(analysis, replace(cfg, height=360))
        self.panel = Image.new("RGB", (cfg.width, cfg.height - 360), "#101820")
        try:
            self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        except OSError:
            self.font = ImageFont.load_default()
        draw = ImageDraw.Draw(self.panel)
        self.x0, self.x1 = 86, cfg.width - 12
        self.sections = analysis.get("story", {}).get("sections", [])
        draw.text((8, 4), "SOURCE AUDIO + CURRENT MIX RENDERER / CACHED ANALYSIS", font=self.font, fill="#dce8ed")
        colors = {"mix": "#dde8ed", "drums": "#73bbff", "bass": "#64d8ab"}
        # One shared absolute RMS scale for this excerpt, not per-stem normalization.
        peak = max((float(v.max()) for _, v in curves.values()), default=1.0) or 1.0
        for row, (name, (times, values)) in enumerate(curves.items()):
            y = 55 + row * 27
            draw.text((8, y - 16), name + " RMS", font=self.font, fill=colors[name])
            points = [(self.x(t), y - 21 * float(v) / peak) for t, v in zip(times, values)]
            if len(points) > 1:
                draw.line(points, fill=colors[name], width=1)
        draw.text((8, 123), "drum hits", font=self.font, fill="#ffbf6b")
        for hit in reduced.get("drums", {}).get("hits", []):
            if start <= hit["t"] < end:
                x = self.x(hit["t"])
                draw.line((x, 122, x, 135), fill="#ffbf6b")
        for beat in analysis.get("beats", {}).get("beat_times_s", []):
            if start <= beat < end:
                x = self.x(beat)
                draw.line((x, 143, x, 148), fill="#7e929e")
        draw.text((8, 139), "beats", font=self.font, fill="#7e929e")
        for sec in self.sections:
            if start < sec["start_s"] < end:
                x = self.x(sec["start_s"])
                draw.line((x, 28, x, 148), fill="#b46cb8")
        draw.text((8, 161), "Purple: cached section boundary. Curves: measured stem energy, not ground truth.",
                  font=self.font, fill="#9fb1bf")

    def x(self, time: float) -> int:
        return round(self.x0 + (time - self.start) / (self.end - self.start) * (self.x1 - self.x0))

    def frame_rgb24(self, t: float) -> bytes:
        source_t = self.start + t
        img = Image.new("RGB", (self.cfg.width, self.cfg.height))
        img.paste(Image.frombytes("RGB", (self.cfg.width, 360), self.base.frame_rgb24(source_t)))
        panel = self.panel.copy()
        draw = ImageDraw.Draw(panel)
        role = next((s.get("role", "unknown") for s in self.sections
                     if s["start_s"] <= source_t < s["end_s"]), "unknown")
        draw.text((8, 19), f"Song {source_t:06.2f}s | excerpt {t:05.2f}s | inferred role: {role}",
                  font=self.font, fill="white")
        x = self.x(source_t)
        draw.line((x, 35, x, 148), fill="white", width=2)
        img.paste(panel, (0, 360))
        return img.tobytes()


def build(config: Path, out: Path) -> None:
    spec = json.loads(config.read_text())
    out.mkdir(parents=True, exist_ok=False)
    inputs = out / "inputs"
    inputs.mkdir()
    manifest = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                "baseline_type": "current mix renderer on historical cached analysis; no extraction rerun",
                "cache_generation_commit": "unknown", "perceptually_validated": False,
                "git_head": run("git", "rev-parse", "HEAD"),
                "git_status": run("git", "status", "--short"),
                "python": sys.version, "ffmpeg": run("ffmpeg", "-version").splitlines()[0],
                "dependencies": {}, "passages": [], "inputs": []}
    for name in ("numpy", "Pillow", "soundfile", "librosa"):
        manifest["dependencies"][name] = importlib.metadata.version(name)
    (inputs / "working-tree.patch").write_text(run("git", "diff", "--binary") + "\n")
    shutil.copy2(config, inputs / "passages.json")
    # Snapshot the renderer and builder as well as the tracked diff (builder can be untracked).
    for path in (Path(__file__), ROOT / "songviz/render.py", ROOT / "songviz/ffmpeg.py",
                 ROOT / "songviz/ingest.py", ROOT / "songviz/paths.py", ROOT / "experiments/templates/review.html"):
        dest = inputs / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        manifest["inputs"].append(fingerprint(path))
    loaded = {}
    ref_index = json.loads((ROOT / "benchmark/songs.json").read_text())
    for passage in spec["passages"]:
        name = passage["song"]
        if name in loaded:
            continue
        song = ROOT / "songs" / name
        audio_fp = fingerprint(song)
        cached = output_dir_for_audio(song, audio_fp["sha256"][:16], outputs_root=ROOT / "outputs")
        analysis_p = cached / "analysis/analysis.json"
        analysis = json.loads(analysis_p.read_text())
        if analysis["meta"]["song_id"] != audio_fp["sha256"][:16]:
            raise ValueError(f"Source hash mismatch: {song}")
        stem_meta = json.loads((cached / "stems/stems.json").read_text())
        if stem_meta["input"]["sha256"] != audio_fp["sha256"]:
            raise ValueError(f"Stem input mismatch: {song}")
        reduced = json.loads((cached / "analysis/reduced.json").read_text())
        snapshot = inputs / song.stem
        snapshot.mkdir()
        for rel in ("analysis/analysis.json", "analysis/reduced.json", "analysis/story.json", "stems/stems.json"):
            src = cached / rel
            shutil.copy2(src, snapshot / Path(rel).name)
            manifest["inputs"].append(fingerprint(src))
        for stem in ("drums", "bass"):
            manifest["inputs"].append(fingerprint(cached / "stems" / f"{stem}.wav"))
        manifest["inputs"].append(audio_fp)
        manifest.setdefault("analysis_metadata", {})[name] = analysis["meta"]
        manifest.setdefault("stem_backend", {})[name] = stem_meta["backend"]
        manifest.setdefault("reduction_sources", {})[name] = {
            k: reduced.get(k, {}).get("source", "unknown") for k in ("drums", "bass", "vocals")}
        ref_name = ref_index.get(audio_fp["sha256"][:16])
        if ref_name:
            ref_dir = ROOT / "benchmark/references" / ref_name
            shutil.copytree(ref_dir, inputs / "references" / ref_name)
            manifest["inputs"].extend(fingerprint(p) for p in sorted(ref_dir.iterdir()) if p.is_file())
        loaded[name] = (song, cached, analysis, reduced)
    cfg = RenderConfig(width=640, height=540, fps=24, seed=0, audio_codec="aac", audio_bitrate="192k")
    manifest["render_config"] = asdict(cfg)
    for p in spec["passages"]:
        print(f"Building {p['id']}: {p['start_s']}..{p['end_s']}s", flush=True)
        song, cached, analysis, reduced = loaded[p["song"]]
        dest = out / p["id"]
        dest.mkdir()
        start, end = float(p["start_s"]), float(p["end_s"])
        curves = {}
        for label, source in (("mix", song), ("drums", cached / "stems/drums.wav"), ("bass", cached / "stems/bass.wav")):
            samples, sr = cut_audio(source, start, end)
            sf.write(dest / f"{label}.wav", samples, sr, subtype="PCM_24")
            curves[label] = rms_curve(samples, sr, start)
        sf.write(dest / "detected-drums.wav", drum_clicks(reduced, start, end), 22050, subtype="PCM_24")
        visualizer = ReviewVisualizer(analysis, reduced, cfg, start, end, curves)
        Image.frombytes("RGB", (cfg.width, cfg.height), visualizer.frame_rgb24(0)).save(dest / "poster.png")
        _render_mp4_with_visualizer(audio_path=dest / "mix.wav", out_path=dest / "preview.mp4",
                                   cfg=cfg, duration_s=end-start, visualizer=visualizer)
        (dest / "signals.json").write_text(json.dumps({
            "absolute_song_seconds": True, "rms_window_s": 0.05,
            "curves": {k: {"times_s": t.tolist(), "rms": v.tolist()} for k, (t, v) in curves.items()},
            "sections": [s for s in analysis["story"]["sections"] if s["start_s"] < end and s["end_s"] > start],
            "drum_hits": [h for h in reduced["drums"]["hits"] if start <= h["t"] < end]}, indent=2) + "\n")
        manifest["passages"].append({**p, "review_status": "awaiting_user",
                                     "artifacts": [fingerprint(f) for f in sorted(dest.iterdir())]})
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    cards = []
    for p in spec["passages"]:
        ident = html.escape(p["id"], quote=True)
        cards.append(f'''<section data-id="{ident}" data-start="{p['start_s']}">
<h2>{html.escape(p['title'])}</h2><p>{p['start_s']}–{p['end_s']}s in the original song</p>
<video controls preload="metadata" poster="{ident}/poster.png" src="{ident}/preview.mp4"></video>
<p>{html.escape(p['question'])}</p><details><summary>Listen to the same excerpt in isolation</summary>
<label>Separated drums<audio controls preload="none" src="{ident}/drums.wav"></audio></label>
<label>Detected drum times (pitched clicks, no quantization)<audio controls preload="none" src="{ident}/detected-drums.wav"></audio></label>
<label>Separated bass<audio controls preload="none" src="{ident}/bass.wav"></audio></label>
<p>These players start at the excerpt beginning. Stems can contain separation artifacts.
Clicks represent cached events; they do not recreate drum timbre.</p></details>
<button class="stamp">Use current video time</button><input class="time" aria-label="Song time in seconds" type="number" step="0.01" value="{p['start_s']}">
<textarea aria-label="Your observation" placeholder="What feels wrong or could be better? A short description is enough."></textarea>
</section>''')
    template = (ROOT / "experiments/templates/review.html").read_text()
    (out / "index.html").write_text(template.replace("{{CARDS}}", "\n".join(cards)).replace("{{MANIFEST_SHA}}", sha256_file(manifest_path)))
    total_s = sum(p["end_s"] - p["start_s"] for p in spec["passages"])
    (out / "README.md").write_text(f"# First review\n\nOpen `index.html` in a browser. {len(spec['passages'])} clips total {total_s:g} seconds.\n\n"
        "The original song is the video soundtrack. Visuals are newly generated with the current mix renderer, driven by cached analysis. "
        "This is a historical analysis baseline, not a verified run of current extraction. "
        "Use the original-song time when describing an issue. Export observations with Download feedback; browser edits are not saved automatically.\n\n"
        "Provenance and input snapshots: `manifest.json` and `inputs/`. No subjective review has happened yet.\n")
    print(f"Review ready: {out / 'index.html'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "benchmark/review_passages.json")
    args = parser.parse_args()
    build(args.config.resolve(), args.out.resolve())
