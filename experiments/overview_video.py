"""Generate a video where the overview.png is the background and a playhead line moves in sync with the audio."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def make_overview_video(song_path: str | Path, fps: int = 24) -> Path:
    song_path = Path(song_path)
    out_dir = Path("outputs") / song_path.stem
    analysis_path = out_dir / "analysis" / "analysis.json"
    overview_path = out_dir / "analysis" / "overview.png"
    output_path = out_dir / "overview_video.mp4"

    with open(analysis_path) as f:
        analysis = json.load(f)
    duration = analysis["meta"]["duration_s"]

    img = np.array(Image.open(overview_path).convert("RGB"))
    H, W = img.shape[:2]

    # Read plot-area pixel bounds (saved by generate_overview in viz.py)
    axes_meta_path = out_dir / "analysis" / "overview_axes.json"
    if axes_meta_path.exists():
        with open(axes_meta_path) as f:
            axes_meta = json.load(f)
        plot_x0 = axes_meta["plot_x0_px"]
        plot_x1 = axes_meta["plot_x1_px"]
    else:
        # Fallback: assume full width
        plot_x0, plot_x1 = 0, W

    n_frames = int(duration * fps) + 1

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{W}x{H}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "pipe:0",
        "-i", str(song_path),
        "-c:v", "libx264", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        "-pix_fmt", "yuv420p",
        "-shortest",
        str(output_path),
    ]

    print(f"Rendering {n_frames} frames ({duration:.1f}s @ {fps}fps) → {output_path}")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    plot_w = plot_x1 - plot_x0
    for i in range(n_frames):
        t = i / fps
        x = plot_x0 + int((t / duration) * plot_w)
        frame = img.copy()
        x0 = max(0, x - 1)
        x1 = min(W, x + 2)
        frame[:, x0:x1] = [255, 255, 255]
        proc.stdin.write(frame.tobytes())

    proc.stdin.close()
    proc.wait()
    print(f"Done: {output_path}")
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python experiments/overview_video.py <song_path>")
        sys.exit(1)
    make_overview_video(sys.argv[1])
