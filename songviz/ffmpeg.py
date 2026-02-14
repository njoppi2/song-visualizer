from __future__ import annotations

import shutil


def require_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if path:
        return path
    raise RuntimeError(
        "ffmpeg was not found on PATH. Install it to enable MP4 output.\n"
        "Linux (Debian/Ubuntu): sudo apt-get update && sudo apt-get install -y ffmpeg\n"
        "macOS (Homebrew): brew install ffmpeg\n"
        "Windows (winget): winget install Gyan.FFmpeg"
    )

