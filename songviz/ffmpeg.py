from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def require_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if path:
        return path

    # Next to the active Python (e.g., `.venv/bin/ffmpeg`) for user-space installs.
    try:
        cand = Path(sys.prefix) / "bin" / "ffmpeg"
        if cand.exists() and os.access(cand, os.X_OK):
            return str(cand)
    except Exception:
        pass

    # Fallback: user-space ffmpeg binary (no system install required).
    try:
        import imageio_ffmpeg  # type: ignore
    except Exception:
        imageio_ffmpeg = None  # type: ignore

    if imageio_ffmpeg is not None:
        try:
            exe = imageio_ffmpeg.get_ffmpeg_exe()
            if exe:
                return str(exe)
        except Exception:
            # If download fails, fall through to a clear install hint.
            pass

    raise RuntimeError(
        "ffmpeg was not found on PATH. Install it to enable MP4 output.\n"
        "Linux (Debian/Ubuntu): sudo apt-get update && sudo apt-get install -y ffmpeg\n"
        "macOS (Homebrew): brew install ffmpeg\n"
        "Windows (winget): winget install Gyan.FFmpeg\n"
        "Alternatively (no system install): pip install imageio-ffmpeg"
    )
