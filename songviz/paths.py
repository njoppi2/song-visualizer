from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path


def safe_dirname(name: str, *, max_len: int = 96) -> str:
    """
    Make a human-readable, cross-platform-ish directory name.
    Keeps spaces/dashes/parentheses, strips path separators and odd chars.
    """
    s = name.strip()
    if not s:
        return "song"

    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("/", "-").replace("\\", "-")
    s = re.sub(r"[^\w .()\\[\\]-]+", "_", s, flags=re.ASCII)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        s = "song"
    if len(s) > max_len:
        s = s[:max_len].rstrip()
    return s


def _read_song_id_from_analysis_json(path: Path) -> str | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    try:
        return str(data["meta"]["song_id"])
    except Exception:
        return None


def output_dir_for_audio(audio_path: str | Path, song_id: str, *, outputs_root: str | Path = "outputs") -> Path:
    p = Path(audio_path)
    label = safe_dirname(p.stem)
    root = Path(outputs_root)

    base = root / label
    analysis_new = base / "analysis" / "analysis.json"
    analysis_old = base / "analysis.json"  # legacy from earlier layout

    existing_id = None
    if analysis_new.exists():
        existing_id = _read_song_id_from_analysis_json(analysis_new)
    elif analysis_old.exists():
        existing_id = _read_song_id_from_analysis_json(analysis_old)

    if existing_id is not None and existing_id != str(song_id):
        return root / f"{label}__{song_id}"
    return base


def analysis_path_for_output_dir(out_dir: Path) -> Path:
    return out_dir / "analysis" / "analysis.json"


def story_path_for_output_dir(out_dir: Path) -> Path:
    return out_dir / "analysis" / "story.json"


def video_path_for_output_dir(out_dir: Path) -> Path:
    return out_dir / "video.mp4"


def lyrics_alignment_path_for_output_dir(out_dir: Path) -> Path:
    return out_dir / "lyrics" / "alignment.json"


def lyrics_corrections_path_for_output_dir(out_dir: Path) -> Path:
    return out_dir / "lyrics" / "corrections.yaml"
