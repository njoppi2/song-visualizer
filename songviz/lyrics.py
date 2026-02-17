from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any


def _whisper_align(
    audio_path: Path,
    *,
    language: str | None = None,
    model_name: str = "base",
) -> list[dict[str, Any]]:
    """Use openai-whisper with word timestamps to produce raw segment data."""
    try:
        import whisper  # type: ignore
    except ImportError as e:
        raise ImportError(
            "openai-whisper is required for lyrics alignment. "
            "Install it with: pip install -e '.[lyrics]'"
        ) from e

    model = whisper.load_model(model_name)
    decode_options: dict[str, Any] = {"word_timestamps": True}
    if language:
        decode_options["language"] = language

    result = model.transcribe(str(audio_path), **decode_options)
    return result.get("segments", [])


def _normalize_whisper_segments(
    raw_segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert openai-whisper segment output to the alignment.json segment format."""
    out = []
    for seg in raw_segments:
        words_raw = seg.get("words", [])
        words = []
        for w in words_raw:
            # Whisper prepends a space to most words; strip to get clean tokens.
            word_text = str(w.get("word", "")).strip()
            if not word_text:
                continue
            words.append(
                {
                    "word": word_text,
                    "start_s": float(w.get("start", 0.0)),
                    "end_s": float(w.get("end", 0.0)),
                    "confidence": float(w.get("probability", 0.0)),
                    # Phones not available from Whisper; empty list per contract.
                    "phones": [],
                }
            )
        out.append(
            {
                "text": str(seg.get("text", "")).strip(),
                "start_s": float(seg.get("start", 0.0)),
                "end_s": float(seg.get("end", 0.0)),
                "words": words,
            }
        )
    return out


def align_lyrics(
    audio_path: str | Path,
    *,
    song_id: str,
    output_dir: Path,
    vocals_stem: Path | None = None,
    language: str = "en",
    model_name: str = "base",
    force: bool = False,
) -> Path:
    """Run the lyrics alignment pipeline and write lyrics/alignment.json.

    Pipeline:
    1. Choose alignment audio — vocals stem preferred, full mix as fallback.
    2. Run Whisper with word timestamps (transcription + alignment in one pass).
    3. Normalize to the alignment.json contract.
    4. Write output.

    Returns the path to the written alignment.json.

    Requires: pip install -e '.[lyrics]'
    """
    from .paths import lyrics_alignment_path_for_output_dir

    out_path = lyrics_alignment_path_for_output_dir(output_dir)
    if out_path.exists() and not force:
        return out_path

    # Choose alignment source; prefer isolated vocals when available.
    if vocals_stem is not None and Path(vocals_stem).exists():
        align_audio = Path(vocals_stem)
        audio_source = "vocals_stem"
    else:
        align_audio = Path(audio_path)
        audio_source = "mix"

    raw_segments = _whisper_align(align_audio, language=language, model_name=model_name)
    segments = _normalize_whisper_segments(raw_segments)

    payload: dict[str, Any] = {
        "metadata": {
            "song_id": song_id,
            "language": language,
            "alignment_tool": "whisper",
            "whisper_model": model_name,
            "audio_source": audio_source,
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
        "segments": segments,
        "quality_flags": [],
        "pitch_summary": {},
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_path


def load_alignment(output_dir: Path) -> dict[str, Any] | None:
    """Load lyrics/alignment.json from output_dir if present.

    Returns None when the file is absent or cannot be parsed.
    Does not require openai-whisper.
    """
    from .paths import lyrics_alignment_path_for_output_dir

    p = lyrics_alignment_path_for_output_dir(output_dir)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def lyric_activity_at(alignment: dict[str, Any], t: float) -> dict[str, Any]:
    """Return lyric activity signals for time t (seconds).

    Returns:
        active_word: word being sung at t, or "" if none
        active_segment: full segment text containing t, or "" if none
        word_confidence: confidence of the active word (0.0 if none)
        word_progress: fractional progress through the current word [0..1]
    """
    for seg in alignment.get("segments", []):
        if seg["start_s"] <= t < seg["end_s"]:
            for w in seg.get("words", []):
                if w["start_s"] <= t < w["end_s"]:
                    dur = max(1e-6, w["end_s"] - w["start_s"])
                    return {
                        "active_word": w["word"],
                        "active_segment": seg.get("text", ""),
                        "word_confidence": float(w.get("confidence", 0.0)),
                        "word_progress": float((t - w["start_s"]) / dur),
                    }
            # Inside segment but between words (gap between aligned tokens).
            return {
                "active_word": "",
                "active_segment": seg.get("text", ""),
                "word_confidence": 0.0,
                "word_progress": 0.0,
            }
    return {
        "active_word": "",
        "active_segment": "",
        "word_confidence": 0.0,
        "word_progress": 0.0,
    }


def lyric_signals_for_timeline(
    alignment: dict[str, Any],
    times_s: list[float],
) -> dict[str, Any]:
    """Compute per-frame lyric signals aligned to an envelope timeline.

    Returns a dict with parallel float arrays (same length as times_s):
        word_active: 1.0 if a word is being sung, 0.0 otherwise
        phrase_active: 1.0 if inside any segment, 0.0 otherwise
        word_confidence: confidence of the active word (0.0 if none)
    """
    word_active: list[float] = []
    phrase_active: list[float] = []
    word_confidence: list[float] = []

    for t in times_s:
        act = lyric_activity_at(alignment, t)
        word_active.append(1.0 if act["active_word"] else 0.0)
        phrase_active.append(1.0 if act["active_segment"] else 0.0)
        word_confidence.append(act["word_confidence"])

    return {
        "word_active": word_active,
        "phrase_active": phrase_active,
        "word_confidence": word_confidence,
        "source": "alignment.json",
    }
