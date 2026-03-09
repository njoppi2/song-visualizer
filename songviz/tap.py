from __future__ import annotations

import datetime
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TapResult:
    taps: list[float] = field(default_factory=list)
    total_words: int = 0
    tapped_count: int = 0
    quit_early: bool = False


class _RawTerminal:
    """Context manager for cbreak-mode terminal input (Unix only).

    Uses cbreak (not raw) so Ctrl+C still raises KeyboardInterrupt.
    """

    def __init__(self) -> None:
        self._old_settings: Any = None
        self._fd: int = -1

    def __enter__(self) -> "_RawTerminal":
        import termios
        import tty

        self._fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._old_settings is not None:
            import termios

            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)

    def key_available(self, timeout: float = 0.0) -> bool:
        import select

        r, _, _ = select.select([sys.stdin], [], [], timeout)
        return bool(r)

    def read_key(self) -> str:
        return sys.stdin.read(1)


def run_tap_session(
    audio_path: str | Path,
    alignment: dict[str, Any],
    *,
    offset: float = 0.0,
    ffplay_path: str | None = None,
) -> TapResult:
    """Play audio via ffplay and record space-bar taps for each word onset.

    Returns a TapResult with timestamps (relative to playback start).
    """
    if ffplay_path is None:
        from .ffmpeg import require_ffplay

        ffplay_path = require_ffplay()

    # Flatten all words.
    words: list[dict[str, Any]] = []
    for seg in alignment.get("segments", []):
        for w in seg.get("words", []):
            words.append(w)

    result = TapResult(total_words=len(words))

    if not words:
        print("No words found in alignment.")
        return result

    print(f"\n  Tap-along session: {len(words)} words")
    print("  Press SPACE at each word onset. Press 'q' to quit early.")
    print("  Press ENTER to start playback...\n")
    sys.stdout.flush()

    # Wait for ENTER.
    input()

    proc = subprocess.Popen(
        [ffplay_path, "-nodisp", "-autoexit", "-loglevel", "quiet", str(audio_path)],
        stdin=subprocess.DEVNULL,
    )

    t0 = time.monotonic()
    word_idx = 0

    try:
        with _RawTerminal() as term:
            while word_idx < len(words):
                if proc.poll() is not None:
                    break

                # Show current word.
                w = words[word_idx]
                elapsed = time.monotonic() - t0
                mins = int(elapsed) // 60
                secs = elapsed % 60
                sys.stdout.write(
                    f"\r  [{mins:02d}:{secs:05.2f}]  "
                    f"({word_idx + 1}/{len(words)})  "
                    f">>> {w['word']} <<<   "
                )
                sys.stdout.flush()

                if term.key_available(timeout=1.0 / 15):
                    key = term.read_key()
                    if key == " ":
                        tap_time = time.monotonic() - t0 + offset
                        result.taps.append(tap_time)
                        result.tapped_count += 1
                        word_idx += 1
                    elif key == "q":
                        result.quit_early = True
                        break
    except KeyboardInterrupt:
        result.quit_early = True
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        print("\n")

    return result


def taps_to_corrections(
    alignment: dict[str, Any],
    taps: list[float],
) -> list[dict[str, Any]]:
    """Map tap timestamps to correction phrases.

    Each tap[i] maps to word[i].start_s. End times derived from next tap
    (minus 50ms gap). Last word in segment uses segment.end_s.
    Tapped words get status: corrected; rest keep status: auto.
    """
    # Flatten words with segment tracking.
    flat: list[tuple[int, int, dict[str, Any]]] = []  # (seg_idx, word_idx_in_seg, word)
    segments = alignment.get("segments", [])
    for si, seg in enumerate(segments):
        for wi, w in enumerate(seg.get("words", [])):
            flat.append((si, wi, w))

    tap_idx = 0
    phrases: list[dict[str, Any]] = []

    for si, seg in enumerate(segments):
        seg_words = seg.get("words", [])
        corr_words: list[dict[str, Any]] = []
        has_correction = False

        for wi, w in enumerate(seg_words):
            orig_start = round(float(w["start_s"]), 3)
            orig_end = round(float(w["end_s"]), 3)

            if tap_idx < len(taps):
                # This word was tapped.
                new_start = round(taps[tap_idx], 3)

                # End time: next tap - 50ms, or segment end for last word in segment.
                if wi == len(seg_words) - 1:
                    new_end = round(float(seg["end_s"]), 3)
                elif tap_idx + 1 < len(taps):
                    new_end = round(taps[tap_idx + 1] - 0.05, 3)
                else:
                    new_end = orig_end

                # Clamp: end >= start + 20ms
                new_end = max(new_end, round(new_start + 0.02, 3))

                corr_words.append({
                    "word": w["word"],
                    "start_s": new_start,
                    "end_s": new_end,
                    "original_start_s": orig_start,
                    "original_end_s": orig_end,
                })
                has_correction = True
                tap_idx += 1
            else:
                # Not tapped — keep original.
                corr_words.append({
                    "word": w["word"],
                    "start_s": orig_start,
                    "end_s": orig_end,
                    "original_start_s": orig_start,
                    "original_end_s": orig_end,
                })

        phrases.append({
            "segment": si,
            "text": seg.get("text", ""),
            "status": "corrected" if has_correction else "auto",
            "words": corr_words,
        })

    return phrases


def write_corrections(
    output_dir: Path,
    alignment: dict[str, Any],
    corrections: list[dict[str, Any]],
    *,
    source: str = "tap_along",
) -> Path:
    """Write corrections.yaml in the phrases-grouped format."""
    import yaml  # type: ignore

    from .paths import lyrics_corrections_path_for_output_dir

    corrections_path = lyrics_corrections_path_for_output_dir(output_dir)

    meta = alignment.get("metadata", {})
    payload = {
        "song_id": meta.get("song_id", ""),
        "source": source,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "phrases": corrections,
    }

    corrections_path.parent.mkdir(parents=True, exist_ok=True)
    corrections_path.write_text(
        yaml.dump(payload, default_flow_style=False, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return corrections_path
