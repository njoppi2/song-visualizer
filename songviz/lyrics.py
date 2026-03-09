from __future__ import annotations

import datetime
import difflib
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

# Matches LRC timestamp lines: [MM:SS.MS] or [MM:SS.MSS]
_LRC_TIMESTAMP_RE = re.compile(r"^\[(\d{1,2}):(\d{2})\.(\d{2,3})\](.*)")
# Matches section markers like [Chorus], [Verse 1] — but not LRC timestamps
_SECTION_MARKER_RE = re.compile(r"^\s*\[[A-Za-z][^\]]*\]\s*$", re.MULTILINE)

_LINE_WINDOW_PADDING_S = 0.6
_REPLACE_TOKEN_SIMILARITY_MIN = 0.72
_DEFAULT_BACKEND_ORDER = ("whisperx", "stable_whisper", "whisper")
_ALIGNMENT_PIPELINE_VERSION = 7

_PHONEME_CLASS_LEAD_IN_S: dict[str, float] = {
    "plosive":     0.055,
    "affricate":   0.045,
    "fricative":   0.025,
    "nasal":       0.020,
    "approximant": 0.020,
    "vowel":       0.005,
    "unknown":     0.040,
}

# 2-char prefixes where the first letter is silent or misleading.
_SILENT_LETTER_EXCEPTIONS: dict[str, str] = {
    "kn": "nasal",       # know, knee    → /n/
    "gn": "nasal",       # gnaw          → /n/
    "mn": "nasal",       # mnemonic      → /n/
    "wr": "approximant", # write, wrong  → /r/
    "wh": "approximant", # what, when    → /w/
    "th": "fricative",   # the, think    → /ð/ or /θ/
    "sh": "fricative",   # shop, she     → /ʃ/
    "ps": "vowel",       # psalm         → silent p
    "pn": "vowel",       # pneumonia     → silent p
    "pt": "vowel",       # pterodactyl   → silent p
    "gh": "vowel",       # ghost         → silent gh
}

_PLOSIVE_INITIALS     = frozenset("bBdDgGkKpPtT")
_FRICATIVE_INITIALS   = frozenset("fFhHsSzZ")
_NASAL_INITIALS       = frozenset("mMnN")
_APPROXIMANT_INITIALS = frozenset("lLrRwWyY")
_VOWEL_INITIALS       = frozenset("aAeEiIoOuU")


def _initial_phoneme_class(word: str) -> str:
    """Return the articulatory class of the word-initial phoneme.

    Pure-text heuristic, no external deps.  Handles ~85% of English words
    correctly; ``_SILENT_LETTER_EXCEPTIONS`` covers common silent-letter cases.

    Returns one of: "plosive", "affricate", "fricative", "nasal",
    "approximant", "vowel", "unknown".
    """
    clean = re.sub(r"^[^A-Za-z]+|[^A-Za-z]+$", "", word).lower()
    if not clean:
        return "unknown"

    prefix2 = clean[:2]
    if prefix2 in _SILENT_LETTER_EXCEPTIONS:
        return _SILENT_LETTER_EXCEPTIONS[prefix2]
    if prefix2 == "ch":
        return "affricate"

    first = clean[0]
    if first == "j":
        return "affricate"
    if first == "c":
        return "fricative" if (len(clean) > 1 and clean[1] in "eiy") else "plosive"
    if first in ("q",):
        return "plosive"
    if first == "x":
        return "fricative"
    if first in _PLOSIVE_INITIALS:     return "plosive"
    if first in _FRICATIVE_INITIALS:   return "fricative"
    if first in _NASAL_INITIALS:       return "nasal"
    if first in _APPROXIMANT_INITIALS: return "approximant"
    if first in _VOWEL_INITIALS:       return "vowel"
    return "unknown"


def _read_audio_metadata(audio_path: Path) -> dict[str, Any]:
    """Read ID3/Vorbis artist and title tags from audio_path using mutagen.

    Returns {"artist": str|None, "title": str|None, "duration_s": float|None}.
    Graceful: returns all-None if mutagen is not installed or file is unreadable.
    """
    result: dict[str, Any] = {"artist": None, "title": None, "duration_s": None}
    try:
        import mutagen  # type: ignore

        f = mutagen.File(str(audio_path))
        if f is None:
            return result

        if hasattr(f, "info") and f.info and hasattr(f.info, "length"):
            result["duration_s"] = float(f.info.length)

        tags = f.tags
        if not tags:
            return result

        def _get_tag(keys: list[str]) -> str | None:
            for k in keys:
                v = tags.get(k)
                if v is not None:
                    # Vorbis comment → list of strings; ID3 frame → str-able object
                    raw = str(v[0]).strip() if isinstance(v, list) else str(v).strip()
                    return raw or None
            return None

        result["artist"] = _get_tag(["TPE1", "artist", "ARTIST"])
        result["title"] = _get_tag(["TIT2", "title", "TITLE"])
    except ImportError:
        pass
    except Exception:
        pass
    return result


def _fetch_lrclib(
    artist: str,
    title: str,
    duration_s: float | None = None,
) -> dict[str, Any] | None:
    """Query LRCLIB API for lyrics.

    Returns the API response dict or None on any failure (network, 404, timeout).
    """
    params: dict[str, str] = {"artist_name": artist, "track_name": title}
    if duration_s is not None:
        params["duration"] = str(int(duration_s))

    url = "https://lrclib.net/api/get?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "songviz/0.0.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode("utf-8"))
            return None
    except Exception:
        return None


def _split_line_into_words(
    text: str,
    line_start: float,
    line_end: float,
) -> list[dict[str, Any]]:
    """Split a lyric line into words with character-proportional timing.

    Each word gets a time slot proportional to its character count.
    A 5% inter-word gap prevents visual run-together in renders.
    confidence is 1.0 (human-curated text); phones is [].
    """
    tokens = text.split()
    if not tokens:
        return []

    total_chars = sum(len(t) for t in tokens)
    if total_chars == 0:
        return []

    total_duration = line_end - line_start
    gap_fraction = 0.05

    words: list[dict[str, Any]] = []
    cursor = line_start
    for i, token in enumerate(tokens):
        char_fraction = len(token) / total_chars
        slot_duration = total_duration * char_fraction
        is_last = i == len(tokens) - 1

        word_end = line_end if is_last else cursor + slot_duration * (1.0 - gap_fraction)

        words.append(
            {
                "word": token,
                "start_s": round(cursor, 6),
                "end_s": round(word_end, 6),
                "confidence": 1.0,
                "phones": [],
            }
        )
        cursor += slot_duration

    return words


def _parse_lrc(lrc_text: str) -> list[dict[str, Any]]:
    """Parse LRC-format text into the alignment.json segment list.

    Each non-empty timestamp line becomes one segment.
    End time = start of the next line; last line gets +5 s.
    Words within each segment are timed proportionally by character count.
    """
    # First pass: collect all timestamped lines (including empty)
    raw_lines: list[tuple[float, str]] = []
    for line in lrc_text.splitlines():
        m = _LRC_TIMESTAMP_RE.match(line.strip())
        if m:
            minutes, seconds, frac, text = m.groups()
            # Normalise 2-digit fractional seconds to milliseconds
            frac_ms = int(frac) * (10 if len(frac) == 2 else 1)
            time_s = int(minutes) * 60.0 + int(seconds) + frac_ms / 1000.0
            raw_lines.append((time_s, text.strip()))

    segments: list[dict[str, Any]] = []
    for i, (start_s, text) in enumerate(raw_lines):
        if not text:
            continue  # Skip empty / instrumental markers
        end_s = raw_lines[i + 1][0] if i + 1 < len(raw_lines) else start_s + 5.0
        words = _split_line_into_words(text, start_s, end_s)
        segments.append(
            {
                "text": text,
                "start_s": start_s,
                "end_s": end_s,
                "words": words,
            }
        )
    return segments


def _normalize_plain_lyrics_for_prompt(plain_text: str) -> str:
    """Strip section markers and normalise whitespace for Whisper's initial_prompt."""
    text = _SECTION_MARKER_RE.sub("", plain_text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _whisper_available() -> bool:
    """Return True if openai-whisper can be imported."""
    try:
        import whisper  # type: ignore  # noqa: F401
        return True
    except ImportError:
        return False


def _whisperx_available() -> bool:
    """Return True if whisperx can be imported."""
    try:
        import whisperx  # type: ignore  # noqa: F401
        return True
    except ImportError:
        return False


def _stable_whisper_available() -> bool:
    """Return True if stable-ts (stable_whisper) can be imported."""
    try:
        import stable_whisper  # type: ignore  # noqa: F401
        return True
    except ImportError:
        return False


def _token_norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _token_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return float(difflib.SequenceMatcher(None, a, b, autojunk=False).ratio())


def _assign_whisper_times_to_lrc_words_with_stats(
    lrc_words: list[str],
    whisper_words: list[dict[str, Any]],
    line_start: float,
    line_end: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Assign Whisper timestamps to LRCLIB words via sequence alignment.

    Normalises both word lists, finds matching positions with difflib,
    then interpolates proportionally for any unmatched LRCLIB words.
    Always returns exactly len(lrc_words) word dicts.
    """
    lrc_norm = [_token_norm(w) for w in lrc_words]
    wh_norm = [_token_norm(str(w["word"])) for w in whisper_words]

    # Map each LRC index → matched Whisper index (or None).
    lrc_to_wh: list[int | None] = [None] * len(lrc_words)
    matcher = difflib.SequenceMatcher(None, lrc_norm, wh_norm, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(min(i2 - i1, j2 - j1)):
                lrc_to_wh[i1 + k] = j1 + k
        elif tag == "replace":
            # Positional mapping only for near-identical token substitutions.
            for k in range(min(i2 - i1, j2 - j1)):
                li = i1 + k
                wi = j1 + k
                sim = _token_similarity(lrc_norm[li], wh_norm[wi])
                if sim >= _REPLACE_TOKEN_SIMILARITY_MIN:
                    lrc_to_wh[li] = wi

    # Whisper coverage boundaries help avoid over-trusting LRCLIB line edges.
    cov_start = float(min((w["start_s"] for w in whisper_words), default=line_start))
    cov_end = float(max((w["end_s"] for w in whisper_words), default=line_end))
    valid_durs = [
        max(0.03, float(w["end_s"]) - float(w["start_s"]))
        for w in whisper_words
        if float(w["end_s"]) > float(w["start_s"])
    ]
    avg_word_dur = (
        float(sum(valid_durs) / len(valid_durs))
        if valid_durs
        else max(0.06, (line_end - line_start) / max(1, len(lrc_words)))
    )

    # Build result list: real entry where matched, None where not.
    result: list[dict[str, Any] | None] = []
    matched_words = 0
    for word, wh_idx in zip(lrc_words, lrc_to_wh):
        if wh_idx is not None:
            wh = whisper_words[wh_idx]
            matched_words += 1
            result.append({
                "word": word,
                "start_s": round(wh["start_s"], 6),
                "end_s": round(wh["end_s"], 6),
                "confidence": float(wh.get("confidence", 1.0)),
                "phones": [],
            })
        else:
            result.append(None)

    # Fill None gaps by proportionally interpolating between anchor timestamps.
    n = len(result)
    i = 0
    while i < n:
        if result[i] is not None:
            i += 1
            continue
        gap_start = i
        while i < n and result[i] is None:
            i += 1
        gap_end = i  # exclusive

        left = result[gap_start - 1] if gap_start > 0 else None
        right = result[gap_end] if gap_end < n else None
        gap_len = gap_end - gap_start
        est_gap_dur = avg_word_dur * max(1, gap_len) * 1.2

        if left is not None and right is not None:
            t_start = float(left["end_s"])
            t_end = float(right["start_s"])
        elif left is not None:
            t_start = float(left["end_s"])
            edge_end = min(line_end, cov_end + _LINE_WINDOW_PADDING_S)
            t_end = min(edge_end, t_start + est_gap_dur)
        elif right is not None:
            t_end = float(right["start_s"])
            edge_start = max(line_start, cov_start - _LINE_WINDOW_PADDING_S)
            t_start = max(edge_start, t_end - est_gap_dur)
        else:
            t_start = max(line_start, cov_start - _LINE_WINDOW_PADDING_S)
            t_end = min(line_end, cov_end + _LINE_WINDOW_PADDING_S)

        if t_end <= t_start + 1e-4:
            t_end = t_start + max(0.06, avg_word_dur * max(1, gap_len))
            if right is not None:
                t_end = min(t_end, float(right["start_s"]))
            else:
                t_end = min(t_end, line_end)
            if t_end <= t_start + 1e-4:
                t_end = t_start + 0.06

        gap_text = " ".join(lrc_words[gap_start:gap_end])
        prop = _split_line_into_words(gap_text, t_start, t_end)
        for j, pw in enumerate(prop):
            result[gap_start + j] = {
                "word": lrc_words[gap_start + j],
                "start_s": pw["start_s"],
                "end_s": pw["end_s"],
                "confidence": 0.65,
                "phones": [],
            }

    interpolated_words = len(lrc_words) - matched_words
    stats = {
        "total_words": len(lrc_words),
        "matched_words": matched_words,
        "interpolated_words": interpolated_words,
    }
    return result, stats  # type: ignore[return-value]


def _assign_whisper_times_to_lrc_words(
    lrc_words: list[str],
    whisper_words: list[dict[str, Any]],
    line_start: float,
    line_end: float,
) -> list[dict[str, Any]]:
    words, _ = _assign_whisper_times_to_lrc_words_with_stats(
        lrc_words,
        whisper_words,
        line_start,
        line_end,
    )
    return words


def _merge_lrc_with_whisper_timing_with_stats(
    lrc_segments: list[dict[str, Any]],
    whisper_segments: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Combine LRC line-level segments with Whisper word-level timestamps.

    For each LRC line, finds all Whisper words whose midpoint falls inside
    the line's time window, then uses sequence alignment to assign real
    audio timestamps to the LRCLIB words.  Falls back to proportional
    timing for lines with no Whisper coverage.
    """
    # Flatten all Whisper words (already sorted by time from Whisper output).
    whisper_words = [w for seg in whisper_segments for w in seg.get("words", [])]

    out: list[dict[str, Any]] = []
    total_words = 0
    matched_words = 0
    interpolated_words = 0
    lines_with_whisper_coverage = 0
    lines_proportional_fallback = 0
    for lrc_seg in lrc_segments:
        line_start = lrc_seg["start_s"]
        line_end = lrc_seg["end_s"]
        lrc_text = lrc_seg["text"]
        lrc_tokens = lrc_text.split()

        if not lrc_tokens:
            continue

        window_start = float(line_start) - _LINE_WINDOW_PADDING_S
        window_end = float(line_end) + _LINE_WINDOW_PADDING_S

        # Use word midpoint for matching, with padded boundaries to reduce
        # line-boundary misses from slight LRC timing drift.
        w_in_line = [
            w for w in whisper_words
            if window_start <= (w["start_s"] + w["end_s"]) / 2 < window_end
        ]

        if w_in_line:
            lines_with_whisper_coverage += 1
            words, line_stats = _assign_whisper_times_to_lrc_words_with_stats(
                lrc_tokens, w_in_line, line_start, line_end
            )
            total_words += int(line_stats["total_words"])
            matched_words += int(line_stats["matched_words"])
            interpolated_words += int(line_stats["interpolated_words"])
        else:
            words = _split_line_into_words(lrc_text, line_start, line_end)
            total_words += len(words)
            interpolated_words += len(words)
            lines_proportional_fallback += 1

        out.append({
            "text": lrc_text,
            "start_s": line_start,
            "end_s": line_end,
            "words": words,
        })

    stats = {
        "lines_total": len(out),
        "lines_with_whisper_coverage": lines_with_whisper_coverage,
        "lines_proportional_fallback": lines_proportional_fallback,
        "total_words": total_words,
        "matched_words": matched_words,
        "interpolated_words": interpolated_words,
        "matched_word_ratio": (
            float(matched_words) / float(total_words)
            if total_words > 0
            else 0.0
        ),
    }
    return out, stats


def _merge_lrc_with_whisper_timing(
    lrc_segments: list[dict[str, Any]],
    whisper_segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out, _ = _merge_lrc_with_whisper_timing_with_stats(lrc_segments, whisper_segments)
    return out


def _build_forced_align_segments_with_stats(
    lrc_segments: list[dict[str, Any]],
    fa_segments: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Map forced-alignment word timestamps back onto LRCLIB segment structure.

    Since FA was run on the exact LRCLIB text, FA words should correspond 1:1
    to LRCLIB words.  We match sequentially by index with a similarity check
    to guard against any tokenization differences.

    Returns (segments, stats) where stats includes matched_word_ratio.
    """
    _FA_TOKEN_SIM_MIN = 0.6

    # Flatten all FA words into a single ordered list.
    fa_words: list[dict[str, Any]] = []
    for seg in fa_segments:
        fa_words.extend(seg.get("words", []))

    fa_cursor = 0
    out: list[dict[str, Any]] = []
    total_words = 0
    matched_words = 0

    for lrc_seg in lrc_segments:
        lrc_text = lrc_seg["text"]
        lrc_tokens = lrc_text.split()
        line_start = lrc_seg["start_s"]
        line_end = lrc_seg["end_s"]

        if not lrc_tokens:
            continue

        words: list[dict[str, Any]] = []
        unmatched_indices: list[int] = []

        for i, lrc_word in enumerate(lrc_tokens):
            total_words += 1
            if fa_cursor < len(fa_words):
                fa_w = fa_words[fa_cursor]
                sim = _token_similarity(
                    _token_norm(lrc_word),
                    _token_norm(str(fa_w.get("word", ""))),
                )
                if sim >= _FA_TOKEN_SIM_MIN:
                    words.append({
                        "word": lrc_word,
                        "start_s": round(float(fa_w["start_s"]), 6),
                        "end_s": round(float(fa_w["end_s"]), 6),
                        "confidence": float(fa_w.get("confidence", 0.9)),
                        "phones": [],
                    })
                    matched_words += 1
                    fa_cursor += 1
                    continue

            # Unmatched — record index for fallback
            words.append(None)  # type: ignore[arg-type]
            unmatched_indices.append(i)

        # Fill unmatched words with proportional timing
        if unmatched_indices:
            # Group consecutive unmatched indices into spans
            spans: list[tuple[int, int]] = []
            span_start = unmatched_indices[0]
            prev = span_start
            for idx in unmatched_indices[1:]:
                if idx == prev + 1:
                    prev = idx
                else:
                    spans.append((span_start, prev + 1))
                    span_start = idx
                    prev = idx
            spans.append((span_start, prev + 1))

            for sp_start, sp_end in spans:
                # Determine time bounds from neighboring matched words
                left_end = words[sp_start - 1]["end_s"] if sp_start > 0 and words[sp_start - 1] is not None else line_start
                right_start = words[sp_end]["start_s"] if sp_end < len(words) and words[sp_end] is not None else line_end
                gap_text = " ".join(lrc_tokens[sp_start:sp_end])
                prop = _split_line_into_words(gap_text, left_end, right_start)
                for j, pw in enumerate(prop):
                    words[sp_start + j] = {
                        "word": lrc_tokens[sp_start + j],
                        "start_s": pw["start_s"],
                        "end_s": pw["end_s"],
                        "confidence": 0.65,
                        "phones": [],
                    }

        out.append({
            "text": lrc_text,
            "start_s": line_start,
            "end_s": line_end,
            "words": words,
        })

    stats = {
        "total_words": total_words,
        "matched_words": matched_words,
        "interpolated_words": total_words - matched_words,
        "matched_word_ratio": (
            float(matched_words) / float(total_words) if total_words > 0 else 0.0
        ),
    }
    return out, stats


def _whisper_align(
    audio_path: Path,
    *,
    language: str | None = None,
    model_name: str = "base",
    initial_prompt: str | None = None,
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
    if initial_prompt:
        decode_options["initial_prompt"] = initial_prompt

    result = model.transcribe(str(audio_path), **decode_options)
    return result.get("segments", [])


def _whisperx_align(
    audio_path: Path,
    *,
    language: str | None = None,
    model_name: str = "base",
    initial_prompt: str | None = None,
) -> list[dict[str, Any]]:
    """Use WhisperX (forced alignment) to produce timestamped segments."""
    try:
        import torch  # type: ignore
        import whisperx  # type: ignore
    except ImportError as e:
        raise ImportError(
            "whisperx is required for backend='whisperx'. "
            "Install it with: pip install whisperx"
        ) from e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    load_kwargs: dict[str, Any] = {}
    if language:
        load_kwargs["language"] = language
    model = whisperx.load_model(model_name, device=device, compute_type=compute_type, **load_kwargs)

    transcribe_kwargs: dict[str, Any] = {}
    if initial_prompt:
        transcribe_kwargs["initial_prompt"] = initial_prompt
    try:
        result = model.transcribe(str(audio_path), batch_size=8, **transcribe_kwargs)
    except TypeError:
        # Backward-compatible fallback for whisperx versions without initial_prompt.
        result = model.transcribe(str(audio_path), batch_size=8)

    lang_code = str(result.get("language") or language or "en")
    align_model, metadata = whisperx.load_align_model(language_code=lang_code, device=device)
    aligned = whisperx.align(
        result.get("segments", []),
        align_model,
        metadata,
        str(audio_path),
        device,
        return_char_alignments=False,
    )
    return aligned.get("segments", [])


def _stable_whisper_align(
    audio_path: Path,
    *,
    language: str | None = None,
    model_name: str = "small",
    initial_prompt: str | None = None,
) -> list[dict[str, Any]]:
    """Use stable-ts for word timestamps via DTW-based refinement on the mel spectrogram.

    stable-ts wraps openai-whisper and applies Dynamic Time Warping post-processing to
    produce significantly more accurate word boundaries than vanilla Whisper attention weights.
    """
    try:
        import stable_whisper  # type: ignore
    except ImportError as e:
        raise ImportError(
            "stable-ts is required for backend='stable_whisper'. "
            "Install it with: pip install stable-ts"
        ) from e

    model = stable_whisper.load_model(model_name)
    kwargs: dict[str, Any] = {}
    if language:
        kwargs["language"] = language
    if initial_prompt:
        kwargs["initial_prompt"] = initial_prompt
    result = model.transcribe(str(audio_path), word_timestamps=True, **kwargs)
    # WhisperResult.to_dict() returns standard Whisper segment format with
    # per-word {"word", "start", "end", "probability"} entries.
    return result.to_dict().get("segments", [])


def _stable_whisper_forced_align(
    audio_path: Path,
    text: str,
    *,
    language: str | None = None,
    model_name: str = "small",
) -> list[dict[str, Any]]:
    """Use stable-ts forced alignment to align known text to audio.

    Instead of transcribing (which generates its own text), this aligns the
    *provided* text directly to the audio, producing word timestamps that
    correspond 1:1 to the input words.
    """
    try:
        import stable_whisper  # type: ignore
    except ImportError as e:
        raise ImportError(
            "stable-ts is required for forced alignment. "
            "Install it with: pip install stable-ts"
        ) from e

    model = stable_whisper.load_model(model_name)
    kwargs: dict[str, Any] = {}
    if language:
        kwargs["language"] = language
    result = model.align(str(audio_path), text, **kwargs)
    if result is None:
        raise RuntimeError("stable-ts align() returned None")
    return result.to_dict().get("segments", [])


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
                    "confidence": float(w.get("probability", w.get("score", 0.0))),
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


def _resolve_backend_order(align_backend: str) -> list[str]:
    backend = str(align_backend).strip().lower()
    if backend not in ("auto", "whisper", "whisperx", "stable_whisper"):
        raise ValueError(
            f"Unsupported backend: {align_backend!r} "
            "(expected: auto|whisper|whisperx|stable_whisper)"
        )
    if backend == "whisper":
        return ["whisper"]
    if backend == "whisperx":
        return ["whisperx"]
    if backend == "stable_whisper":
        return ["stable_whisper"]

    # Auto: best available backend first.
    # Priority: whisperx (forced alignment, most accurate) > stable_whisper
    # (DTW refinement, much better than vanilla Whisper) > whisper (fallback).
    order: list[str] = []
    if _whisperx_available():
        order.append("whisperx")
    if _stable_whisper_available():
        order.append("stable_whisper")
    if _whisper_available():
        order.append("whisper")
    if not order:
        # Optimistic fallback: still try whisper and surface a clean ImportError.
        order.append("whisper")
    return order


def _align_with_backend(
    backend: str,
    *,
    audio_path: Path,
    language: str,
    model_name: str,
    initial_prompt: str | None,
) -> list[dict[str, Any]]:
    if backend == "whisper":
        raw = _whisper_align(
            audio_path,
            language=language,
            model_name=model_name,
            initial_prompt=initial_prompt,
        )
        return _normalize_whisper_segments(raw)
    if backend == "whisperx":
        raw = _whisperx_align(
            audio_path,
            language=language,
            model_name=model_name,
            initial_prompt=initial_prompt,
        )
        return _normalize_whisper_segments(raw)
    if backend == "stable_whisper":
        raw = _stable_whisper_align(
            audio_path,
            language=language,
            model_name=model_name,
            initial_prompt=initial_prompt,
        )
        return _normalize_whisper_segments(raw)
    raise ValueError(f"Unknown backend: {backend!r}")


def _forced_align_with_backend(
    backend: str,
    *,
    audio_path: Path,
    text: str,
    language: str,
    model_name: str,
) -> list[dict[str, Any]]:
    """Run forced alignment with the given backend and known text."""
    if backend == "stable_whisper":
        raw = _stable_whisper_forced_align(
            audio_path,
            text,
            language=language,
            model_name=model_name,
        )
        return _normalize_whisper_segments(raw)
    raise ValueError(
        f"Forced alignment not supported for backend {backend!r}. "
        "Only 'stable_whisper' is supported."
    )


def _shift_segments_time(segments: list[dict[str, Any]], offset_s: float) -> None:
    if abs(offset_s) < 1e-9:
        return
    for seg in segments:
        seg_start = round(max(0.0, float(seg.get("start_s", 0.0)) + offset_s), 6)
        seg_end = round(max(seg_start, float(seg.get("end_s", seg_start)) + offset_s), 6)
        seg["start_s"] = seg_start
        seg["end_s"] = seg_end
        words = seg.get("words", [])
        if not isinstance(words, list):
            continue
        for w in words:
            w_start = round(max(0.0, float(w.get("start_s", 0.0)) + offset_s), 6)
            w_end = round(max(w_start, float(w.get("end_s", w_start)) + offset_s), 6)
            w["start_s"] = w_start
            w["end_s"] = w_end


def _snap_words_to_vocal_onset_rms(
    segments: list[dict[str, Any]],
    audio_path: Path,
    *,
    frame_hz: int = 100,
    silence_ratio: float = 0.10,
    onset_ratio: float = 0.15,
    lead_in_s: float = 0.05,
    min_word_dur_s: float = 0.05,
) -> dict[str, Any]:
    """Snap word start times forward to actual vocal onset when they begin in silence.

    After forced alignment (or any alignment), some words may be placed in silence
    gaps before the singer actually starts voicing.  This detects those cases using
    the RMS envelope and moves word starts to where vocal energy actually begins.

    Returns stats dict with count of snapped words.

    .. note:: Legacy v5 implementation kept for rollback.  See
       ``_snap_words_to_vocal_onset`` for the v6 onset_detect approach.
    """
    try:
        import librosa  # type: ignore
        import numpy as np
    except ImportError:
        return {"snapped_words": 0, "error": "librosa not available"}

    try:
        y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    except Exception:
        return {"snapped_words": 0, "error": "audio load failed"}

    if y.size < 1024:
        return {"snapped_words": 0, "error": "audio too short"}

    hop = max(1, int(sr / frame_hz))
    frame_len = max(512, hop * 4)
    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop, center=True)[0]

    def _t_to_frame(t: float) -> int:
        return max(0, min(len(rms) - 1, int(round(t * frame_hz))))

    snapped = 0
    details: list[dict[str, Any]] = []
    for seg_idx, seg in enumerate(segments):
        words = seg.get("words", [])
        if not isinstance(words, list):
            continue
        for w in words:
            ws = float(w.get("start_s", 0.0))
            we = float(w.get("end_s", ws))
            if we - ws < min_word_dur_s:
                continue

            f_start = _t_to_frame(ws)
            f_end = _t_to_frame(we)
            if f_start >= f_end:
                continue

            word_rms = rms[f_start:f_end]
            word_max = float(np.max(word_rms))
            if word_max < 1e-6:
                continue  # entirely silent word, don't touch

            start_rms = float(rms[f_start])
            if start_rms > word_max * silence_ratio:
                continue  # start is not in silence, leave it

            # Find onset: first frame where RMS exceeds onset_ratio of word max.
            threshold = word_max * onset_ratio
            onset_frame = f_start
            for f in range(f_start, f_end):
                if float(rms[f]) >= threshold:
                    onset_frame = f
                    break

            # Apply lead-in so we don't clip consonant onsets.
            new_start = max(ws, (onset_frame / frame_hz) - lead_in_s)
            if new_start - ws < 0.03:
                continue  # negligible shift, skip
            if we - new_start < min_word_dur_s:
                continue  # would make word too short

            w["start_s"] = round(new_start, 6)
            snapped += 1
            details.append({
                "word": w.get("word", ""),
                "seg_idx": seg_idx,
                "original_start_s": ws,
                "snapped_start_s": round(new_start, 6),
                "delta_s": round(new_start - ws, 6),
            })

    return {"snapped_words": snapped, "snap_method": "rms", "details": details}


def _snap_words_to_vocal_onset(
    segments: list[dict[str, Any]],
    audio_path: Path,
    *,
    hop_length: int = 256,
    max_forward_s: float = 0.20,
    lead_in_s: float = 0.04,
    min_word_dur_s: float = 0.05,
    min_shift_s: float = 0.02,
    onset_wait: int = 10,
) -> dict[str, Any]:
    """Snap word start times forward to the nearest spectral onset.

    Uses ``librosa.onset.onset_detect`` (spectral flux) to find acoustic onsets
    even in continuous singing where there is no silence gap between words.
    For each word, binary-searches for the first onset in
    ``[word_start, word_start + max_forward_s]`` and nudges the word start to
    ``max(word_start, onset - lead_in_s)`` so it only moves *forward*, never
    earlier.

    Returns stats dict compatible with the v5 ``_snap_words_to_vocal_onset_rms``.
    """
    try:
        import librosa  # type: ignore
        import numpy as np
    except ImportError:
        return {"snapped_words": 0, "snap_method": "onset_detect", "error": "librosa not available"}

    try:
        y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    except Exception:
        return {"snapped_words": 0, "snap_method": "onset_detect", "error": "audio load failed"}

    if y.size < 1024:
        return {"snapped_words": 0, "snap_method": "onset_detect", "error": "audio too short"}

    onsets = librosa.onset.onset_detect(
        y=y, sr=sr, hop_length=hop_length, backtrack=False, units="time", wait=onset_wait,
    )

    snapped = 0
    details: list[dict[str, Any]] = []
    for seg_idx, seg in enumerate(segments):
        words = seg.get("words", [])
        if not isinstance(words, list):
            continue
        for w in words:
            ws = float(w.get("start_s", 0.0))
            we = float(w.get("end_s", ws))
            if we - ws < min_word_dur_s:
                continue

            # Per-word phoneme-aware lead-in.
            ph_class = _initial_phoneme_class(w.get("word", ""))
            word_lead_in = _PHONEME_CLASS_LEAD_IN_S.get(ph_class, lead_in_s)
            if isinstance(w.get("phones"), list):
                w["phones"] = [{"class": ph_class, "source": "text_heuristic"}]

            # Binary-search for first onset in [ws, ws + max_forward_s]
            idx = int(np.searchsorted(onsets, ws))
            if idx >= len(onsets):
                continue
            onset_t = float(onsets[idx])
            if onset_t > ws + max_forward_s:
                continue  # no onset within the look-ahead window

            new_start = max(ws, onset_t - word_lead_in)
            shift = new_start - ws
            if shift < min_shift_s:
                continue  # negligible shift
            if we - new_start < min_word_dur_s:
                continue  # would make word too short

            w["start_s"] = round(new_start, 6)
            snapped += 1
            details.append({
                "word": w.get("word", ""),
                "seg_idx": seg_idx,
                "original_start_s": ws,
                "snapped_start_s": round(new_start, 6),
                "delta_s": round(shift, 6),
                "phoneme_class": ph_class,
                "lead_in_s": word_lead_in,
            })

    return {
        "snapped_words": snapped,
        "snap_method": "onset_detect",
        "total_onsets": len(onsets),
        "details": details,
    }


def _should_apply_calibration(offset_s: float, improvement: float, best_corr: float) -> bool:
    """Decide whether to apply a calibration offset.

    The standard threshold (improvement >= 0.01) works for most songs, but
    singing produces inherently flat correlation curves — a binary word-activity
    signal correlates poorly with RMS energy.  For large offsets (>= 100 ms) a
    relaxed but still measurable improvement (>= 0.005) is required to filter
    out noise-level correlation changes while still catching real timing shifts.
    """
    return (
        abs(offset_s) >= 0.03
        and best_corr >= 0.08
        and (
            improvement >= 0.01                              # standard case
            or (abs(offset_s) >= 0.10 and improvement >= 0.005)  # large offset, require measurable gain
        )
    )


def _estimate_global_offset_s(
    *,
    audio_path: Path,
    segments: list[dict[str, Any]],
    max_offset_s: float = 0.8,
    frame_hz: int = 100,
) -> dict[str, Any] | None:
    """Estimate a global timing shift by correlating word activity vs vocal energy."""
    words: list[dict[str, Any]] = []
    for seg in segments:
        ws = seg.get("words", [])
        if isinstance(ws, list):
            words.extend(w for w in ws if isinstance(w, dict))
    if not words:
        return None

    try:
        import librosa  # type: ignore
        import numpy as np
    except Exception:
        return None

    try:
        if audio_path.stat().st_size < 4096:
            return None
    except Exception:
        return None

    try:
        y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    except Exception:
        return None
    if y.size < 1024:
        return None

    hop = max(1, int(sr / frame_hz))
    frame_len = max(1024, hop * 4)
    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop, center=True)[0]
    if rms.size < 10:
        return None

    lo = float(np.percentile(rms, 5))
    hi = float(np.percentile(rms, 95))
    den = max(1e-9, hi - lo)
    audio_sig = np.clip((rms - lo) / den, 0.0, 1.0).astype(np.float32)
    gate = float(np.percentile(audio_sig, 60))
    audio_sig = np.clip((audio_sig - gate) / max(1e-9, 1.0 - gate), 0.0, 1.0)

    word_sig = np.zeros_like(audio_sig, dtype=np.float32)
    n = int(word_sig.shape[0])
    for w in words:
        ws = int(round(float(w.get("start_s", 0.0)) * frame_hz))
        we = int(round(float(w.get("end_s", ws / frame_hz)) * frame_hz))
        ws = max(0, min(n - 1, ws))
        we = max(ws + 1, min(n, we))
        word_sig[ws:we] = 1.0

    if not float(word_sig.sum()) > 0.0:
        return None

    k = 9
    kernel = np.ones((k,), dtype=np.float32) / float(k)
    audio_s = np.convolve(audio_sig, kernel, mode="same")
    word_s = np.convolve(word_sig, kernel, mode="same")
    audio_z = (audio_s - float(audio_s.mean())) / max(1e-9, float(audio_s.std()))
    word_z = (word_s - float(word_s.mean())) / max(1e-9, float(word_s.std()))

    def _shift(sig: Any, lag: int) -> Any:
        out = np.zeros_like(sig)
        if lag > 0:
            out[lag:] = sig[:-lag]
            return out
        if lag < 0:
            out[:lag] = sig[-lag:]
            return out
        return sig.copy()

    def _corr(lag: int) -> float:
        sw = _shift(word_z, lag)
        return float(np.dot(audio_z, sw) / max(1, audio_z.size))

    max_lag = max(1, int(round(max_offset_s * frame_hz)))
    corr_0 = _corr(0)
    best_lag = 0
    best_corr = corr_0
    for lag in range(-max_lag, max_lag + 1):
        c = _corr(lag)
        if c > best_corr:
            best_corr = c
            best_lag = lag

    offset_s = float(best_lag) / float(frame_hz)
    improvement = best_corr - corr_0
    should_apply = _should_apply_calibration(offset_s, improvement, best_corr)
    return {
        "estimated_offset_s": round(offset_s, 6),
        "corr_at_zero": round(corr_0, 6),
        "corr_at_best": round(best_corr, 6),
        "corr_improvement": round(improvement, 6),
        "applied": bool(should_apply),
    }


def _cached_alignment_is_compatible(
    *,
    alignment_path: Path,
    language: str,
    model_name: str,
    align_backend: str,
    auto_calibrate: bool,
) -> bool:
    try:
        payload = json.loads(alignment_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    meta = payload.get("metadata", {})
    if not isinstance(meta, dict):
        return False

    try:
        version = int(meta.get("pipeline_version", 0))
    except Exception:
        version = 0
    if version < _ALIGNMENT_PIPELINE_VERSION:
        return False

    if str(meta.get("language", "")) != str(language):
        return False
    if str(meta.get("whisper_model", "")) != str(model_name):
        return False
    if str(meta.get("backend_requested", "auto")) != str(align_backend):
        return False
    if bool(meta.get("auto_calibrate", True)) != bool(auto_calibrate):
        return False
    return True


def align_lyrics(
    audio_path: str | Path,
    *,
    song_id: str,
    output_dir: Path,
    vocals_stem: Path | None = None,
    language: str = "en",
    model_name: str = "small",
    align_backend: str = "auto",
    auto_calibrate: bool = True,
    force: bool = False,
    artist: str | None = None,
    title: str | None = None,
) -> Path:
    """Run the lyrics alignment pipeline and write lyrics/alignment.json.

    Fallback chain:
    1. LRCLIB synced + backend available → backend timing + LRCLIB text
    2. LRCLIB synced, no backend         → proportional word timing (lrclib_synced)
    3. LRCLIB plain text + backend       → backend with initial_prompt
    4. No LRCLIB match                   → pure backend transcription

    Returns the path to the written alignment.json.

    Requires: pip install -e '.[lyrics]'
    """
    from .paths import lyrics_alignment_path_for_output_dir

    out_path = lyrics_alignment_path_for_output_dir(output_dir)
    if out_path.exists() and not force:
        if _cached_alignment_is_compatible(
            alignment_path=out_path,
            language=language,
            model_name=model_name,
            align_backend=align_backend,
            auto_calibrate=auto_calibrate,
        ):
            return out_path

    # Choose alignment audio source; prefer isolated vocals when available.
    if vocals_stem is not None and Path(vocals_stem).exists():
        align_audio = Path(vocals_stem)
        audio_source = "vocals_stem"
    else:
        align_audio = Path(audio_path)
        audio_source = "mix"

    # Resolve artist / title: CLI overrides take priority, then ID3/Vorbis tags.
    meta = _read_audio_metadata(Path(audio_path))
    if artist is None:
        artist = meta.get("artist")
    if title is None:
        title = meta.get("title")
    duration_s: float | None = meta.get("duration_s")

    # --- LRCLIB lookup ---
    lrclib_data: dict[str, Any] | None = None
    if artist and title:
        lrclib_data = _fetch_lrclib(artist, title, duration_s)

    backend_order = _resolve_backend_order(align_backend)
    backend_used: str | None = None
    alignment_tool = "whisper"
    initial_prompt: str | None = None
    segments: list[dict[str, Any]] | None = None
    merge_stats: dict[str, Any] | None = None
    quality_flags: list[str] = []
    backend_errors: list[str] = []
    # Tracks calibration applied pre-merge (lrclib+backend path) vs post-merge (pure Whisper).
    pre_calibration: dict[str, Any] | None = None
    _lrclib_backend_used = False

    if lrclib_data:
        synced = (lrclib_data.get("syncedLyrics") or "").strip()
        plain = (lrclib_data.get("plainLyrics") or "").strip()

        if synced:
            lrc_parsed = _parse_lrc(synced)
            if lrc_parsed:
                prompt = _normalize_plain_lyrics_for_prompt(plain) if plain else None

                # --- Try forced alignment first (preferred) ---
                # FA aligns the known LRCLIB text directly to audio, producing
                # 1:1 word timestamps without lossy difflib matching.
                if backend_order:
                    fa_backends = [b for b in backend_order if b == "stable_whisper"]
                    lrc_text = "\n".join(seg["text"] for seg in lrc_parsed)
                    for backend in fa_backends:
                        try:
                            fa_segs = _forced_align_with_backend(
                                backend,
                                audio_path=align_audio,
                                text=lrc_text,
                                language=language,
                                model_name=model_name,
                            )
                            backend_used = backend

                            if auto_calibrate:
                                pre_calibration = _estimate_global_offset_s(
                                    audio_path=align_audio, segments=fa_segs
                                )
                                if pre_calibration and pre_calibration.get("applied"):
                                    _shift_segments_time(
                                        fa_segs,
                                        float(pre_calibration["estimated_offset_s"]),
                                    )

                            segments, merge_stats = _build_forced_align_segments_with_stats(
                                lrc_parsed, fa_segs,
                            )
                            alignment_tool = f"lrclib+{backend}_forced"
                            _lrclib_backend_used = True
                            break
                        except Exception as e:
                            backend_errors.append(f"{backend}_forced:{type(e).__name__}")

                # --- Fallback: existing transcribe+merge path ---
                if segments is None and backend_order:
                    for backend in backend_order:
                        try:
                            backend_segs = _align_with_backend(
                                backend,
                                audio_path=align_audio,
                                language=language,
                                model_name=model_name,
                                initial_prompt=prompt,
                            )
                            backend_used = backend

                            # Pre-merge calibration: adjust backend timestamps against the
                            # actual audio BEFORE merging with LRCLIB. This preserves
                            # LRCLIB segment boundaries (which are accurate) and ensures
                            # the merge window finds words at the right places.
                            if auto_calibrate:
                                pre_calibration = _estimate_global_offset_s(
                                    audio_path=align_audio, segments=backend_segs
                                )
                                if pre_calibration and pre_calibration.get("applied"):
                                    _shift_segments_time(
                                        backend_segs,
                                        float(pre_calibration["estimated_offset_s"]),
                                    )

                            segments, merge_stats = _merge_lrc_with_whisper_timing_with_stats(
                                lrc_parsed,
                                backend_segs,
                            )
                            if backend == "whisper":
                                alignment_tool = "lrclib+whisper_timing"
                            else:
                                alignment_tool = f"lrclib+{backend}_timing"
                            _lrclib_backend_used = True
                            break
                        except Exception as e:
                            backend_errors.append(f"{backend}:{type(e).__name__}")

                if segments is None:
                    # Backend not installed → proportional word timing only.
                    segments = lrc_parsed
                    alignment_tool = "lrclib_synced"

        if segments is None and plain:
            initial_prompt = _normalize_plain_lyrics_for_prompt(plain)
            alignment_tool = "whisper+lrclib_prompt"

    # --- Backend fallback (pure or with LRCLIB plain prompt) ---
    if segments is None:
        if not backend_order:
            raise ImportError(
                "No lyrics alignment backend is available. "
                "Install openai-whisper (or whisperx) to align lyrics."
            )
        for backend in backend_order:
            try:
                segments = _align_with_backend(
                    backend,
                    audio_path=align_audio,
                    language=language,
                    model_name=model_name,
                    initial_prompt=initial_prompt,
                )
                backend_used = backend
                if backend == "whisper":
                    if initial_prompt:
                        alignment_tool = "whisper+lrclib_prompt"
                    else:
                        alignment_tool = "whisper"
                else:
                    if initial_prompt:
                        alignment_tool = f"{backend}+lrclib_prompt"
                    else:
                        alignment_tool = backend
                break
            except Exception as e:
                backend_errors.append(f"{backend}:{type(e).__name__}")

    if segments is None:
        details = ", ".join(backend_errors) if backend_errors else "no backend errors captured"
        raise RuntimeError(f"Lyrics alignment failed for all backends ({details})")

    calibration: dict[str, Any] | None = None
    if auto_calibrate:
        if _lrclib_backend_used:
            # LRCLIB segment boundaries are the timing reference; calibration was already
            # applied pre-merge to the raw backend output. Don't shift segments again.
            calibration = pre_calibration
        else:
            # Pure-Whisper paths: all timestamps come from the backend → shift everything.
            calibration = _estimate_global_offset_s(audio_path=align_audio, segments=segments)
            if calibration and calibration.get("applied"):
                _shift_segments_time(segments, float(calibration["estimated_offset_s"]))

    # Snap word starts to actual vocal onsets (fixes words placed in silence gaps).
    onset_snap_stats: dict[str, Any] | None = None
    if alignment_tool != "lrclib_synced":
        onset_snap_stats = _snap_words_to_vocal_onset(segments, align_audio)

    if merge_stats is not None:
        if float(merge_stats.get("matched_word_ratio", 0.0)) < 0.65:
            quality_flags.append("low_word_match_ratio")
        lines_total = int(merge_stats.get("lines_total", 0))
        lines_fallback = int(merge_stats.get("lines_proportional_fallback", 0))
        if lines_total > 0 and (float(lines_fallback) / float(lines_total)) > 0.35:
            quality_flags.append("many_lines_without_whisper_coverage")

    if calibration is not None:
        cal_offset = abs(float(calibration.get("estimated_offset_s", 0.0)))
        if cal_offset >= 0.25:
            quality_flags.append("large_global_offset_detected")
        if not calibration.get("applied"):
            if float(calibration.get("corr_improvement", 0.0)) >= 0.02:
                quality_flags.append("offset_not_applied_low_confidence")
            elif cal_offset >= 0.05:
                quality_flags.append("offset_detected_but_not_applied")

    if backend_errors:
        quality_flags.append("backend_fallback_used")

    audio_source_meta = "lrclib" if alignment_tool == "lrclib_synced" else audio_source

    payload: dict[str, Any] = {
        "metadata": {
            "song_id": song_id,
            "pipeline_version": _ALIGNMENT_PIPELINE_VERSION,
            "language": language,
            "alignment_tool": alignment_tool,
            "whisper_model": model_name,
            "backend_requested": align_backend,
            "backend_used": backend_used or "none",
            "audio_source": audio_source_meta,
            "auto_calibrate": bool(auto_calibrate),
            "auto_offset_s": (
                float(calibration["estimated_offset_s"])
                if calibration is not None
                else 0.0
            ),
            "calibration_applied": bool(calibration.get("applied")) if calibration else False,
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
        "segments": segments,
        "quality_flags": sorted(set(quality_flags)),
        "alignment_stats": {
            "merge": merge_stats or {},
            "calibration": calibration or {},
            "onset_snap": onset_snap_stats or {},
            "backend_errors": backend_errors,
        },
        "pitch_summary": {},
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Auto-reapply corrections if corrections.yaml exists.
    from .paths import lyrics_corrections_path_for_output_dir

    if lyrics_corrections_path_for_output_dir(output_dir).exists():
        try:
            apply_corrections(output_dir)
        except Exception:
            pass  # Don't fail alignment because corrections couldn't be applied

    return out_path


def generate_corrections_template(output_dir: Path) -> Path:
    """Generate corrections.yaml from alignment.json with all phrases status: auto.

    If corrections.yaml already exists, merge: preserve status and edited
    timestamps for segments matching by index+text.
    """
    import yaml  # type: ignore

    from .paths import lyrics_alignment_path_for_output_dir, lyrics_corrections_path_for_output_dir

    alignment_path = lyrics_alignment_path_for_output_dir(output_dir)
    corrections_path = lyrics_corrections_path_for_output_dir(output_dir)

    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))

    # Build new phrases from alignment.
    phrases: list[dict[str, Any]] = []
    for i, seg in enumerate(alignment.get("segments", [])):
        words: list[dict[str, Any]] = []
        for w in seg.get("words", []):
            words.append({
                "word": w["word"],
                "start_s": round(float(w["start_s"]), 3),
                "end_s": round(float(w["end_s"]), 3),
                "original_start_s": round(float(w["start_s"]), 3),
                "original_end_s": round(float(w["end_s"]), 3),
            })
        phrases.append({
            "segment": i,
            "text": seg.get("text", ""),
            "status": "auto",
            "words": words,
        })

    # If corrections.yaml already exists, merge: preserve edits.
    if corrections_path.exists():
        try:
            existing = yaml.safe_load(corrections_path.read_text(encoding="utf-8"))
            existing_phrases = existing.get("phrases", []) if existing else []
            existing_by_key: dict[tuple[int, str], dict[str, Any]] = {}
            for ep in existing_phrases:
                key = (ep.get("segment", -1), ep.get("text", ""))
                existing_by_key[key] = ep
            for phrase in phrases:
                key = (phrase["segment"], phrase["text"])
                if key in existing_by_key:
                    old = existing_by_key[key]
                    if old.get("status", "auto") != "auto":
                        phrase["status"] = old["status"]
                        phrase["words"] = old["words"]
        except Exception:
            pass

    meta = alignment.get("metadata", {})
    header = {
        "song_id": meta.get("song_id", ""),
        "source": "template",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "phrases": phrases,
    }

    corrections_path.parent.mkdir(parents=True, exist_ok=True)
    corrections_path.write_text(yaml.dump(header, default_flow_style=False, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return corrections_path


def load_corrections(output_dir: Path) -> dict[str, Any] | None:
    """Read and parse corrections.yaml if it exists."""
    import yaml  # type: ignore

    from .paths import lyrics_corrections_path_for_output_dir

    corrections_path = lyrics_corrections_path_for_output_dir(output_dir)
    if not corrections_path.exists():
        return None
    try:
        return yaml.safe_load(corrections_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def apply_corrections(output_dir: Path) -> dict[str, Any]:
    """Apply corrections.yaml to alignment.json.

    For status: corrected → replace word start_s/end_s.
    For status: verified → tag as ground truth (no timestamp change).
    Validates segment text matches before applying (skip mismatches with warning).

    Returns: {applied, verified, skipped, mismatched}
    """
    from .paths import lyrics_alignment_path_for_output_dir, lyrics_corrections_path_for_output_dir

    alignment_path = lyrics_alignment_path_for_output_dir(output_dir)
    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    corrections = load_corrections(output_dir)

    stats: dict[str, int] = {"applied": 0, "verified": 0, "skipped": 0, "mismatched": 0}
    if corrections is None:
        return stats

    segments = alignment.get("segments", [])
    for phrase in corrections.get("phrases", []):
        seg_idx = phrase.get("segment", -1)
        status = phrase.get("status", "auto")
        if status == "auto":
            stats["skipped"] += 1
            continue

        if seg_idx < 0 or seg_idx >= len(segments):
            stats["mismatched"] += 1
            continue

        seg = segments[seg_idx]
        if seg.get("text", "").strip() != phrase.get("text", "").strip():
            stats["mismatched"] += 1
            continue

        seg_words = seg.get("words", [])
        corr_words = phrase.get("words", [])

        if status == "corrected":
            for j, cw in enumerate(corr_words):
                if j < len(seg_words):
                    seg_words[j]["start_s"] = round(float(cw["start_s"]), 3)
                    seg_words[j]["end_s"] = round(float(cw["end_s"]), 3)
            stats["applied"] += 1
        elif status == "verified":
            stats["verified"] += 1

    # Update alignment metadata.
    alignment.setdefault("metadata", {})
    alignment["metadata"]["corrections_applied"] = stats["applied"]
    alignment["metadata"]["corrections_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    alignment["metadata"]["corrections_stats"] = stats

    alignment_path.write_text(json.dumps(alignment, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return stats


def measure_alignment_quality(output_dir: Path) -> dict[str, Any]:
    """Measure alignment quality by comparing corrected vs original timestamps.

    For corrected words: compute abs(original_start_s - start_s) per word.
    Returns: mean_error_s, median_error_s, max_error_s, pct_within_100ms,
             pct_within_200ms, systematic_offset_s (signed mean).
    """
    corrections = load_corrections(output_dir)
    if corrections is None:
        return {}

    deltas: list[float] = []
    signed_deltas: list[float] = []
    for phrase in corrections.get("phrases", []):
        if phrase.get("status") != "corrected":
            continue
        for w in phrase.get("words", []):
            orig = w.get("original_start_s")
            curr = w.get("start_s")
            if orig is not None and curr is not None:
                d = float(curr) - float(orig)
                deltas.append(abs(d))
                signed_deltas.append(d)

    if not deltas:
        return {"total_corrected_words": 0}

    sorted_deltas = sorted(deltas)
    n = len(sorted_deltas)
    mean_err = sum(deltas) / n
    median_err = sorted_deltas[n // 2] if n % 2 else (sorted_deltas[n // 2 - 1] + sorted_deltas[n // 2]) / 2
    max_err = max(deltas)
    pct_100 = sum(1 for d in deltas if d <= 0.1) / n * 100
    pct_200 = sum(1 for d in deltas if d <= 0.2) / n * 100
    systematic = sum(signed_deltas) / n

    return {
        "total_corrected_words": n,
        "mean_error_s": round(mean_err, 4),
        "median_error_s": round(median_err, 4),
        "max_error_s": round(max_err, 4),
        "pct_within_100ms": round(pct_100, 1),
        "pct_within_200ms": round(pct_200, 1),
        "systematic_offset_s": round(systematic, 4),
    }


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
        active_word_index: index of active word in segment's words list (-1 if none)
        active_segment: full segment text containing t, or "" if none
        word_confidence: confidence of the active word (0.0 if none)
        word_progress: fractional progress through the current word [0..1]
    """
    for seg in alignment.get("segments", []):
        if seg["start_s"] <= t < seg["end_s"]:
            words = seg.get("words", [])
            for word_i, w in enumerate(words):
                if w["start_s"] <= t < w["end_s"]:
                    dur = max(1e-6, w["end_s"] - w["start_s"])
                    return {
                        "active_word": w["word"],
                        "active_word_index": word_i,
                        "active_segment": seg.get("text", ""),
                        "word_confidence": float(w.get("confidence", 0.0)),
                        "word_progress": float((t - w["start_s"]) / dur),
                    }
            # Inside segment but between words — snap to most recent ended word.
            last_word = None
            last_word_i = -1
            for word_i, w in enumerate(words):
                if w["end_s"] <= t:
                    last_word = w
                    last_word_i = word_i
            if last_word is not None:
                return {
                    "active_word": last_word["word"],
                    "active_word_index": last_word_i,
                    "active_segment": seg.get("text", ""),
                    "word_confidence": float(last_word.get("confidence", 0.0)),
                    "word_progress": 1.0,
                }
            # t is before the first word in the segment.
            return {
                "active_word": "",
                "active_word_index": -1,
                "active_segment": seg.get("text", ""),
                "word_confidence": 0.0,
                "word_progress": 0.0,
            }
    return {
        "active_word": "",
        "active_word_index": -1,
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
