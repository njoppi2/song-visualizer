from __future__ import annotations

import datetime
import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from songviz.lyrics import (
    _assign_whisper_times_to_lrc_words,
    _build_forced_align_segments_with_stats,
    _fetch_lrclib,
    _initial_phoneme_class,
    _merge_lrc_with_whisper_timing,
    _normalize_plain_lyrics_for_prompt,
    _parse_lrc,
    _read_audio_metadata,
    _should_apply_calibration,
    _snap_words_to_vocal_onset,
    _snap_words_to_vocal_onset_rms,
    _split_line_into_words,
    align_lyrics,
    apply_corrections,
    generate_corrections_template,
    load_alignment,
    load_corrections,
    lyric_activity_at,
    lyric_signals_for_timeline,
    measure_alignment_quality,
)
from songviz.paths import lyrics_alignment_path_for_output_dir, lyrics_corrections_path_for_output_dir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_alignment_payload(song_id: str = "test123") -> dict:
    """Return a minimal valid alignment.json payload (contract-compliant)."""
    return {
        "metadata": {
            "song_id": song_id,
            "pipeline_version": 7,
            "language": "en",
            "alignment_tool": "whisper",
            "whisper_model": "small",
            "backend_requested": "auto",
            "backend_used": "whisper",
            "auto_calibrate": True,
            "auto_offset_s": 0.0,
            "calibration_applied": False,
            "audio_source": "mix",
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
        "segments": [
            {
                "text": "Hello world",
                "start_s": 0.5,
                "end_s": 2.0,
                "words": [
                    {
                        "word": "Hello",
                        "start_s": 0.5,
                        "end_s": 1.0,
                        "confidence": 0.9,
                        "phones": [],
                    },
                    {
                        "word": "world",
                        "start_s": 1.2,
                        "end_s": 2.0,
                        "confidence": 0.85,
                        "phones": [],
                    },
                ],
            }
        ],
        "quality_flags": [],
        "pitch_summary": {},
    }


def _write_alignment(tmp_path: Path, payload: dict | None = None) -> dict:
    """Write alignment.json to tmp_path and return the payload."""
    if payload is None:
        payload = _make_alignment_payload()
    p = lyrics_alignment_path_for_output_dir(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return payload


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------

def test_alignment_json_required_top_level_keys(tmp_path: Path) -> None:
    payload = _make_alignment_payload()
    for key in ("metadata", "segments", "quality_flags", "pitch_summary"):
        assert key in payload, f"Missing required key: {key}"


def test_alignment_json_metadata_keys(tmp_path: Path) -> None:
    meta = _make_alignment_payload()["metadata"]
    for k in ("song_id", "language", "alignment_tool", "created_utc"):
        assert k in meta, f"Missing metadata key: {k}"


def test_alignment_json_word_time_ordering(tmp_path: Path) -> None:
    """All words must satisfy start_s <= end_s (playbook: >=90% valid)."""
    payload = _make_alignment_payload()
    words = [w for seg in payload["segments"] for w in seg.get("words", [])]
    assert words, "Expected at least one word in test payload"
    invalid = [w for w in words if w["start_s"] > w["end_s"]]
    assert not invalid, f"Words with start > end: {invalid}"


def test_alignment_json_word_fields(tmp_path: Path) -> None:
    payload = _make_alignment_payload()
    for seg in payload["segments"]:
        for w in seg.get("words", []):
            for field in ("word", "start_s", "end_s", "confidence"):
                assert field in w, f"Missing word field: {field}"
            assert isinstance(w.get("phones"), list), "phones must be a list"


# ---------------------------------------------------------------------------
# load_alignment
# ---------------------------------------------------------------------------

def test_load_alignment_returns_none_when_missing(tmp_path: Path) -> None:
    assert load_alignment(tmp_path) is None


def test_load_alignment_returns_none_on_corrupt_json(tmp_path: Path) -> None:
    p = lyrics_alignment_path_for_output_dir(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{not valid json", encoding="utf-8")
    assert load_alignment(tmp_path) is None


def test_load_alignment_reads_valid_file(tmp_path: Path) -> None:
    _write_alignment(tmp_path)
    alignment = load_alignment(tmp_path)
    assert alignment is not None
    assert alignment["metadata"]["song_id"] == "test123"
    assert alignment["metadata"]["alignment_tool"] == "whisper"
    assert len(alignment["segments"]) == 1


# ---------------------------------------------------------------------------
# lyric_activity_at
# ---------------------------------------------------------------------------

def test_lyric_activity_at_active_word(tmp_path: Path) -> None:
    alignment = _make_alignment_payload()
    result = lyric_activity_at(alignment, t=0.75)  # inside "Hello" [0.5, 1.0)
    assert result["active_word"] == "Hello"
    assert result["active_word_index"] == 0
    assert result["active_segment"] == "Hello world"
    assert result["word_confidence"] == pytest.approx(0.9)
    assert 0.0 <= result["word_progress"] <= 1.0


def test_lyric_activity_at_second_word(tmp_path: Path) -> None:
    alignment = _make_alignment_payload()
    result = lyric_activity_at(alignment, t=1.5)  # inside "world" [1.2, 2.0)
    assert result["active_word"] == "world"
    assert result["active_word_index"] == 1
    assert result["word_confidence"] == pytest.approx(0.85)


def test_lyric_activity_at_between_words(tmp_path: Path) -> None:
    alignment = _make_alignment_payload()
    # t=1.1 is between "Hello" [0.5,1.0) and "world" [1.2,2.0), still inside segment [0.5,2.0)
    # Snaps to the most recently ended word ("Hello").
    result = lyric_activity_at(alignment, t=1.1)
    assert result["active_word"] == "Hello"
    assert result["active_word_index"] == 0
    assert result["active_segment"] == "Hello world"
    assert result["word_progress"] == 1.0


def test_lyric_activity_at_outside_all_segments(tmp_path: Path) -> None:
    alignment = _make_alignment_payload()
    result = lyric_activity_at(alignment, t=5.0)
    assert result["active_word"] == ""
    assert result["active_word_index"] == -1
    assert result["active_segment"] == ""
    assert result["word_confidence"] == 0.0
    assert result["word_progress"] == 0.0


def test_lyric_activity_at_empty_alignment(tmp_path: Path) -> None:
    result = lyric_activity_at({"segments": []}, t=0.5)
    assert result["active_word"] == ""
    assert result["active_word_index"] == -1


def test_lyric_activity_at_before_first_word(tmp_path: Path) -> None:
    """t is inside the segment but before the first word starts."""
    alignment = _make_alignment_payload()
    # Segment starts at 0.5, first word "Hello" starts at 0.5 — shift segment start earlier.
    alignment["segments"][0]["start_s"] = 0.2
    result = lyric_activity_at(alignment, t=0.3)  # inside segment [0.2,2.0) but before "Hello" [0.5,1.0)
    assert result["active_word"] == ""
    assert result["active_word_index"] == -1
    assert result["active_segment"] == "Hello world"


def test_lyric_activity_at_duplicate_words(tmp_path: Path) -> None:
    """Duplicate words in a line are distinguished by index, not text."""
    alignment = {
        "segments": [
            {
                "text": "never never",
                "start_s": 0.0,
                "end_s": 4.0,
                "words": [
                    {"word": "never", "start_s": 0.0, "end_s": 1.5, "confidence": 0.9, "phones": []},
                    {"word": "never", "start_s": 2.0, "end_s": 3.5, "confidence": 0.9, "phones": []},
                ],
            }
        ]
    }
    result = lyric_activity_at(alignment, t=2.5)  # inside second "never"
    assert result["active_word"] == "never"
    assert result["active_word_index"] == 1


# ---------------------------------------------------------------------------
# lyric_signals_for_timeline
# ---------------------------------------------------------------------------

def test_lyric_signals_length_matches_timeline(tmp_path: Path) -> None:
    alignment = _make_alignment_payload()
    times = [0.0, 0.75, 1.1, 1.5, 5.0]
    signals = lyric_signals_for_timeline(alignment, times)
    assert len(signals["word_active"]) == len(times)
    assert len(signals["phrase_active"]) == len(times)
    assert len(signals["word_confidence"]) == len(times)


def test_lyric_signals_values(tmp_path: Path) -> None:
    alignment = _make_alignment_payload()
    times = [0.0, 0.75, 1.5, 5.0]
    signals = lyric_signals_for_timeline(alignment, times)
    # t=0.0: before any segment
    assert signals["word_active"][0] == 0.0
    assert signals["phrase_active"][0] == 0.0
    # t=0.75: inside "Hello"
    assert signals["word_active"][1] == 1.0
    assert signals["phrase_active"][1] == 1.0
    # t=1.5: inside "world"
    assert signals["word_active"][2] == 1.0
    # t=5.0: after all segments
    assert signals["word_active"][3] == 0.0
    assert signals["phrase_active"][3] == 0.0


def test_lyric_signals_empty_timeline(tmp_path: Path) -> None:
    alignment = _make_alignment_payload()
    signals = lyric_signals_for_timeline(alignment, [])
    assert signals["word_active"] == []
    assert signals["phrase_active"] == []
    assert signals["word_confidence"] == []


# ---------------------------------------------------------------------------
# align_lyrics (pipeline flow — whisper mocked)
# ---------------------------------------------------------------------------

_MOCK_WHISPER_SEGMENTS = [
    {
        "text": " Hello world",
        "start": 0.5,
        "end": 2.0,
        "words": [
            {"word": " Hello", "start": 0.5, "end": 1.0, "probability": 0.9},
            {"word": " world", "start": 1.2, "end": 2.0, "probability": 0.85},
        ],
    }
]


def test_align_lyrics_writes_file(tmp_path: Path) -> None:
    fake_audio = tmp_path / "test.wav"
    fake_audio.write_bytes(b"\x00" * 44)  # fake WAV header

    with patch("songviz.lyrics._whisper_align", return_value=_MOCK_WHISPER_SEGMENTS):
        result = align_lyrics(
            fake_audio,
            song_id="abc123",
            output_dir=tmp_path,
            language="en",
        )

    assert result == lyrics_alignment_path_for_output_dir(tmp_path)
    assert result.exists()

    data = json.loads(result.read_text())
    assert data["metadata"]["song_id"] == "abc123"
    assert data["metadata"]["alignment_tool"] == "whisper"
    assert data["metadata"]["audio_source"] == "mix"
    assert len(data["segments"]) == 1
    # Leading/trailing spaces on words must be stripped.
    assert data["segments"][0]["words"][0]["word"] == "Hello"
    assert data["segments"][0]["words"][1]["word"] == "world"


def test_align_lyrics_prefers_vocals_stem(tmp_path: Path) -> None:
    fake_audio = tmp_path / "test.wav"
    fake_audio.write_bytes(b"\x00" * 44)
    vocals = tmp_path / "stems" / "vocals.wav"
    vocals.parent.mkdir(parents=True, exist_ok=True)
    vocals.write_bytes(b"\x00" * 44)

    with patch("songviz.lyrics._whisper_align", return_value=_MOCK_WHISPER_SEGMENTS) as mock_align:
        align_lyrics(
            fake_audio,
            song_id="abc123",
            output_dir=tmp_path,
            vocals_stem=vocals,
            force=True,
        )
        called_path = mock_align.call_args[0][0]
        assert called_path == vocals

    data = json.loads(lyrics_alignment_path_for_output_dir(tmp_path).read_text())
    assert data["metadata"]["audio_source"] == "vocals_stem"


def test_align_lyrics_cached_not_rerun(tmp_path: Path) -> None:
    """align_lyrics should skip whisper when file already exists (force=False)."""
    _write_alignment(tmp_path)

    fake_audio = tmp_path / "test.wav"
    fake_audio.write_bytes(b"\x00" * 44)

    with patch("songviz.lyrics._whisper_align") as mock_align:
        align_lyrics(fake_audio, song_id="test123", output_dir=tmp_path, force=False)
        mock_align.assert_not_called()


def test_align_lyrics_force_reruns(tmp_path: Path) -> None:
    _write_alignment(tmp_path)

    fake_audio = tmp_path / "test.wav"
    fake_audio.write_bytes(b"\x00" * 44)

    with patch("songviz.lyrics._whisper_align", return_value=_MOCK_WHISPER_SEGMENTS) as mock_align:
        align_lyrics(fake_audio, song_id="test123", output_dir=tmp_path, force=True)
        mock_align.assert_called_once()


def test_align_lyrics_stale_cache_reruns(tmp_path: Path) -> None:
    payload = _make_alignment_payload()
    del payload["metadata"]["pipeline_version"]
    _write_alignment(tmp_path, payload=payload)

    fake_audio = tmp_path / "test.wav"
    fake_audio.write_bytes(b"\x00" * 44)

    with patch("songviz.lyrics._whisper_align", return_value=_MOCK_WHISPER_SEGMENTS) as mock_align:
        align_lyrics(fake_audio, song_id="test123", output_dir=tmp_path, force=False)
        mock_align.assert_called_once()


def test_align_lyrics_90pct_word_time_ordering(tmp_path: Path) -> None:
    """Playbook acceptance criterion: >=90% of aligned words have start_s <= end_s."""
    fake_audio = tmp_path / "test.wav"
    fake_audio.write_bytes(b"\x00" * 44)

    with patch("songviz.lyrics._whisper_align", return_value=_MOCK_WHISPER_SEGMENTS):
        result_path = align_lyrics(fake_audio, song_id="t", output_dir=tmp_path)

    data = json.loads(result_path.read_text())
    words = [w for seg in data["segments"] for w in seg.get("words", [])]
    if words:
        valid = sum(1 for w in words if w["start_s"] <= w["end_s"])
        assert valid / len(words) >= 0.9


# ---------------------------------------------------------------------------
# _parse_lrc
# ---------------------------------------------------------------------------

_BASIC_LRC = "[00:10.50] Hello world\n[00:15.00] Goodbye friend\n"
_3DIGIT_LRC = "[01:02.345] Testing one two three\n"


def test_parse_lrc_basic() -> None:
    segs = _parse_lrc(_BASIC_LRC)
    assert len(segs) == 2
    assert segs[0]["text"] == "Hello world"
    assert segs[0]["start_s"] == pytest.approx(10.5)
    assert segs[0]["end_s"] == pytest.approx(15.0)
    assert segs[1]["text"] == "Goodbye friend"
    assert segs[1]["start_s"] == pytest.approx(15.0)
    assert segs[1]["end_s"] == pytest.approx(20.0)  # last line +5 s
    # Words present in both segments
    assert len(segs[0]["words"]) == 2
    assert len(segs[1]["words"]) == 2


def test_parse_lrc_empty() -> None:
    assert _parse_lrc("") == []


def test_parse_lrc_skips_empty_lines() -> None:
    lrc = "[00:01.00] First line\n[00:05.00]\n[00:10.00] Second line\n"
    segs = _parse_lrc(lrc)
    # Empty timestamp at 00:05 is skipped; First line ends at 00:05
    assert len(segs) == 2
    assert segs[0]["end_s"] == pytest.approx(5.0)
    assert segs[1]["text"] == "Second line"


def test_parse_lrc_3digit_ms() -> None:
    segs = _parse_lrc(_3DIGIT_LRC)
    assert len(segs) == 1
    # [01:02.345] → 62 + 0.345 = 62.345 s
    assert segs[0]["start_s"] == pytest.approx(62.345)
    assert segs[0]["end_s"] == pytest.approx(62.345 + 5.0)


# ---------------------------------------------------------------------------
# _split_line_into_words
# ---------------------------------------------------------------------------


def test_split_line_proportional_timing() -> None:
    # "A" (1 char) and "BBBBBB" (6 chars) over 7 s total.
    # Slots are proportional: slot_A = 1 s, slot_B = 6 s.
    # Non-last words have a 5% gap, so actual word A duration = 0.95 s.
    # Last word fills its full slot: word B start = 1.0, end = 7.0 → 6.0 s.
    words = _split_line_into_words("A BBBBBB", 0.0, 7.0)
    assert len(words) == 2
    a_dur = words[0]["end_s"] - words[0]["start_s"]
    b_dur = words[1]["end_s"] - words[1]["start_s"]
    # B's slot is 6× A's slot; actual durations reflect that ordering clearly
    assert b_dur > a_dur * 5  # at least 5× longer (strict proportionality holds for slots)


def test_split_line_empty() -> None:
    assert _split_line_into_words("", 0.0, 5.0) == []


def test_split_line_last_word_fills_to_end() -> None:
    words = _split_line_into_words("one two", 0.0, 10.0)
    assert words[-1]["end_s"] == pytest.approx(10.0)


def test_split_line_confidence_and_phones() -> None:
    words = _split_line_into_words("test", 0.0, 1.0)
    assert words[0]["confidence"] == 1.0
    assert words[0]["phones"] == []


# ---------------------------------------------------------------------------
# _normalize_plain_lyrics_for_prompt
# ---------------------------------------------------------------------------


def test_normalize_plain_strips_section_markers() -> None:
    text = "[Chorus]\nHello world\n[Verse 2]\nGoodbye friend"
    result = _normalize_plain_lyrics_for_prompt(text)
    assert "[Chorus]" not in result
    assert "[Verse 2]" not in result
    assert "Hello world" in result
    assert "Goodbye friend" in result


def test_normalize_plain_preserves_lrc_timestamps() -> None:
    # LRC timestamp lines should NOT be stripped (they start with digits after [)
    text = "[00:10.50] Hello world"
    result = _normalize_plain_lyrics_for_prompt(text)
    # The regex only removes [Word...] style markers, not [MM:SS.MS]
    assert "[00:10.50]" in result


# ---------------------------------------------------------------------------
# _fetch_lrclib
# ---------------------------------------------------------------------------


def test_fetch_lrclib_success() -> None:
    fake_response = json.dumps(
        {"syncedLyrics": "[00:01.00] test", "plainLyrics": "test"}
    ).encode("utf-8")

    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.status = 200
    mock_resp.read.return_value = fake_response

    with patch("urllib.request.urlopen", return_value=mock_resp):
        result = _fetch_lrclib("Artist", "Title", 180.0)

    assert result is not None
    assert "syncedLyrics" in result


def test_fetch_lrclib_network_error() -> None:
    with patch("urllib.request.urlopen", side_effect=OSError("timeout")):
        result = _fetch_lrclib("Artist", "Title")

    assert result is None


# ---------------------------------------------------------------------------
# _read_audio_metadata
# ---------------------------------------------------------------------------


def test_read_audio_metadata_no_mutagen(tmp_path: Path) -> None:
    fake = tmp_path / "audio.mp3"
    fake.write_bytes(b"\x00" * 100)

    with patch("builtins.__import__", side_effect=ImportError("no mutagen")):
        # We can't easily mock __import__ cleanly in all cases, so patch at module level
        pass

    # Simpler: patch mutagen inside the function
    with patch.dict("sys.modules", {"mutagen": None}):
        result = _read_audio_metadata(fake)

    assert result == {"artist": None, "title": None, "duration_s": None}


def test_read_audio_metadata_unreadable_file(tmp_path: Path) -> None:
    fake = tmp_path / "garbage.mp3"
    fake.write_bytes(b"\xff\xfe" * 10)  # nonsense bytes

    # Should return all-None gracefully, not raise
    result = _read_audio_metadata(fake)
    assert isinstance(result, dict)
    for key in ("artist", "title", "duration_s"):
        assert key in result


# ---------------------------------------------------------------------------
# align_lyrics — LRCLIB paths
# ---------------------------------------------------------------------------

_SYNCED_LRC = "[00:01.00] Hello world\n[00:05.00] Goodbye friend\n"
_PLAIN_LYRICS = "[Chorus]\nHello world\nGoodbye friend"
_LRCLIB_SYNCED_RESPONSE = {"syncedLyrics": _SYNCED_LRC, "plainLyrics": _PLAIN_LYRICS}
_LRCLIB_PLAIN_RESPONSE = {"syncedLyrics": None, "plainLyrics": _PLAIN_LYRICS}


def test_align_lyrics_lrclib_synced_path(tmp_path: Path) -> None:
    """When LRCLIB returns synced lyrics and Whisper is available, use hybrid timing."""
    fake_audio = tmp_path / "test.wav"
    fake_audio.write_bytes(b"\x00" * 44)

    with (
        patch("songviz.lyrics._read_audio_metadata", return_value={"artist": "A", "title": "T", "duration_s": 180.0}),
        patch("songviz.lyrics._fetch_lrclib", return_value=_LRCLIB_SYNCED_RESPONSE),
        patch("songviz.lyrics._whisper_available", return_value=True),
        patch("songviz.lyrics._whisper_align", return_value=_MOCK_WHISPER_SEGMENTS) as mock_whisper,
    ):
        result = align_lyrics(fake_audio, song_id="s1", output_dir=tmp_path)
        mock_whisper.assert_called_once()

    data = json.loads(result.read_text())
    assert data["metadata"]["alignment_tool"] == "lrclib+whisper_timing"
    assert data["metadata"]["audio_source"] == "mix"
    assert len(data["segments"]) == 2


def test_align_lyrics_lrclib_synced_no_whisper(tmp_path: Path) -> None:
    """When LRCLIB has synced lyrics but Whisper is not installed, use proportional timing."""
    fake_audio = tmp_path / "test.wav"
    fake_audio.write_bytes(b"\x00" * 44)

    with (
        patch("songviz.lyrics._read_audio_metadata", return_value={"artist": "A", "title": "T", "duration_s": 180.0}),
        patch("songviz.lyrics._fetch_lrclib", return_value=_LRCLIB_SYNCED_RESPONSE),
        patch("songviz.lyrics._resolve_backend_order", return_value=[]),
        patch("songviz.lyrics._whisper_available", return_value=False),
        patch("songviz.lyrics._whisper_align") as mock_whisper,
    ):
        result = align_lyrics(fake_audio, song_id="s1", output_dir=tmp_path)
        mock_whisper.assert_not_called()

    data = json.loads(result.read_text())
    assert data["metadata"]["alignment_tool"] == "lrclib_synced"
    assert data["metadata"]["audio_source"] == "lrclib"
    assert len(data["segments"]) == 2


def test_align_lyrics_lrclib_plain_prompts_whisper(tmp_path: Path) -> None:
    """When LRCLIB has only plain lyrics, Whisper is called with initial_prompt."""
    fake_audio = tmp_path / "test.wav"
    fake_audio.write_bytes(b"\x00" * 44)

    with (
        patch("songviz.lyrics._read_audio_metadata", return_value={"artist": "A", "title": "T", "duration_s": 180.0}),
        patch("songviz.lyrics._fetch_lrclib", return_value=_LRCLIB_PLAIN_RESPONSE),
        patch("songviz.lyrics._whisper_align", return_value=_MOCK_WHISPER_SEGMENTS) as mock_whisper,
    ):
        result = align_lyrics(fake_audio, song_id="s2", output_dir=tmp_path)
        mock_whisper.assert_called_once()
        _, kwargs = mock_whisper.call_args
        assert kwargs.get("initial_prompt") is not None
        assert "Hello world" in kwargs["initial_prompt"]

    data = json.loads(result.read_text())
    assert data["metadata"]["alignment_tool"] == "whisper+lrclib_prompt"


def test_align_lyrics_falls_back_to_pure_whisper(tmp_path: Path) -> None:
    """When no metadata is found, LRCLIB is skipped and pure Whisper is used."""
    fake_audio = tmp_path / "test.wav"
    fake_audio.write_bytes(b"\x00" * 44)

    with (
        patch("songviz.lyrics._read_audio_metadata", return_value={"artist": None, "title": None, "duration_s": None}),
        patch("songviz.lyrics._fetch_lrclib") as mock_lrclib,
        patch("songviz.lyrics._whisper_align", return_value=_MOCK_WHISPER_SEGMENTS) as mock_whisper,
    ):
        result = align_lyrics(fake_audio, song_id="s3", output_dir=tmp_path)
        mock_lrclib.assert_not_called()
        mock_whisper.assert_called_once()

    data = json.loads(result.read_text())
    assert data["metadata"]["alignment_tool"] == "whisper"


def test_align_lyrics_cli_overrides_tags(tmp_path: Path) -> None:
    """CLI --artist/--title override ID3 tag values."""
    fake_audio = tmp_path / "test.wav"
    fake_audio.write_bytes(b"\x00" * 44)

    with (
        patch("songviz.lyrics._read_audio_metadata", return_value={"artist": "TagArtist", "title": "TagTitle", "duration_s": None}),
        patch("songviz.lyrics._fetch_lrclib", return_value=None) as mock_lrclib,
        patch("songviz.lyrics._whisper_align", return_value=_MOCK_WHISPER_SEGMENTS),
    ):
        align_lyrics(fake_audio, song_id="s4", output_dir=tmp_path, artist="CLIArtist", title="CLITitle")
        _, kwargs = mock_lrclib.call_args
        # _fetch_lrclib is called with positional args: artist, title, duration_s
        call_args = mock_lrclib.call_args[0]
        assert call_args[0] == "CLIArtist"
        assert call_args[1] == "CLITitle"


def test_align_lyrics_auto_backend_falls_back_to_whisper(tmp_path: Path) -> None:
    """If whisperx fails in auto mode, fallback to whisper and mark quality flag."""
    fake_audio = tmp_path / "test.wav"
    fake_audio.write_bytes(b"\x00" * 44)

    with (
        patch("songviz.lyrics._read_audio_metadata", return_value={"artist": "A", "title": "T", "duration_s": 180.0}),
        patch("songviz.lyrics._fetch_lrclib", return_value=_LRCLIB_SYNCED_RESPONSE),
        patch("songviz.lyrics._resolve_backend_order", return_value=["whisperx", "whisper"]),
        patch("songviz.lyrics._whisperx_align", side_effect=RuntimeError("whisperx failed")),
        patch("songviz.lyrics._whisper_align", return_value=_MOCK_WHISPER_SEGMENTS),
        patch("songviz.lyrics._estimate_global_offset_s", return_value=None),
    ):
        result = align_lyrics(fake_audio, song_id="s5", output_dir=tmp_path)

    data = json.loads(result.read_text())
    assert data["metadata"]["alignment_tool"] == "lrclib+whisper_timing"
    assert data["metadata"]["backend_used"] == "whisper"
    assert "backend_fallback_used" in data["quality_flags"]


def test_align_lyrics_applies_auto_calibration_offset(tmp_path: Path) -> None:
    fake_audio = tmp_path / "test.wav"
    fake_audio.write_bytes(b"\x00" * 44)

    calibration = {
        "estimated_offset_s": 0.1,
        "corr_at_zero": 0.1,
        "corr_at_best": 0.2,
        "corr_improvement": 0.1,
        "applied": True,
    }

    with (
        patch("songviz.lyrics._read_audio_metadata", return_value={"artist": None, "title": None, "duration_s": None}),
        patch("songviz.lyrics._whisper_align", return_value=_MOCK_WHISPER_SEGMENTS),
        patch("songviz.lyrics._estimate_global_offset_s", return_value=calibration),
    ):
        result = align_lyrics(fake_audio, song_id="s6", output_dir=tmp_path, align_backend="whisper")

    data = json.loads(result.read_text())
    w0 = data["segments"][0]["words"][0]
    assert w0["start_s"] == pytest.approx(0.6)
    assert data["metadata"]["auto_offset_s"] == pytest.approx(0.1)
    assert data["metadata"]["calibration_applied"] is True


def test_align_lyrics_whisperx_backend_tool_label(tmp_path: Path) -> None:
    fake_audio = tmp_path / "test.wav"
    fake_audio.write_bytes(b"\x00" * 44)

    with (
        patch("songviz.lyrics._read_audio_metadata", return_value={"artist": None, "title": None, "duration_s": None}),
        patch("songviz.lyrics._whisperx_align", return_value=_MOCK_WHISPER_SEGMENTS),
        patch("songviz.lyrics._estimate_global_offset_s", return_value=None),
    ):
        result = align_lyrics(fake_audio, song_id="s7", output_dir=tmp_path, align_backend="whisperx")

    data = json.loads(result.read_text())
    assert data["metadata"]["alignment_tool"] == "whisperx"
    assert data["metadata"]["backend_used"] == "whisperx"


# ---------------------------------------------------------------------------
# _assign_whisper_times_to_lrc_words and _merge_lrc_with_whisper_timing
# ---------------------------------------------------------------------------

def _make_whisper_word(word: str, start: float, end: float) -> dict:
    return {"word": word, "start_s": start, "end_s": end, "confidence": 0.9, "phones": []}


def test_assign_whisper_times_exact_match() -> None:
    """Exact word match → Whisper timestamps used directly."""
    lrc = ["Hello", "world"]
    wh = [_make_whisper_word("Hello", 1.0, 1.4), _make_whisper_word("world", 1.5, 2.0)]
    result = _assign_whisper_times_to_lrc_words(lrc, wh, 1.0, 2.0)
    assert result[0]["word"] == "Hello"
    assert result[0]["start_s"] == pytest.approx(1.0)
    assert result[1]["word"] == "world"
    assert result[1]["start_s"] == pytest.approx(1.5)


def test_assign_whisper_times_spelling_variant() -> None:
    """LRCLIB 'colour' vs Whisper 'color' → Whisper timestamp used, LRCLIB word kept."""
    lrc = ["colour"]
    wh = [_make_whisper_word("color", 2.0, 2.5)]
    result = _assign_whisper_times_to_lrc_words(lrc, wh, 2.0, 2.5)
    assert result[0]["word"] == "colour"
    assert result[0]["start_s"] == pytest.approx(2.0)


def test_assign_whisper_times_missing_word_interpolated() -> None:
    """When Whisper misses a word, its timing is interpolated from neighbors."""
    lrc = ["one", "two", "three"]
    # Whisper only found "one" and "three", missed "two"
    wh = [_make_whisper_word("one", 0.0, 0.5), _make_whisper_word("three", 1.5, 2.0)]
    result = _assign_whisper_times_to_lrc_words(lrc, wh, 0.0, 2.0)
    assert result[0]["word"] == "one"
    assert result[2]["word"] == "three"
    # "two" gets an interpolated time between 0.5 and 1.5
    assert result[1]["word"] == "two"
    assert 0.5 <= result[1]["start_s"] <= 1.5


def test_merge_lrc_with_whisper_timing_uses_whisper_timestamps() -> None:
    """merge function: words inside a line window get Whisper timestamps."""
    lrc_segs = [{"text": "Hello world", "start_s": 1.0, "end_s": 3.0, "words": []}]
    wh_segs = [{"text": "Hello world", "start_s": 1.0, "end_s": 3.0, "words": [
        _make_whisper_word("Hello", 1.1, 1.6),
        _make_whisper_word("world", 1.8, 2.5),
    ]}]
    result = _merge_lrc_with_whisper_timing(lrc_segs, wh_segs)
    assert len(result) == 1
    assert result[0]["words"][0]["start_s"] == pytest.approx(1.1)
    assert result[0]["words"][1]["start_s"] == pytest.approx(1.8)


def test_merge_lrc_with_whisper_timing_fallback_on_no_coverage() -> None:
    """Lines with no Whisper words in window fall back to proportional timing."""
    lrc_segs = [{"text": "Hello world", "start_s": 10.0, "end_s": 12.0, "words": []}]
    wh_segs = []  # no Whisper words at all
    result = _merge_lrc_with_whisper_timing(lrc_segs, wh_segs)
    assert len(result) == 1
    # Proportional: "Hello"(5) and "world"(5) split evenly
    assert result[0]["words"][0]["start_s"] == pytest.approx(10.0)
    assert result[0]["words"][-1]["end_s"] == pytest.approx(12.0)


# ---------------------------------------------------------------------------
# Forced alignment
# ---------------------------------------------------------------------------

def test_stable_whisper_forced_align_path(tmp_path: Path) -> None:
    """When stable-ts forced alignment succeeds, alignment_tool is 'lrclib+stable_whisper_forced'
    and model.transcribe is NOT called."""
    fake_audio = tmp_path / "test.wav"
    fake_audio.write_bytes(b"\x00" * 44)

    # FA output: segments with word-level timestamps matching LRCLIB text
    fa_raw_segments = [
        {
            "text": "Hello world",
            "start": 1.0,
            "end": 5.0,
            "words": [
                {"word": " Hello", "start": 1.1, "end": 2.0, "probability": 0.95},
                {"word": " world", "start": 2.1, "end": 4.5, "probability": 0.92},
            ],
        },
        {
            "text": "Goodbye friend",
            "start": 5.0,
            "end": 10.0,
            "words": [
                {"word": " Goodbye", "start": 5.2, "end": 7.0, "probability": 0.88},
                {"word": " friend", "start": 7.1, "end": 9.5, "probability": 0.91},
            ],
        },
    ]

    mock_result = MagicMock()
    mock_result.to_dict.return_value = {"segments": fa_raw_segments}

    mock_model = MagicMock()
    mock_model.align.return_value = mock_result

    mock_stable_whisper = MagicMock()
    mock_stable_whisper.load_model.return_value = mock_model

    with (
        patch("songviz.lyrics._read_audio_metadata", return_value={"artist": "A", "title": "T", "duration_s": 180.0}),
        patch("songviz.lyrics._fetch_lrclib", return_value=_LRCLIB_SYNCED_RESPONSE),
        patch("songviz.lyrics._stable_whisper_available", return_value=True),
        patch("songviz.lyrics._whisper_available", return_value=False),
        patch("songviz.lyrics._whisperx_available", return_value=False),
        patch.dict("sys.modules", {"stable_whisper": mock_stable_whisper}),
        patch("songviz.lyrics._estimate_global_offset_s", return_value=None),
    ):
        result = align_lyrics(fake_audio, song_id="fa1", output_dir=tmp_path)

    data = json.loads(result.read_text())
    assert data["metadata"]["alignment_tool"] == "lrclib+stable_whisper_forced"
    assert data["metadata"]["backend_used"] == "stable_whisper"
    # Word timestamps come from FA output (after normalization)
    assert data["segments"][0]["words"][0]["start_s"] == pytest.approx(1.1)
    assert data["segments"][0]["words"][1]["start_s"] == pytest.approx(2.1)
    # model.transcribe should NOT have been called
    mock_model.transcribe.assert_not_called()
    # model.align should have been called
    mock_model.align.assert_called_once()


def test_forced_align_fallback_to_transcribe_merge(tmp_path: Path) -> None:
    """When forced alignment fails, fall back to the existing transcribe+merge path."""
    fake_audio = tmp_path / "test.wav"
    fake_audio.write_bytes(b"\x00" * 44)

    mock_model = MagicMock()
    mock_model.align.side_effect = RuntimeError("FA failed")
    # transcribe returns standard Whisper output for the fallback path
    mock_transcribe_result = MagicMock()
    mock_transcribe_result.to_dict.return_value = {"segments": [
        {
            "text": " Hello world",
            "start": 1.0,
            "end": 5.0,
            "words": [
                {"word": " Hello", "start": 1.0, "end": 2.0, "probability": 0.9},
                {"word": " world", "start": 2.5, "end": 4.5, "probability": 0.85},
            ],
        },
    ]}
    mock_model.transcribe.return_value = mock_transcribe_result

    mock_stable_whisper = MagicMock()
    mock_stable_whisper.load_model.return_value = mock_model

    with (
        patch("songviz.lyrics._read_audio_metadata", return_value={"artist": "A", "title": "T", "duration_s": 180.0}),
        patch("songviz.lyrics._fetch_lrclib", return_value=_LRCLIB_SYNCED_RESPONSE),
        patch("songviz.lyrics._stable_whisper_available", return_value=True),
        patch("songviz.lyrics._whisper_available", return_value=False),
        patch("songviz.lyrics._whisperx_available", return_value=False),
        patch.dict("sys.modules", {"stable_whisper": mock_stable_whisper}),
        patch("songviz.lyrics._estimate_global_offset_s", return_value=None),
    ):
        result = align_lyrics(fake_audio, song_id="fa2", output_dir=tmp_path)

    data = json.loads(result.read_text())
    # Should fall back to transcribe+merge
    assert data["metadata"]["alignment_tool"] == "lrclib+stable_whisper_timing"
    assert data["metadata"]["backend_used"] == "stable_whisper"
    assert "backend_fallback_used" in data["quality_flags"]


def test_build_forced_align_segments_word_mapping() -> None:
    """_build_forced_align_segments_with_stats maps FA words 1:1 to LRCLIB words."""
    lrc_segments = [
        {"text": "Hello world", "start_s": 1.0, "end_s": 5.0, "words": []},
        {"text": "Goodbye friend", "start_s": 5.0, "end_s": 10.0, "words": []},
    ]
    fa_segments = [
        {
            "text": "Hello world",
            "start_s": 1.1,
            "end_s": 4.5,
            "words": [
                {"word": "Hello", "start_s": 1.1, "end_s": 2.0, "confidence": 0.95, "phones": []},
                {"word": "world", "start_s": 2.1, "end_s": 4.5, "confidence": 0.92, "phones": []},
            ],
        },
        {
            "text": "Goodbye friend",
            "start_s": 5.2,
            "end_s": 9.5,
            "words": [
                {"word": "Goodbye", "start_s": 5.2, "end_s": 7.0, "confidence": 0.88, "phones": []},
                {"word": "friend", "start_s": 7.1, "end_s": 9.5, "confidence": 0.91, "phones": []},
            ],
        },
    ]
    segments, stats = _build_forced_align_segments_with_stats(lrc_segments, fa_segments)

    assert len(segments) == 2
    # LRCLIB segment boundaries are preserved
    assert segments[0]["start_s"] == 1.0
    assert segments[0]["end_s"] == 5.0
    # Word timestamps come from FA
    assert segments[0]["words"][0]["word"] == "Hello"
    assert segments[0]["words"][0]["start_s"] == pytest.approx(1.1)
    assert segments[0]["words"][1]["word"] == "world"
    assert segments[0]["words"][1]["start_s"] == pytest.approx(2.1)
    assert segments[1]["words"][0]["word"] == "Goodbye"
    assert segments[1]["words"][0]["start_s"] == pytest.approx(5.2)
    assert segments[1]["words"][1]["word"] == "friend"
    assert segments[1]["words"][1]["start_s"] == pytest.approx(7.1)
    # All words matched
    assert stats["total_words"] == 4
    assert stats["matched_words"] == 4
    assert stats["matched_word_ratio"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Corrections
# ---------------------------------------------------------------------------


def _write_corrections_yaml(tmp_path: Path, phrases: list[dict]) -> Path:
    """Write a corrections.yaml file and return the path."""
    import yaml

    corrections_path = lyrics_corrections_path_for_output_dir(tmp_path)
    corrections_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"song_id": "test123", "source": "test", "phrases": phrases}
    corrections_path.write_text(yaml.dump(payload, sort_keys=False), encoding="utf-8")
    return corrections_path


def test_generate_corrections_template(tmp_path: Path) -> None:
    _write_alignment(tmp_path)
    result = generate_corrections_template(tmp_path)
    assert result.exists()

    import yaml

    data = yaml.safe_load(result.read_text())
    assert len(data["phrases"]) == 1
    assert data["phrases"][0]["status"] == "auto"
    assert data["phrases"][0]["text"] == "Hello world"
    assert len(data["phrases"][0]["words"]) == 2


def test_generate_corrections_template_preserves_edits(tmp_path: Path) -> None:
    _write_alignment(tmp_path)

    # Write initial corrections with an edit.
    _write_corrections_yaml(tmp_path, [{
        "segment": 0,
        "text": "Hello world",
        "status": "corrected",
        "words": [
            {"word": "Hello", "start_s": 0.6, "end_s": 1.1, "original_start_s": 0.5, "original_end_s": 1.0},
            {"word": "world", "start_s": 1.3, "end_s": 2.1, "original_start_s": 1.2, "original_end_s": 2.0},
        ],
    }])

    # Regenerate template — should preserve the edit.
    result = generate_corrections_template(tmp_path)

    import yaml

    data = yaml.safe_load(result.read_text())
    assert data["phrases"][0]["status"] == "corrected"
    assert data["phrases"][0]["words"][0]["start_s"] == 0.6


def test_apply_corrections_replaces_timestamps(tmp_path: Path) -> None:
    _write_alignment(tmp_path)
    _write_corrections_yaml(tmp_path, [{
        "segment": 0,
        "text": "Hello world",
        "status": "corrected",
        "words": [
            {"word": "Hello", "start_s": 0.6, "end_s": 1.1, "original_start_s": 0.5, "original_end_s": 1.0},
            {"word": "world", "start_s": 1.3, "end_s": 2.1, "original_start_s": 1.2, "original_end_s": 2.0},
        ],
    }])

    stats = apply_corrections(tmp_path)
    assert stats["applied"] == 1

    alignment = load_alignment(tmp_path)
    assert alignment is not None
    assert alignment["segments"][0]["words"][0]["start_s"] == pytest.approx(0.6)
    assert alignment["segments"][0]["words"][1]["start_s"] == pytest.approx(1.3)


def test_apply_corrections_skips_auto(tmp_path: Path) -> None:
    _write_alignment(tmp_path)
    _write_corrections_yaml(tmp_path, [{
        "segment": 0,
        "text": "Hello world",
        "status": "auto",
        "words": [
            {"word": "Hello", "start_s": 99.0, "end_s": 99.5, "original_start_s": 0.5, "original_end_s": 1.0},
        ],
    }])

    stats = apply_corrections(tmp_path)
    assert stats["skipped"] == 1
    assert stats["applied"] == 0

    alignment = load_alignment(tmp_path)
    assert alignment is not None
    # Timestamps unchanged.
    assert alignment["segments"][0]["words"][0]["start_s"] == pytest.approx(0.5)


def test_apply_corrections_skips_text_mismatch(tmp_path: Path) -> None:
    _write_alignment(tmp_path)
    _write_corrections_yaml(tmp_path, [{
        "segment": 0,
        "text": "Wrong text entirely",
        "status": "corrected",
        "words": [],
    }])

    stats = apply_corrections(tmp_path)
    assert stats["mismatched"] == 1
    assert stats["applied"] == 0


def test_measure_alignment_quality(tmp_path: Path) -> None:
    _write_alignment(tmp_path)
    # Corrections with known deltas: +0.1 and +0.1.
    _write_corrections_yaml(tmp_path, [{
        "segment": 0,
        "text": "Hello world",
        "status": "corrected",
        "words": [
            {"word": "Hello", "start_s": 0.6, "end_s": 1.1, "original_start_s": 0.5, "original_end_s": 1.0},
            {"word": "world", "start_s": 1.3, "end_s": 2.1, "original_start_s": 1.2, "original_end_s": 2.0},
        ],
    }])

    quality = measure_alignment_quality(tmp_path)
    assert quality["total_corrected_words"] == 2
    assert quality["mean_error_s"] == pytest.approx(0.1)
    assert quality["median_error_s"] == pytest.approx(0.1)
    assert quality["pct_within_200ms"] == pytest.approx(100.0)
    assert quality["systematic_offset_s"] == pytest.approx(0.1)


def test_corrections_reapplied_after_force(tmp_path: Path) -> None:
    """align_lyrics(force=True) should reapply corrections if corrections.yaml exists."""
    fake_audio = tmp_path / "test.wav"
    fake_audio.write_bytes(b"\x00" * 44)

    # Initial alignment.
    with patch("songviz.lyrics._whisper_align", return_value=_MOCK_WHISPER_SEGMENTS):
        align_lyrics(fake_audio, song_id="abc123", output_dir=tmp_path)

    # Write corrections that shift Hello from 0.5 to 0.6.
    _write_corrections_yaml(tmp_path, [{
        "segment": 0,
        "text": "Hello world",
        "status": "corrected",
        "words": [
            {"word": "Hello", "start_s": 0.6, "end_s": 1.1, "original_start_s": 0.5, "original_end_s": 1.0},
            {"word": "world", "start_s": 1.3, "end_s": 2.1, "original_start_s": 1.2, "original_end_s": 2.0},
        ],
    }])

    # Force re-run — corrections should be reapplied.
    with patch("songviz.lyrics._whisper_align", return_value=_MOCK_WHISPER_SEGMENTS):
        align_lyrics(fake_audio, song_id="abc123", output_dir=tmp_path, force=True)

    alignment = load_alignment(tmp_path)
    assert alignment is not None
    assert alignment["segments"][0]["words"][0]["start_s"] == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# _should_apply_calibration threshold tests
# ---------------------------------------------------------------------------


def test_calibration_standard_threshold_applies() -> None:
    """Moderate offset + good improvement → applies."""
    assert _should_apply_calibration(offset_s=0.08, improvement=0.02, best_corr=0.15) is True


def test_calibration_standard_threshold_rejects_low_improvement() -> None:
    """Moderate offset + low improvement → rejects."""
    assert _should_apply_calibration(offset_s=0.08, improvement=0.005, best_corr=0.15) is False


def test_calibration_large_offset_measurable_improvement_applies() -> None:
    """Large offset (>=100ms) + measurable improvement (>=0.005) → applies."""
    assert _should_apply_calibration(offset_s=0.14, improvement=0.006, best_corr=0.31) is True


def test_calibration_large_offset_noise_improvement_rejects() -> None:
    """Large offset + noise-level improvement (<0.005) → rejects (Arctic Monkeys case)."""
    assert _should_apply_calibration(offset_s=0.14, improvement=0.001, best_corr=0.31) is False


def test_calibration_large_offset_zero_improvement_rejects() -> None:
    """Large offset + zero improvement → rejects."""
    assert _should_apply_calibration(offset_s=0.14, improvement=0.0, best_corr=0.31) is False


def test_calibration_large_offset_negative_improvement_rejects() -> None:
    """Large offset + negative improvement → rejects."""
    assert _should_apply_calibration(offset_s=0.14, improvement=-0.01, best_corr=0.31) is False


def test_calibration_low_best_corr_rejects() -> None:
    """Low best_corr → rejects even with large offset."""
    assert _should_apply_calibration(offset_s=0.14, improvement=0.005, best_corr=0.05) is False


def test_calibration_tiny_offset_always_rejects() -> None:
    """Tiny offset (<30ms) → always rejects regardless of improvement."""
    assert _should_apply_calibration(offset_s=0.02, improvement=0.05, best_corr=0.30) is False


def test_onset_snap_returns_per_word_details(tmp_path: Path) -> None:
    """_snap_words_to_vocal_onset (v6, onset_detect) returns details with per-word info."""
    import numpy as np
    import soundfile as sf

    segments = [
        {
            "text": "Hello world",
            "words": [
                {"word": "Hello", "start_s": 0.0, "end_s": 1.0},
                {"word": "world", "start_s": 1.0, "end_s": 2.0},
            ],
        },
    ]

    # Build a WAV with sine-wave bursts so onset_detect reliably triggers.
    # Word "Hello" [0.0-1.0]: silence until 0.5s, then 440 Hz burst → should snap forward
    # Word "world" [1.0-2.0]: 440 Hz from the start → onset at ~1.0s, no forward snap
    sr = 22050
    y = np.zeros(sr * 2, dtype=np.float32)
    t_burst = np.arange(int(0.5 * sr), int(1.0 * sr)) / sr
    y[int(0.5 * sr):int(1.0 * sr)] = 0.5 * np.sin(2 * np.pi * 440 * t_burst).astype(np.float32)
    t_word2 = np.arange(int(1.0 * sr), int(2.0 * sr)) / sr
    y[int(1.0 * sr):int(2.0 * sr)] = 0.5 * np.sin(2 * np.pi * 440 * t_word2).astype(np.float32)

    audio_path = tmp_path / "test.wav"
    sf.write(str(audio_path), y, sr)

    result = _snap_words_to_vocal_onset(segments, audio_path)

    assert result["snap_method"] == "onset_detect"
    assert "total_onsets" in result
    assert "details" in result
    assert isinstance(result["details"], list)
    assert result["snapped_words"] == len(result["details"])
    # "Hello" starts in silence → should be snapped forward to onset
    if result["snapped_words"] > 0:
        detail = result["details"][0]
        assert "word" in detail
        assert "seg_idx" in detail
        assert "original_start_s" in detail
        assert "snapped_start_s" in detail
        assert "delta_s" in detail
        assert detail["delta_s"] > 0


def test_onset_snap_continuous_singing(tmp_path: Path) -> None:
    """Two back-to-back tones: second word snaps forward to spectral change."""
    import numpy as np
    import soundfile as sf

    sr = 22050
    duration = 2.0
    n_samples = int(sr * duration)
    y = np.zeros(n_samples, dtype=np.float32)
    # First tone: 220 Hz from 0.0 to 1.0
    t1 = np.arange(0, int(1.0 * sr)) / sr
    y[:int(1.0 * sr)] = 0.5 * np.sin(2 * np.pi * 220 * t1).astype(np.float32)
    # Second tone: 440 Hz from 1.0 to 2.0 (spectral change at boundary)
    t2 = np.arange(int(1.0 * sr), n_samples) / sr
    y[int(1.0 * sr):] = 0.5 * np.sin(2 * np.pi * 440 * t2).astype(np.float32)

    # Place second word slightly before the actual frequency change
    segments = [
        {
            "text": "one two",
            "words": [
                {"word": "one", "start_s": 0.0, "end_s": 0.95},
                {"word": "two", "start_s": 0.95, "end_s": 2.0},
            ],
        },
    ]

    audio_path = tmp_path / "continuous.wav"
    sf.write(str(audio_path), y, sr)

    result = _snap_words_to_vocal_onset(segments, audio_path)
    assert result["snap_method"] == "onset_detect"
    assert result["total_onsets"] >= 1


def test_onset_snap_no_onset_in_window(tmp_path: Path) -> None:
    """Steady tone with no spectral change near word boundary → 0 snapped."""
    import numpy as np
    import soundfile as sf

    sr = 22050
    duration = 2.0
    n_samples = int(sr * duration)
    t = np.arange(n_samples) / sr
    # Continuous 440 Hz tone — no spectral transitions
    y = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    segments = [
        {
            "text": "steady tone",
            "words": [
                {"word": "steady", "start_s": 0.5, "end_s": 1.0},
                {"word": "tone", "start_s": 1.0, "end_s": 1.5},
            ],
        },
    ]

    audio_path = tmp_path / "steady.wav"
    sf.write(str(audio_path), y, sr)

    result = _snap_words_to_vocal_onset(segments, audio_path)
    assert result["snap_method"] == "onset_detect"
    assert result["snapped_words"] == 0


def test_onset_snap_rms_legacy_preserved(tmp_path: Path) -> None:
    """Legacy _snap_words_to_vocal_onset_rms is importable and returns valid dict."""
    import numpy as np
    import soundfile as sf

    sr = 22050
    y = np.zeros(sr * 2, dtype=np.float32)
    y[int(0.5 * sr):int(2.0 * sr)] = 0.5

    segments = [
        {
            "text": "test",
            "words": [{"word": "test", "start_s": 0.0, "end_s": 2.0}],
        },
    ]

    audio_path = tmp_path / "legacy.wav"
    sf.write(str(audio_path), y, sr)

    result = _snap_words_to_vocal_onset_rms(segments, audio_path)
    assert isinstance(result, dict)
    assert "snapped_words" in result
    assert "details" in result


# ---------------------------------------------------------------------------
# _initial_phoneme_class
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("word,expected", [
    # plosives
    ("blue", "plosive"),
    ("time", "plosive"),
    # affricates
    ("child", "affricate"),
    ("jump", "affricate"),
    # fricatives
    ("sing", "fricative"),
    ("the", "fricative"),   # "th" exception
    # nasals
    ("moon", "nasal"),
    ("know", "nasal"),      # "kn" exception
    # approximants
    ("run", "approximant"),
    ("write", "approximant"),  # "wr" exception
    # vowels
    ("apple", "vowel"),
    ("psalm", "vowel"),     # "ps" exception
    # 'c' context-sensitive
    ("city", "fricative"),  # c+i → fricative
    ("come", "plosive"),    # c+o → plosive
])
def test_phoneme_class_basic(word: str, expected: str) -> None:
    assert _initial_phoneme_class(word) == expected, f"{word!r} → expected {expected!r}"


def test_phoneme_class_punctuation_stripped() -> None:
    assert _initial_phoneme_class('"apple"') == "vowel"
    assert _initial_phoneme_class("---") == "unknown"


def test_snap_uses_phoneme_lead_in(tmp_path: Path) -> None:
    """Plosive word gets larger lead_in_s than vowel word in snap details."""
    import numpy as np
    import soundfile as sf

    sr = 22050
    # Two sine bursts at known positions: 0.5s and 1.5s
    y = np.zeros(int(sr * 3.0), dtype=np.float32)
    burst_len = int(0.3 * sr)
    y[int(0.5 * sr): int(0.5 * sr) + burst_len] = 0.6 * np.sin(
        2 * np.pi * 440 * np.arange(burst_len) / sr
    ).astype(np.float32)
    y[int(1.5 * sr): int(1.5 * sr) + burst_len] = 0.6 * np.sin(
        2 * np.pi * 880 * np.arange(burst_len) / sr
    ).astype(np.float32)

    # Place words slightly before their burst so snapping has room to move them.
    segments = [
        {
            "text": "blue apple",
            "words": [
                {"word": "blue",  "start_s": 0.42, "end_s": 0.95, "phones": []},
                {"word": "apple", "start_s": 1.42, "end_s": 1.95, "phones": []},
            ],
        }
    ]

    audio_path = tmp_path / "bursts.wav"
    sf.write(str(audio_path), y, sr)

    result = _snap_words_to_vocal_onset(segments, audio_path)
    assert result["snap_method"] == "onset_detect"

    details_by_word = {d["word"]: d for d in result["details"]}

    # phones should be populated on the word dicts
    words = segments[0]["words"]
    assert words[0]["phones"] == [{"class": "plosive", "source": "text_heuristic"}]
    assert words[1]["phones"] == [{"class": "vowel",   "source": "text_heuristic"}]

    # If both words were snapped, plosive lead_in > vowel lead_in
    if "blue" in details_by_word and "apple" in details_by_word:
        assert details_by_word["blue"]["lead_in_s"] > details_by_word["apple"]["lead_in_s"]
        assert details_by_word["blue"]["phoneme_class"] == "plosive"
        assert details_by_word["apple"]["phoneme_class"] == "vowel"
