from __future__ import annotations

import datetime
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from songviz.lyrics import (
    align_lyrics,
    load_alignment,
    lyric_activity_at,
    lyric_signals_for_timeline,
)
from songviz.paths import lyrics_alignment_path_for_output_dir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_alignment_payload(song_id: str = "test123") -> dict:
    """Return a minimal valid alignment.json payload (contract-compliant)."""
    return {
        "metadata": {
            "song_id": song_id,
            "language": "en",
            "alignment_tool": "whisper",
            "whisper_model": "base",
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
    assert result["active_segment"] == "Hello world"
    assert result["word_confidence"] == pytest.approx(0.9)
    assert 0.0 <= result["word_progress"] <= 1.0


def test_lyric_activity_at_second_word(tmp_path: Path) -> None:
    alignment = _make_alignment_payload()
    result = lyric_activity_at(alignment, t=1.5)  # inside "world" [1.2, 2.0)
    assert result["active_word"] == "world"
    assert result["word_confidence"] == pytest.approx(0.85)


def test_lyric_activity_at_between_words(tmp_path: Path) -> None:
    alignment = _make_alignment_payload()
    # t=1.1 is between "Hello" [0.5,1.0) and "world" [1.2,2.0), still inside segment [0.5,2.0)
    result = lyric_activity_at(alignment, t=1.1)
    assert result["active_word"] == ""
    assert result["active_segment"] == "Hello world"
    assert result["word_confidence"] == 0.0


def test_lyric_activity_at_outside_all_segments(tmp_path: Path) -> None:
    alignment = _make_alignment_payload()
    result = lyric_activity_at(alignment, t=5.0)
    assert result["active_word"] == ""
    assert result["active_segment"] == ""
    assert result["word_confidence"] == 0.0
    assert result["word_progress"] == 0.0


def test_lyric_activity_at_empty_alignment(tmp_path: Path) -> None:
    result = lyric_activity_at({"segments": []}, t=0.5)
    assert result["active_word"] == ""


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
