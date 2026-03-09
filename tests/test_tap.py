from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from songviz.tap import TapResult, taps_to_corrections, write_corrections


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_alignment(segments: list[dict] | None = None) -> dict:
    if segments is None:
        segments = [
            {
                "text": "Hello world foo",
                "start_s": 1.0,
                "end_s": 4.0,
                "words": [
                    {"word": "Hello", "start_s": 1.0, "end_s": 1.5, "confidence": 0.9, "phones": []},
                    {"word": "world", "start_s": 1.6, "end_s": 2.5, "confidence": 0.85, "phones": []},
                    {"word": "foo", "start_s": 2.6, "end_s": 3.8, "confidence": 0.8, "phones": []},
                ],
            },
            {
                "text": "bar baz",
                "start_s": 5.0,
                "end_s": 8.0,
                "words": [
                    {"word": "bar", "start_s": 5.0, "end_s": 6.0, "confidence": 0.9, "phones": []},
                    {"word": "baz", "start_s": 6.5, "end_s": 7.5, "confidence": 0.88, "phones": []},
                ],
            },
        ]
    return {"metadata": {"song_id": "test"}, "segments": segments}


# ---------------------------------------------------------------------------
# taps_to_corrections
# ---------------------------------------------------------------------------


def test_taps_to_corrections_all_tapped() -> None:
    alignment = _make_alignment()
    taps = [1.1, 1.7, 2.8, 5.2, 6.6]  # one tap per word

    corrections = taps_to_corrections(alignment, taps)
    assert len(corrections) == 2

    # All segments should be corrected.
    assert corrections[0]["status"] == "corrected"
    assert corrections[1]["status"] == "corrected"

    # First word: start_s = tap[0]
    assert corrections[0]["words"][0]["start_s"] == pytest.approx(1.1, abs=0.001)
    # Original preserved.
    assert corrections[0]["words"][0]["original_start_s"] == pytest.approx(1.0, abs=0.001)

    # All 5 words tapped.
    total_words = sum(len(p["words"]) for p in corrections)
    assert total_words == 5


def test_taps_to_corrections_partial() -> None:
    alignment = _make_alignment()
    # Only tap 2 out of 5 words.
    taps = [1.1, 1.7]

    corrections = taps_to_corrections(alignment, taps)

    # First segment is corrected (has tapped words).
    assert corrections[0]["status"] == "corrected"
    # Second segment has no taps → auto.
    assert corrections[1]["status"] == "auto"

    # Un-tapped words keep original times.
    assert corrections[1]["words"][0]["start_s"] == pytest.approx(5.0, abs=0.001)
    assert corrections[1]["words"][0]["original_start_s"] == pytest.approx(5.0, abs=0.001)


def test_taps_to_corrections_end_time_gap() -> None:
    """Consecutive tapped words: end = next_tap - 0.05."""
    alignment = _make_alignment()
    taps = [1.1, 1.7, 2.8, 5.2, 6.6]

    corrections = taps_to_corrections(alignment, taps)

    # First word end = tap[1] - 0.05 = 1.65.
    assert corrections[0]["words"][0]["end_s"] == pytest.approx(1.65, abs=0.001)
    # Second word end = tap[2] - 0.05 = 2.75.
    assert corrections[0]["words"][1]["end_s"] == pytest.approx(2.75, abs=0.001)


def test_taps_to_corrections_last_word_uses_segment_end() -> None:
    """Last word in a segment uses segment.end_s."""
    alignment = _make_alignment()
    taps = [1.1, 1.7, 2.8, 5.2, 6.6]

    corrections = taps_to_corrections(alignment, taps)

    # Last word in segment 0 (index 2, "foo"): end_s = segment end = 4.0.
    assert corrections[0]["words"][2]["end_s"] == pytest.approx(4.0, abs=0.001)
    # Last word in segment 1 (index 1, "baz"): end_s = segment end = 8.0.
    assert corrections[1]["words"][1]["end_s"] == pytest.approx(8.0, abs=0.001)


def test_taps_to_corrections_clamp_end() -> None:
    """End time is clamped to at least start + 20ms."""
    alignment = _make_alignment([{
        "text": "a b",
        "start_s": 0.0,
        "end_s": 2.0,
        "words": [
            {"word": "a", "start_s": 0.0, "end_s": 0.5, "confidence": 0.9, "phones": []},
            {"word": "b", "start_s": 0.5, "end_s": 1.0, "confidence": 0.9, "phones": []},
        ],
    }])
    # Taps very close together: tap[1] - 0.05 < tap[0].
    taps = [1.0, 1.01]

    corrections = taps_to_corrections(alignment, taps)
    # First word: end = max(1.01 - 0.05, 1.0 + 0.02) = max(0.96, 1.02) = 1.02.
    assert corrections[0]["words"][0]["end_s"] >= corrections[0]["words"][0]["start_s"] + 0.02


# ---------------------------------------------------------------------------
# run_tap_session (mocked)
# ---------------------------------------------------------------------------


def test_run_tap_session_mocked() -> None:
    """Mock terminal and ffplay to verify taps are recorded."""
    from songviz.tap import run_tap_session

    alignment = _make_alignment([{
        "text": "Hello",
        "start_s": 0.0,
        "end_s": 2.0,
        "words": [
            {"word": "Hello", "start_s": 0.0, "end_s": 1.0, "confidence": 0.9, "phones": []},
        ],
    }])

    mock_proc = MagicMock()
    mock_proc.poll.side_effect = [None, None, None]
    mock_proc.kill.return_value = None
    mock_proc.wait.return_value = 0

    mock_term = MagicMock()
    # First call: key available, return space. Second call: song ends.
    mock_term.key_available.side_effect = [True, False]
    mock_term.read_key.return_value = " "
    mock_term.__enter__ = lambda s: s
    mock_term.__exit__ = MagicMock(return_value=False)

    with (
        patch("songviz.tap.subprocess.Popen", return_value=mock_proc),
        patch("songviz.tap._RawTerminal", return_value=mock_term),
        patch("builtins.input", return_value=""),
    ):
        result = run_tap_session("/fake/audio.flac", alignment, ffplay_path="/usr/bin/ffplay")

    assert result.tapped_count == 1
    assert len(result.taps) == 1
    assert result.total_words == 1


# ---------------------------------------------------------------------------
# require_ffplay
# ---------------------------------------------------------------------------


def test_require_ffplay_found() -> None:
    from songviz.ffmpeg import require_ffplay

    with patch("songviz.ffmpeg.shutil.which", return_value="/usr/bin/ffplay"):
        assert require_ffplay() == "/usr/bin/ffplay"


def test_require_ffplay_not_found() -> None:
    from songviz.ffmpeg import require_ffplay

    with (
        patch("songviz.ffmpeg.shutil.which", return_value=None),
        patch("songviz.ffmpeg.Path.exists", return_value=False),
        pytest.raises(RuntimeError, match="ffplay was not found"),
    ):
        require_ffplay()


# ---------------------------------------------------------------------------
# write_corrections
# ---------------------------------------------------------------------------


def test_write_corrections_creates_file(tmp_path: Path) -> None:
    alignment = _make_alignment()
    corrections = taps_to_corrections(alignment, [1.1, 1.7])

    result = write_corrections(tmp_path, alignment, corrections)
    assert result.exists()

    import yaml

    data = yaml.safe_load(result.read_text())
    assert data["source"] == "tap_along"
    assert len(data["phrases"]) == 2
