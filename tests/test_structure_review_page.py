"""Static contract checks for the self-contained structure review template."""

from pathlib import Path


PAGE = Path(__file__).parents[1] / "experiments" / "templates" / "structure_review.html"


def test_structure_review_embeds_contract_and_manifest_placeholders():
    html = PAGE.read_text(encoding="utf-8")

    assert '<script id="review-data" type="application/json">{{REVIEW_JSON}}</script>' in html
    assert "const manifestHash = '{{MANIFEST_SHA}}';" in html
    assert "original.wav" in html
    assert "songviz-structure-feedback.json" in html
    assert "review.audio_path" in html
    assert "resolved.origin!==location.origin" in html
    assert "audio.src=nativeAudioUrl();audio.load()" in html
    assert "fetch(" not in html
    assert "audioSources" not in html


def test_structure_review_covers_review_controls_and_diagnostic_links():
    html = PAGE.read_text(encoding="utf-8")

    for required in (
        "boundary_questions",
        "repeat_questions",
        "corrected_time_s",
        "marked_observations",
        "global_notes",
        "sections-comparison.png",
        "novelty-comparison.png",
        "evidence.png",
        "Play A",
        "Play B",
        "Mark current time",
        "Download feedback JSON",
        "Acoustic similarity score",
    ):
        assert required in html

    assert "innerHTML" not in html
    assert "value:'unreviewed'" in html
    assert "setTimeout(()=>{if(state.programmaticSeekTarget" not in html


def test_bounded_segment_seek_guards_against_late_native_seeking_event():
    html = PAGE.read_text(encoding="utf-8")

    # The target is installed before stopAt, so either synchronous or queued
    # seeking events can identify this one programmatic seek and preserve the
    # bounded stop. A later unmatched seek still cancels it.
    assert "state.programmaticSeekTarget=range.start;" in html
    assert html.index("state.programmaticSeekTarget=range.start;") < html.index("state.stopAt=range.end;")
    assert "Math.abs((Number(audio.currentTime)||0)-target)<0.05" in html
    assert "state.programmaticSeekTarget=null;\n  cancelBoundedPlayback();" in html


def test_structure_review_uses_native_range_audio_and_actionable_errors():
    html = PAGE.read_text(encoding="utf-8")

    assert "url.searchParams.set('v',manifestHash)" in html
    assert "url.searchParams.set('attempt',String(state.audioAttempt))" in html
    assert "audio.seekable" in html
    assert "MEDIA_ERR_NETWORK" in html
    assert "audio.error.message" in html
    assert "Notes and answers remain editable." in html
