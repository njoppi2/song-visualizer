# Roadmap

## v0 — “First video out”
Goal: prove the pipeline end-to-end with basic audio features (no stem separation yet).
- Input: local audio file (flac/mp3/wav)
- Analysis:
  - tempo + beat times
  - overall loudness envelope
  - onset strength / spectral flux proxy
- Render:
  - simple abstract visuals
  - beat flashes, intensity-driven motion
- Output: MP4 with original audio

Definition of Done:
- `python -m songviz render <audio> --out <mp4>` works on a normal laptop
- creates `analysis.json` next to video in outputs folder

## v1 — “Stem-aware layers”
Goal: separate and drive visuals per stem.
- Add stem separation (Demucs first choice)
- Compute per-stem energy envelopes
- Render layered visuals (vocals/drums/bass/other each controls a layer)

## v2 — “Song structure (scenes)”
Goal: section boundaries -> scene changes.
- Structural segmentation: A/B sections, intro/chorus-like boundaries
- Scene system:
  - each section gets a visual preset
  - scene transitions (color palette shifts, motion style changes)

## v3 — “Anticipation + drops”
Goal: “pre-drop buildup” and “drop state”.
- Heuristics based on rising intensity, onset density, downbeats
- Storyboard events:
  - buildup_start, buildup_peak, drop_hit
- Visual grammar: introduce elements during buildup, explode on drop

## v4 — “Kinetic typography”
Goal: animated text synced with the song.
- Use lyrics/transcript (optional input)
- Kinetic typography layer (timed words/phrases)
- Scene-aware typography styles

## v5+ — “Meaning”
Goal: section summaries and semantic-driven visuals.
- lyric alignment + section summarization
- visual motifs tied to lyrical themes/emotions
