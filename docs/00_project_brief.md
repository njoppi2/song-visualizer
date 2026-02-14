# SongViz — Visualize Music as a Story

## One-liner
SongViz turns a song into a generated video by analyzing audio structure (beats, sections, stems) and driving a visual “storyboard” over time.

## Why
Most music visualizers are just waveforms/spectra. SongViz aims to feel like the visuals *understand* the song:
- builds tension before a drop
- changes scenes on chorus/verse boundaries
- optionally uses lyrics/meaning later (future versions)

## Core concept
Everything becomes a timeline:
- Features: loudness, stem energy, onset strength, spectral descriptors
- Events: beats/downbeats, section boundaries, drop candidates
- Scenes: rules that map features/events -> visuals

The renderer only asks: “At time t, what scene are we in and what values drive it?”

## Constraints / principles
- MVP should be runnable from a single CLI command and produce an MP4.
- Favor deterministic, reproducible outputs (seeded RNG).
- Keep analysis outputs as JSON for debuggability and future tooling.
- Layered architecture: separation -> analysis -> storyboard -> render.

## Non-goals (for now)
- Real-time rendering
- Perfect “semantic meaning” without lyrics
- GPU-only dependencies required for basic run (nice if available, not required)

## Success criteria (MVP)
Given `song.flac` (local file), the tool produces:
- outputs/<song_name>/analysis/analysis.json
- outputs/<song_name>/video.mp4
And the video visibly reacts to beats + stem energies + section changes.
