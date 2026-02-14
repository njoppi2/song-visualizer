# Codex Operating Instructions (SongViz)

You are working inside a git repository. Work incrementally and leave the repo in a runnable state.

## Ground rules
- Implement the roadmap sequentially (v0 first).
- Make small commits with clear messages.
- Update `docs/03_working_state.md` after each commit.
- Do not add large binary files to git. Do not commit copyrighted audio.
- Keep generated outputs under `outputs/` (already gitignored).
- Prefer standard, widely available dependencies.
- Provide a CLI that works from repo root.

## v0 target
Implement:
- CLI: `python -m songviz render <audio> --out <mp4>`
- Analysis written to `outputs/<song_id>/analysis.json`
- Video rendering: abstract visuals responding to beat times + loudness + onset strength
- Add basic unit tests for analysis array shapes and JSON schema keys

## Tooling preferences
- Use librosa for beat tracking and features.
- Use Pillow (or pure numpy -> Pillow) to render frames.
- Use ffmpeg (via subprocess) to mux audio + frames into mp4.
- Make the render deterministic with a seed option.

## Suggested commit sequence
1) Scaffold package + CLI + README updates
2) Implement audio decode/load utilities
3) Implement analyze stage + write analysis.json
4) Implement render stage + MP4 export
5) Tests + polish + docs updates
