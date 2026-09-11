# Working State — implementation inventory and history

**Start/resume at [CONTINUE.md](../CONTINUE.md).** It is the canonical current
checkpoint and next-task entry point. This file retains the detailed inventory
and prior results; use [the roadmap](01_roadmap.md) for milestones.

**September checkpoint:** the [four-policy comparison](18_local_structure_comparison.md)
is complete, with 16/39/69/79 change proposals and no production promotion.
The [guided listening feedback](20_listening_feedback.md) informs the
[multiscale response experiment](21_change_episodes.md). Use CONTINUE.md for
current verification, artifacts and the next bounded task.

**Earlier directing milestone:** the analysis/render foundation is implemented.
A rule-based saved plan/render/replay exists for a 48-second cached passage, not
a full-song or LLM director. Its musical emphasis still needs user review;
Milestone 2 remains open. See [10_directing_prototype.md](10_directing_prototype.md).

The implementation inventory and results below retain earlier development notes;
they are not a fresh runtime or perceptual certification. Historical phase numbers
do not define the current queue. In this restart review, the existing test suite
reported 329 passed and 1 skipped. A subsequent isolated full-song structural
regeneration is documented in `12_structure_grid.md`; listening quality has not
been revalidated. A reference derived from the algorithm being evaluated is
diagnostic evidence, not independent ground truth.

## Start here

- Project roadmap and phases: `docs/01_roadmap.md`
- Target architecture, implemented boundary, and open decisions: `docs/02_architecture.md`
- Runtime status and commands: this file
- Repo and module map: `docs/04_repo_reference.md`
- Reduced-representation design (historical): `docs/06_reduced_representation.md`
- Canonical lyrics implementation path: `docs/05_lyrics_playbook.md`
- Lyrics research notes (non-default): `docs/research/lyrics_syncing_research.md`

## What is implemented
- Python package scaffold: `songviz/` with module entrypoint (`python -m songviz`).
- CLI:
  - `songviz analyze <audio>` writes `outputs/<song_name>/analysis/analysis.json`.
  - `songviz render <audio> --out outputs/demo.mp4` writes:
    - `outputs/<song_name>/analysis/analysis.json`
    - `outputs/<song_name>/analysis/story.json`
    - `outputs/<song_name>/video.mp4` (always)
    - plus a copy/hardlink at `--out` if provided
  - `songviz stems <audio>` writes `outputs/<song_name>/stems/{drums,bass,vocals,other}.wav` and `outputs/<song_name>/stems/stems.json` (requires Demucs).
  - `songviz ui` provides a simple interactive picker for songs in `songs/` to render/regenerate videos.
  - Shared render orchestration extracted into `songviz/pipeline.py` (used by both cli and ui).
- Packaging via `pyproject.toml` with an optional console script: `songviz ...` after `pip install -e .`.
- Analysis:
  - tempo + beat times
  - loudness (RMS) envelope normalized to [0,1]
  - onset strength normalized to [0,1]
- Story:
  - coarse section segmentation with **functional role-based labels**: each section is assigned a role (`intro`, `build`, `payoff`, `valley`, `contrast`, `outro`) based on energy flow, position, and repetition features. Letters (A/B/C) are derived from role + acoustic similarity. Boundary detection (SSM + checkerboard novelty + tension valleys) unchanged.
  - **SSM + checkerboard novelty** segmentation (primary): beat-synchronous chroma_cqt + MFCC, self-similarity matrix with path enhancement, checkerboard kernel novelty curve, adaptive peak-picking. Fallbacks: tension valley boundaries (if ≤2 sections for songs >90s), then agglomerative clustering (on exception).
  - **Role assignment pipeline**: `_compute_section_features` (11 features per section, min-max normalized), `_assign_roles` (weighted scoring per role with eligibility constraints), `_revise_roles_globally` (sequence constraints: intro/outro position, build→payoff proximity, repeated-section consistency, payoff guarantee), `_resolve_visual_behavior` (role→visual mapping with context, e.g. first payoff = `release_payoff`, subsequent = `sustain_euphoria`), `_assign_role_based_labels` (cluster within same-role groups by cosine similarity).
  - Section output includes: `role`, `visual_behavior`, `confidence`, `intensity`, `repetition_strength`, `novelty_to_prev`, `relative_intensity_rank`, plus existing `label`, `start_s`, `end_s`, `subsections`.
  - per-frame `tension` curve aligned to envelope frames
  - `drop_times_s` event detection (wired to renderer: full-frame flash on drop)
  - **buildup detection**: buildup windows exported to story + renderer shows extra rays and intensity bar during buildups
  - `min_len_s` 12 s (SSM path) / 15 s (agglomerative fallback); adjacent sections merged via `_merge_same_label_sections` when same role AND low novelty across boundary (< 0.3); drop detection requires ≥ 0.6 peak tension in preceding 2 s.
  - **subsections**: each section contains `subsections[]` with finer-grained energy regions (detected via tension valleys within each section, 4 s smoothing, min 8 s subsection length). Each subsection has an energy descriptor: `low`, `mid`, `high`, `rising`, or `falling`.
- Renderer:
  - 30 fps (default)
  - visuals driven by loudness + onset + beat flashes
  - **section crossfade** at boundaries (1.5 s smoothstep blend)
  - deterministic with `--seed`
  - ffmpeg muxes original audio into MP4
  - `--audio-codec aac|mp3` (default `mp3`)
  - `--layout stems4` renders a 2x2 stem grid with stem-specific visuals
  - `_VisualizerBase` base class shared by `Visualizer` (mix) and `StemQuadVisualizer` (stems4)
  - Drop flash is now a two-phase sharp white spike (decay 60 ms) + accent afterglow (decay 300 ms); section boundaries have a 150 ms color wash in the incoming palette; buildup fraction drives orb radius swell (+12% at peak).
  - **Timeline bar**: thin 24 px bar at the top of every frame (mix + stems4). Colored blocks per section (palette `bot` color), section labels, section boundary lines, and a 3 px white playhead + downward triangle. Subsection dividers removed. Implemented via `_VisualizerBase._draw_timeline_bar(draw, t, w, h)`.
- Minimal pytest coverage for analysis keys/array lengths and `song_id` stability.
- Analysis visualization:
  - `songviz analyze <audio>` and `songviz render <audio>` now auto-generate `analysis/overview.png` (2-panel dark-theme: envelopes + section timeline) and `analysis/README.md` (metadata table, sections, signal glossary, re-generate instructions).
  - `analysis/stems_overview.png` is also written when `stems/*.wav` exist (stem RMS envelopes + per-section heatmap).
  - Requires optional dep: `pip install -e '.[viz]'` (pulls `matplotlib`); README is always written even without it.
  - Module: `songviz/viz.py` — public API: `generate_overview`, `generate_stems_overview`, `generate_analysis_readme`, `generate_all`.
- Lyrics alignment + render integration:
  - `songviz lyrics <audio>` writes `outputs/<song_id>/lyrics/alignment.json`.
  - Backend-based (whisper/whisperx word-level timestamps + confidence); uses vocals stem when present.
  - Automatic global timing calibration is enabled by default (large offsets ≥100ms require improvement ≥0.005 to filter noise-level correlation changes).
  - Pipeline v7: phoneme-aware `lead_in_s` — `_initial_phoneme_class()` maps word-initial letter(s) to articulatory class; per-class lead-in from `_PHONEME_CLASS_LEAD_IN_S` (5–55ms). Populates `w["phones"]` with `[{"class":…,"source":"text_heuristic"}]`. Details include `phoneme_class` and `lead_in_s` per snapped word. v6 onset_detect-based snapping (spectral flux via `librosa.onset.onset_detect`) retained as base; v5 RMS function preserved as `_snap_words_to_vocal_onset_rms` for rollback.
  - `lyrics.load_alignment()` / `lyrics.lyric_activity_at()` / `lyrics.lyric_signals_for_timeline()` expose derived signals.
  - `songviz render <audio> --lyrics` loads `alignment.json` and renders a **full-line overlay**: active word in accent color, remaining words dimmed (mix layout: bottom-center; stems4: vocals quadrant bottom-center).
- `cli.py` has a local `_copy_or_link` helper (intentional — separate from `stems.py`'s internal copy logic).
- Drum hit extraction (`songviz/reduction.py`):
  - First piece of the reduced representation (`analysis/reduced.json`, `"drums"` key).
  - **Template path (primary)**: `extract_drum_hits_template` — beat-level groove detection. Computes per-beat RMS for each DrumSep component, applies relative-energy (deviation above 8-beat running mean) to strip reverb/bleed baseline, detects kick/snare phase via 2-phase energy scoring, places hits EXACTLY on beat times (zero timing jitter). Hi-hat added at every beat + midpoint ("and"). Source: `"template"`. Activity F1 improved from ~0.85 to 1.00 on Feel Good Inc.
  - DrumSep onset path (first fallback): per-component onset strength + peak-picking with tuned per-instrument parameters.
  - Heuristic fallback: onset detect on full drum stem + spectral band classification (kick <150Hz, snare 150–2500Hz, hh >2500Hz).
  - Helper functions: `_beat_rms(y, beat_arr, sr)` — per-beat RMS array; `_relative_energy(rms, window=8)` — deviation above running mean, normalized to [0,1].
  - Dual velocity: `velocity` (per-component normalized 0–1) for dynamics, `velocity_raw` (unnormalized RMS) for cross-component loudness.
  - Beat alignment: `beat_idx` + `beat_phase` [0.0, 1.0) preserving syncopation/offbeat info.
  - Auto-wired into `_build_stem_analyses()` in `pipeline.py`; writes `reduced.json` with read-merge-write pattern for future stem extensions.
- Vocal note extraction (`songviz/reduction.py`):
  - Second piece of the reduced representation (`analysis/reduced.json`, `"vocals"` key).
  - Primary: basic-pitch note events (from `vocals_note_events_basic_pitch`) → remap field names + beat alignment.
  - **basic-pitch ONNX fix**: `vocals_note_events_basic_pitch` now resolves the ONNX model path (`nmp.onnx`) and passes it via `model_or_model_path`, bypassing `tflite-runtime` incompatibility with numpy 2.x. Falls back to default (tflite) if ONNX model not found.
  - Fallback: pYIN pitch track → group consecutive same-MIDI frames into notes, RMS-based velocity with 99th-percentile normalization.
  - Schema: `onset_s`/`offset_s` (duration), `midi` (float, 2dp), `velocity` (normalized [0,1]), `beat_idx`/`beat_phase`.
  - Auto-wired into `_build_stem_analyses()` vocals block; read-merge-write into `reduced.json`.
- Bass note extraction (`songviz/reduction.py`):
  - Third piece of the reduced representation (`analysis/reduced.json`, `"bass"` key).
  - **Priority: torchcrepe > basic-pitch > pYIN**. Added `crepe_pitch_hz` parameter to `extract_bass_notes`.
  - **Primary: torchcrepe** (via `bass_pitch_crepe` in `features.py`) — neural F0 tracker using CREPE `full` model with `weighted_argmax` decoder, fmin=40 Hz, fmax=400 Hz (or 80 Hz in sub_bass_mode). Resamples audio to 16 kHz, confidence-gates frames with periodicity < 0.3, RMS-gates silence, quantizes to nearest semitone. 3× more voiced frames than pYIN (51% vs 17% coverage). Pitch-class accuracy jumped from 38.8% to 100% in-scale on Feel Good Inc. Note: CREPE `tiny` + Viterbi fails at sub-bass (periodicity = -inf); must use `full` + `weighted_argmax`. Helper: `_torchcrepe_available()` availability check.
  - **Secondary: basic-pitch** note events (from `bass_note_events_basic_pitch`) → same dispatcher pattern as vocals. Bass-specific thresholds: onset=0.50, frame=0.25, min_note=150ms, freq 30–400 Hz (exposed as `_BASS_BP_*` constants in `features.py`).
  - **Octave correction pipeline** (basic-pitch and pYIN paths, partial for crepe): (1) `_bass_global_octave_fix` shifts ALL notes ±12 when median MIDI is outside expected bass register (36–55); (2) `_refine_bass_pitch_cqt` per-note CQT harmonic ratio test — if octave-above has more energy than detected frequency, shift up 12 (sub-harmonic artifact fix); (3) `_correct_octave_by_context` shifts ±12 toward local median (window=5, min_gain=6). Benchmark: in_range_pct 50%→84%, below_range_pct 49%→11%, octave_jump_pct 6%→3%.
  - **Key estimation disabled**: `estimate_key_scale` (Krumhansl-Kessler profiles) and `_snap_bass_to_scale` exist but are NOT used in the pipeline — Demucs stems consistently yield wrong key estimates (e.g. F# major for G minor songs), and scale snapping with the wrong key degrades pitch-class accuracy. Re-enable when a reliable key estimator is available.
  - **Energy gating**: `_gate_and_prune_bass_notes` removes false-positive notes in near-silent stem regions (RMS threshold = 0.5 × 10th-percentile of nonzero per-note RMS); `_rescale_velocity_to_stem_energy` replaces basic-pitch confidence / pYIN self-normalized velocity with stem-energy-based velocity. Isolated weak notes (>4s gap to neighbors AND velocity <0.35) also pruned. Applied in all paths.
  - **Fallback: pYIN** pitch track → note events with gap-merge (`max_gap_frames=3`).
  - Same schema as vocals: `onset_s`/`offset_s`/`midi`/`velocity`/`beat_idx`/`beat_phase`.
  - `source`: `"crepe"`, `"basic_pitch"`, `"pyin"`, or `"none"`.
  - Auto-wired into `_build_stem_analyses()` bass block; read-merge-write into `reduced.json`.
- Sonifier (`songviz/sonify.py`):
  - `songviz sonify <audio>` reads `analysis/reduced.json` and writes `analysis/reduced.wav`.
  - Debug/validation tool: noise/sine bursts for drums (6 distinct templates), sine for vocals, triangle wave for bass.
  - Per-layer WAVs: `reduced_{drums,vocals,bass}_only.wav`, `reduced_vocals_plus_bass.wav`, `reduced_bass_only_up1oct.wav`.
  - `--diagnose` flag prints per-layer stats + warning heuristics. Bass diagnostics include `velocity_min/max/p10`, `isolated_note_count`, and warnings `many_isolated_bass_notes` / `bass_velocity_floor_high`.
  - No new deps — numpy + soundfile only.
- Evaluation framework (`songviz/eval.py`):
  - `songviz eval <audio>` compares `reduced.json` against human-curated reference annotations in `benchmark/references/`.
  - Three evidence levels: gold (synthetic ground truth), silver (tabs/listening), weak (rough annotations).
  - Metrics: section-level activity F1 (is the layer playing?), silent-region false positive count/rate, pitch range accuracy (% in expected MIDI range, below/above breakdown), onset matching P/R/F1 (when per-note references available), octave-invariant pitch-class analysis (in_scale_pct, root_pc_pct, cross-section consistency).
  - `benchmark/songs.json` maps song_id → reference subdirectory. 5 benchmark songs: feel-good-inc, do-i-wanna-know, feeling-this, shy-away, die-for-you.
  - `--json` flag outputs raw JSON for programmatic use. `--reference-dir` for custom references.
- Benchmark runner (`songviz/bench.py`):
  - `songviz bench --songs-dir songs` evaluates all benchmark songs, reports per-song + aggregate metrics.
  - `--save-baseline` saves results to `benchmark/baselines/baseline_<timestamp>.json` + `latest.json`.
  - `--baseline <path>` compares current results against saved baseline, flags regressions (exit code 1).
  - `--force-reduce` regenerates `reduced.json` from stems (skips cache).
  - `--json` outputs machine-readable JSON for agent automation.
  - Aggregate metrics: mean/min/max/n per metric per layer across all songs.

## Lyrics status
- **Implemented**: `songviz lyrics <audio>` runs the fallback chain below and writes `outputs/<song_id>/lyrics/alignment.json`.
- **Fallback chain** (best to worst):
  1. LRCLIB synced + **forced alignment** → `lrclib+stable_whisper_forced`: uses `stable_whisper.load_model().align()` to align the known LRCLIB text directly to audio, producing 1:1 word timestamps without lossy difflib matching.
  2. LRCLIB synced + backend transcribe+merge → `lrclib+stable_whisper_timing` / `lrclib+whisperx_timing` / `lrclib+whisper_timing`: backend transcribes independently, then difflib matches words to LRCLIB text (fallback when FA fails).
  3. LRCLIB synced + no backend → `lrclib_synced`: proportional word timing within each LRC line.
  4. LRCLIB plain lyrics + backend → `stable_whisper+lrclib_prompt` / `whisper+lrclib_prompt`.
  5. No metadata or no LRCLIB match → pure backend (`stable_whisper` / `whisper` / `whisperx`).
  6. Auto calibration applies a **pre-merge** global offset (for lrclib paths: calibrates raw backend output before merge so LRCLIB segment boundaries are preserved; for pure-whisper paths: applies post-merge).
- Default model: `small` (was `base`); default auto backend order: `whisperx` > `stable_whisper` > `whisper`.
- `stable-ts` (stable_whisper backend) now included in `.[lyrics]` dep — uses DTW on mel spectrogram for much more accurate word boundaries than vanilla Whisper attention weights.
- Uses vocals stem when present; falls back to full mix for backend paths.
- Requires optional dep: `pip install -e '.[lyrics]'` (openai-whisper + stable-ts + mutagen). `pip install -e '.[lyricsx]'` adds whisperx backend.
- Key public API in `songviz/lyrics.py`: `align_lyrics()`, `load_alignment()`, `lyric_activity_at()`, `lyric_signals_for_timeline()`.
- The output contract and full pipeline spec remain in `docs/05_lyrics_playbook.md`.
- Do not treat `docs/research/lyrics_syncing_research.md` as an execution default.
- **Render integration done**: `--lyrics` flag on `songviz render` draws active word as text overlay.
- **Manual corrections workflow**:
  - `songviz lyrics-tap <audio>` — tap-along session: play audio, press space at each word onset, writes corrections.yaml, auto-applies to alignment.json.
  - `songviz lyrics-template <audio>` — generate blank corrections.yaml for manual YAML editing.
  - `songviz lyrics-correct <audio>` — apply corrections.yaml + print quality stats (mean/median error, pct within 100ms/200ms, systematic offset).
  - `songviz lyrics-preview <audio>` — render fast lyrics-only video (960x270 @ 15fps) for timing verification.
  - Corrections auto-reapplied on `songviz lyrics --force` reruns.
  - Module: `songviz/tap.py` (tap session + mapping logic), corrections logic in `songviz/lyrics.py`.
- **Not yet done**: pYIN pitch summary, `lyrics-aligner` fallback (wav2vec2).

## Current priorities

- **Local-change/transition candidates implemented experimentally:**
  `docs/17_local_structure.md`; review at `outputs/reviews/local-structure-02/index.html`
  (port 8770). Fixed 2/4/8-beat contrasts produce 16 local changes; a separate
  dip-and-recovery hypothesis produces two intervals. Neither forms a new
  partition or supplies identity labels. Useful additional cuts coexist with
  missed chorus energy variations and partial/missing transition intervals;
  no production promotion. Next: separate channel/sustained-activity proposals,
  retaining this run as a control and all misses/burden as development evidence.
- **Separate structural evaluation implemented:** `docs/16_structural_evaluation.md`;
  ready report at `outputs/reviews/structure-evaluation-03/report.md`. Exact user
  spans/positive motif groups are separate from fingerprint-bound analyst
  variation/transition tags. Recurrence now exposes pattern and arrangement
  evidence plus local versus historical context, retaining all 12,898 legacy
  pair scores. Identity windows can cross adjacent same-motif variations without
  merging annotations; short-span/variation support gaps are explicit. The first
  shorter-scale proposal experiment is described above; automatic identity
  clustering and general transition recognition remain open. No production
  segmentation/render behavior or raw feedback was changed by this step.
- **User sections received and validated:** `benchmark/feedback/section-editor-02.json`;
  interpretation and exact provenance in `docs/15_section_feedback.md`. The user
  supplied 19 labeled spans with explicit chorus/verse-2 returns and intentional
  short transitions. Main mismatch: the detector misses internal changes and
  conflates arrangement/roles with musical identity. The separate representation
  and development evaluation are implemented as described above;
  do not merely tune toward a target count of 19. Raw feedback is preserved,
  not silently promoted to high-confidence benchmark ground truth.
- **User-defined sections remain the preferred source of guidance:** the free-form editor
  in `docs/14_section_annotation.md` starts without predicted boundaries and
  is ready at `outputs/reviews/section-editor-02/index.html` (port 8770). It
  supports independent named layers for broad parts and finer changes. Prefer
  the user's own labels/notes over forcing their interpretation into our detector's
  questions. The earlier A/B review remains optional evidence, not the required
  annotation workflow. Initial examples and an explicit, revisable analyst mapping
  are now available; formal hierarchy remains open.
- **Boundary/recurrence review implemented:** `docs/13_structure_review.md`.
  Open `outputs/reviews/structure-review-03/index.html` (port 8770, served from
  `outputs/reviews` using `python -m songviz.review_server`). Version 03 repairs
  playback with native ranged audio; questions, predictions and audio are
  unchanged from 02. The previous package remains preserved.
  Fresh story analysis now fuses agreeing SSM/energy candidates one-to-one and
  masks missing novelty history without per-lag/per-song rescaling. The isolated
  review compares previous and candidate logic on the same reviewed grid, with
  role-independent 16/32-beat passage pairs. Six versus seven sections is a
  controlled behavior change, not musical validation. The user later supplied
  free-form section feedback (docs 15–17); the earlier proposed-question review
  remains optional, not a blocker requiring another annotation round.
- **Structural beat-grid step completed:** audit in `docs/11_beat_grid_audit.md`,
  implementation and results in `docs/12_structure_grid.md`. `compute_story` now
  accepts an explicit shared grid and persists exact/effective timing hashes and
  fallback provenance. `outputs/reviews/structure-grid-01/` compares two fresh
  full-song runs under the same code with cached versus reviewed timing; all
  four stem diagnostics are present and original caches are unchanged. Eight
  versus seven sections is not evidence of improved musical correctness. Next,
  use the new audio-linked review for section boundaries and repeated passages before further tuning segmentation,
  recurrence or novelty. The historical story's internal grid remains unknown.
- **Current review:** `outputs/reviews/directed-review-01/index.html` (port 8768),
  Feel Good Inc 130–178s. An automatic rule policy chooses snare → other stem →
  vocals, with selective visibility and reusable treatments. Same evidence/audio
  drives the fixed comparison. The source-derived plan boundaries are not verified
  musical sections. Review focus/omissions, especially whether vocals should be
  foregrounded earlier within the sparse span, before expanding the policy.
- `songviz/direction.py` provides the version-1 contract and policy;
  `songviz/directed_render.py` executes it. The experiment command is
  `experiments/build_directed_review.py`, with saved-package `--replay` and an
  optional separate edited `--plan`. Existing production commands are unchanged.
  127 focused tests pass; exact audio equivalence, browser feedback/switching and
  byte-identical video replay were checked. Terra wrote the renderer/tests; Luna
  wrote the page/builder tests; the lead handled planning/evidence/integration.
- Full-song directing, model-backed planning, provider/budget settings, richer
  styles and phrase-level surprise decisions remain open. No runtime LLM calls,
  uploads, separation or beat refitting were introduced by this experiment.
- First artistic preview is ready at `outputs/reviews/visual-passage-01/index.html`
  (22s opening; local server port 8767). It uses the exact reviewed pulse and drum
  events, with original audio, and does not change production behavior. Terra
  implemented the visualizer/tests/page; the lead integrated, refined and verified
  the package. 98 focused tests pass; browser switching/export/retry and media
  integrity checks passed. See `docs/09_visual_passage.md`. The user likes the
  available musical information but clarified that the final video must direct
  attention and vary its visual language. This is not blanket artistic approval.
  Keep the preview as a vocabulary study; polishing it is no longer the next gate.
- Milestone 1 technical artifacts are prepared in `outputs/reviews/restart-02/`:
  three source-audio previews, optional stem/event audition, input snapshots,
  provenance manifest, and a feedback export page. See `docs/07_restart_review.md`
  for the reference audit and reproduction commands. User feedback is received
  verbatim in `benchmark/feedback/restart-02.json`; its manifest hash was verified.
- First response to that feedback: `outputs/reviews/rhythm-review-01/index.html`
  compares cached vs regular pulse with identical original audio and percussion.
  Waveform/spectrogram evidence is included. Experimental sidecars `beat_grid.py`
  and `percussion.py` do not alter the production tracker or cached reduced events.
  Current focused verification: 91 tests passed, all six videos checked for 60 FPS,
  expected duration, identical A/B audio, and exact source PCM cuts. Browser checks
  covered seeking, same-position switching, playback continuation, stem loading,
  and manifest-linked feedback export. Subsequent written user feedback supports
  regular-pulse alignment and recognizable percussion, with snare easiest to clap
  to. See `benchmark/feedback/rhythm-review-01.json`; dropdowns remained unreviewed,
  so this is not three explicit A/B votes. The opening's ambiguous pulse remark
  referred to the cached version, as confirmed in chat and preserved in
  `benchmark/feedback/rhythm-review-01-clarification.md`. Artistic approval and
  cross-song generalization remain pending.
- Batch section evaluation is now connected, with malformed-input reporting,
  explicitly diagnostic reference groups, and version-aware section comparisons.
  This evaluates historical cached output; it does not certify current extraction.
- The first cached batch evaluated all five benchmark songs without run errors;
  results and input snapshots are in `outputs/reviews/restart-02/evaluation/`.
  Targeted verification: 74 benchmark/evaluation tests plus 5 review/render tests
  passed. Browser playback, seeking, and feedback export were checked.
- Prioritize analysis failures that block the directed passage; exhaustive
  transcription or harmony remains unnecessary unless that example needs it.
- Record timestamped feedback so the user does not have to reconstruct old bugs.

### Restart inventory (2026-09-09, preliminary)

A read-only Luna worker found 11 source songs and reusable outputs for several
tracks. A first candidate is Gorillaz / Feel Good Inc. The lead confirmed these
local artifacts exist under `outputs/Gorillaz - Feel Good Inc (featuring De La Soul)/`:

- `analysis/reduced.wav` (filesystem date: April 12)
- `overview_video.mp4` (filesystem date: April 29)
- `analysis/story.json` (filesystem date: April 30)

Cached section boundaries near 137.86s and 165.67s suggest passages to inspect.
These are candidate review locations, not listening-verified transitions. The
different file dates do not establish matching provenance; confirm timeline and
generation settings before presenting them as a synchronized baseline. The first
review now renders directly from the cached JSON with sample-aligned source audio,
rather than reusing these older audio/video renders. `restart-01` is the initial
technical draft; `restart-02` is the selected review package.

## Historical progress and priorities (April 2026)

These dated measurements describe earlier runs, not verified current quality.

- **Milestone 2: Extraction quality** (substantially improved 2026-04-12):
  - Drums activity F1: ✓ 1.00 on Feel Good Inc (beat-level template extraction)
  - Bass pitch range: ✓ 100% in range (was 83.5% below range with basic-pitch)
  - Bass in-scale: ✓ 100% on Feel Good Inc (was 38.8% with basic-pitch)
  - Bass pitch-class root match: ✗ root still not detected correctly (F# dominates instead of Eb)
  - **Former next milestone**: M3 Harmony + Arrangement (superseded by the September roadmap)
- **Reduced representation** (Phase 4 — operational):
  - `songviz/reduction.py`: all three layers — drums (template → DrumSep onset → heuristic fallback), vocals (basic-pitch + pYIN + octave correction), bass (torchcrepe → basic-pitch → pYIN + octave correction + energy gating)
  - Output: `analysis/reduced.json` — unified file with `schema_version` and `"drums"`, `"vocals"`, `"bass"` keys
  - Wired into `pipeline.py` `_build_stem_analyses()` — auto-generates `reduced.json` during stems4 render
  - Sonifier done: `songviz sonify <audio>` → `analysis/reduced.wav` + per-layer debug WAVs + `--diagnose` stats
  - Beat quantization in `sonify.py`: `_quantize_for_sonification` snaps drum hits and bass onsets to 16th-note grid before synthesis
  - Benchmark: `songviz bench --songs-dir songs` runs eval across 5 songs with aggregate metrics
  - **Benchmark results (Feel Good Inc, 2026-04-12 — after template drums + torchcrepe bass)**:
    - **Drums**: activity_f1=1.00, onset_f1=0.234 (was ~0.05), pitch_accuracy=100% (template hits on beat)
    - **Bass**: activity_f1=0.75, in_range_pct=100% (was 83.5%), in_scale_pct=100% (was 38.8%), onset_f1=0.137
    - **Vocals**: activity_f1=0.86 (unchanged)
  - **Known limitation**: bass onset F1 remains low (0.137) — crepe pitch track segments don't align with MIDI reference onset times; timing comes from pitch-track segment boundaries, not note attacks. Root PC detection poor (F# dominates vs expected Eb).
  - Detailed plan: `docs/06_reduced_representation.md`
- Deepen lyrics integration (pipeline + render done; remaining):
  - pYIN pitch summary per word
  - `lyrics-aligner` fallback (wav2vec2, pip-installable)
- Story improvements (**substantially improved 2026-04-23**):
  - **Section boundary detection revamp** (all in `story.py`):
    - `_score_and_filter_boundaries()`: replaced naive union with scored intersection — SSM+energy agreed boundaries kept unconditionally; SSM-only kept if novelty peak ≥ 0.50; energy-only always kept. Rescue heuristic: if nothing survived in first 25% of song, rescues strongest SSM candidate with relaxed floor (novelty ≥ 0.20).
    - `_force_split_long_sections()`: any section > 45s gets split at its deepest tension valley; midpoint fallback removed (uniform-energy long sections left unsplit, preventing false splits on slow-build intros).
    - `_merge_short_segments` min_len_s lowered 12.0 → 8.0s to preserve genuine short intros (e.g. Feeling This 11.3s drum intro, SSM novelty=0.809).
    - `_detect_intro_onset_boundary()`: detects quiet-intro end (first-5s tension mean < 0.25) and injects the boundary AFTER the merge pass so it bypasses the 8s minimum. Fires only for songs with near-silent openings (e.g. Die For You 5.5s silent fade-in); returns None for loud starts (Feel Good Inc, Do I Wanna Know, Shy Away).
    - `_merge_same_label_sections` max_merged_len_s = 50.0 (from 30.0).
  - **Label improvements** (`_assign_roles()`):
    - Intro: bonus for first section with low relative-intensity-rank (quiet opener gets extra score); build duration penalty for short (<24-beat) first sections.
    - Build: quiet-island penalty — if rir < 0.01 (the single quietest section in the song, rank-minimum), build score is zeroed. Prevents windmill/bridge sections with rising internal slope from being mislabeled "build".
    - Contrast: energy-dip bonus (0.20×) when a section's mean_rms is below both neighbors.
    - Contrast position gate: ramps from 0 at sp=0 to full weight at sp=0.10 (contrast can't open a song).
    - `effective_ntp = ntp if i > 0 else 0.0`: first section uses sentinel ntp=1.0; zeroing prevents spurious contrast score on section 0.
  - **Section evaluation infrastructure**:
    - `benchmark/references/{feel-good-inc,do-i-wanna-know,feeling-this,shy-away,die-for-you}/sections.json`: ground truth annotations for all 5 benchmark songs.
    - `evaluate_sections()` in `eval.py`: boundary F1 @3s and @0.5s, over/under-segmentation ratio, pairwise frame-clustering F1.
    - `songviz eval` automatically runs section eval when `story.json` + `sections.json` exist (no extra flag needed).
  - **Benchmark results (all 5 songs, 2026-04-23)**:
    - Feel Good Inc (silver, 7 sections): F1@3s=1.000, F1@0.5s=0.667 (beat-quantization gap ≤0.7s); windmill correctly labeled valley.
    - Do I Wanna Know (silver, 8 sections): F1@3s=1.000, F1@0.5s=1.000.
    - Feeling This (silver, 6 sections): F1@3s=1.000, F1@0.5s=1.000.
    - Shy Away (silver, 7 sections): F1@3s=1.000, F1@0.5s=1.000.
    - Die For You (bronze, 8 sections): F1@3s=1.000, F1@0.5s=1.000 (circular ground truth).
- Render improvements:
  - stronger chapter transition visuals (currently: smoothstep crossfade; want: more dramatic)
- Separation stack is **frozen** — Demucs + DrumSep. Vocal model experiments deferred (see `experiments/README.md`).

## How to run locally

### Gerar um vídeo (caminho mais curto)
```
pip install -e '.[stems,viz]'   # Demucs + TorchCodec + matplotlib
# coloque músicas em songs/
make ui                      # picker interativo → stems → video.mp4
make ui UI_LAYOUT=mix        # sem stems, mais rápido
```

### Comandos individuais
- `python3 -m songviz --help`
- `python3 -m songviz analyze path/to/song.flac`
- `python3 -m songviz render path/to/song.flac` (ou `--layout stems4`, `--lyrics`, etc.)
- `python3 -m songviz stems path/to/song.flac` (requer `.[stems]`)
- `python3 -m songviz lyrics path/to/song.flac` (requer `.[lyrics]`; `--backend whisperx` requires `.[lyricsx]`; `--artist`/`--title` overrides ID3 tags)
- `python3 -m songviz lyrics-tap path/to/song.flac` (tap-along session to correct word timing)
- `python3 -m songviz lyrics-template path/to/song.flac` (generate blank corrections.yaml)
- `python3 -m songviz lyrics-correct path/to/song.flac` (apply corrections + print quality stats)
- `python3 -m songviz lyrics-preview path/to/song.flac` (fast lyrics-only video preview)
- `python3 -m songviz sonify path/to/song.flac` (sonify reduced.json → `analysis/reduced.wav` for debug)
- `python3 -m songviz tidy` (limpeza de outputs antigos)
- `pytest -q` (ou `pip install -e '.[test]' && pytest -q`)

## Known issues
- Rendering uses ffmpeg:
  - if `ffmpeg` is on PATH (or available as `$VIRTUAL_ENV/bin/ffmpeg`), SongViz uses it
  - otherwise SongViz tries a user-space ffmpeg binary via `imageio-ffmpeg` (first run may download it)
  - final MP4 writes are atomic: renderer encodes to a temp file in the output directory and only replaces `video.mp4` after ffmpeg exits successfully (prevents partial/corrupt final files on failed/interrupted renders)

## Dev notes
- Keep outputs out of git
- Local dev artifacts are grouped under `.songviz/` (venv, pytest cache, egg-info)
- Prefer small commits after each milestone
