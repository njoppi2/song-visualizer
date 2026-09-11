# Opening visual study — 2026-09-09

Status updated 2026-09-10: `outputs/reviews/visual-passage-01/index.html` is rendered
and technically verified. The user's response clarified product direction rather
than accepting/rejecting this particular animation. Keep it as a visual-vocabulary
study; the next task is a directing-layer prototype in `01_roadmap.md`.

## Product clarification after this study

The user likes the available musical information, but the final output should not
continuously display every instrument or permanently map each one to one effect.
The intended command-driven system should decide what deserves attention from
song structure and context, including changes within sections. It should select,
hide, foreground and reinterpret components, using visual changes to accompany
musical surprises and a coherent story. An LLM may help make those decisions;
it is not a required provider or an instruction to call a model on every frame.

The README now owns this product intent, `02_architecture.md` owns the proposed
director and open decisions, and `01_roadmap.md` owns the revised next experiment.
This study's fixed warm-snare/cyan-kick/gold-hat mapping is not a product constraint.
The earlier question about whether the "motion" felt good was too narrow to test
automatic direction. No further rating of this clip is required to begin that work.

Milestone 2 experiment: one 22-second artistic passage for the opening of the
local Feel Good Inc file. The user explicitly authorized this step and requested
cheaper-model implementation. This is not a production renderer replacement.

## What carries forward

The user confirmed that the hard-to-follow pulse in the previous review was the
cached version, not the regular candidate. The regular pulse and distinct drum
indicators have positive local listening feedback. See `08_rhythm_feedback.md`
and `benchmark/feedback/rhythm-review-01-clarification.md`.

The builder verifies the parent manifest against that feedback, checks timing,
percussion, source audio, and reference media hashes, then consumes those exact
inputs. No beat fitting, separation, or drum detection is rerun. The diagnostic
reference remains available alongside the artistic version with identical audio.
It is a timing reference, not a competing artistic design or a blinded A/B test.

## Visual hypothesis

- Snare/clap: the clearest warm central accent.
- Kick: a separate grounded cyan impulse.
- Hi-hat: small gold peripheral detail, less visually prominent.
- Regular pulse: subtle central motion rather than the strongest flash.

The composition is deliberately calm before percussion enters. That does not
claim the source audio is silent: this study does not yet animate vocals, bass,
or every musical layer. No beat is reinterpreted as a kick or snare, and no detected
hit is snapped to a grid. Avoid global flashes and camera shake; local rhythmic
motion still occurs. There are no timing charts inside the artistic video.

Terra owns the bulk implementation: `experiments/passage_visualizer.py`, its
tests, and `experiments/templates/passage_review.html`. The lead owns the builder,
provenance checks, integration, rendered-output inspection and handoff. Terra
completed the main files and some playback refinements, then hit a usage limit.
The lead finished local contrast/thickness adjustments and playback recovery.
This
records the delegation, not a measured billing comparison or a claim that model
agreement validates artistic quality.

## Build and review

Open `http://127.0.0.1:8767/` while the local server is running, or open the output
HTML directly. To serve it again:

```bash
python3 -m http.server 8767 --bind 127.0.0.1 --directory outputs/reviews/visual-passage-01
```

Generate into a new directory; existing packages are never overwritten:

```bash
.songviz/venv/bin/python experiments/build_visual_passage.py --out outputs/reviews/visual-passage-next
```

The builder consumes `outputs/reviews/rhythm-review-01` by default. The output
contains `visual.mp4`, `reference.mp4`, `original.wav`, a poster and frame samples,
source-code/input snapshots, a manifest, and an HTML feedback page. All generated
audio/video remains in ignored local `outputs/`.

The original review asked whether this visual language strengthens the groove or
distracts from it, and which movement needs to change. Timestamped observations
should target visual behavior, since the accepted underlying event schedules are
unchanged. Do not treat approval of timing as prior approval of the animation.

Milestone 2 remains open under the revised directing-plan acceptance criteria in
the roadmap. Cross-song generalization of the timing candidate remains untested.

## Verification

- 98 focused tests passed across the visualizer, builder, rhythm, percussion,
  review clock, benchmark and evaluation. Frame tests cover deterministic seeking,
  absolute song offsets, independent drum signals, zero-velocity events and no
  pre-onset activation. Input checksum validation rejects changed evidence.
- Both visual and reference videos are 22 seconds, 60 FPS, 1,320 frames. The PCM
  excerpt exactly matches source samples; decoded visual/reference AAC audio is
  identical. Zero-offset AAC/source correlation is 0.999311. Visual frame sampling
  can delay an event by up to 16.67ms; this is not a listening-quality score.
- The lead viewed sample frames around drum events, found the initial strokes
  too faint, and increased local contrast/thickness without changing event timing
  or the quiet base pulse. The final poster was also inspected.
- Browser verification covered seeking, same-position switching while paused and
  playing, timestamp marking, empty-export prevention, manifest-linked export,
  and recovery/retry after a simulated failed fetch. Automated feedback was
  intercepted for testing, not saved as a user judgment.
- Source snapshots and output checksums match the manifest. Selected manifest
  SHA-256: `ea33794b3c0d4e3862826a4f2fe8ad2cb2a522e6d5b6178cc8a52791bae5086a`.

The quiet onset and abstract geometric style are intentional design hypotheses,
not claims of user approval. Future review should test the directed sequence's
musical emphasis and continuity, with individual motion quality as one part of
that judgment rather than the sole objective.
