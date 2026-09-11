# Architecture

This file describes the pipeline shape and key concepts.
For the current checkpoint, start at [CONTINUE.md](../CONTINUE.md).
For the phased plan, see `docs/01_roadmap.md`.

Updated 2026-09-10 after clarification of the final product: automatic visual
direction, not a fixed instrument dashboard. Product intent belongs in the
README, design and open questions here, execution order in the roadmap, and
current checkpoint in `CONTINUE.md` (detailed history in `03_working_state.md`).

## Target architecture — not yet implemented end to end

```text
Audio → musical evidence + whole-song context → saved direction plan → renderer → video
                    analysis                    director             execution
```

One command should orchestrate this workflow and preserve intermediates so a plan
can be inspected or revised without repeating expensive analysis. Exact CLI names
and flags are undecided; the current `songviz render` command is not evidence that
the directing workflow already exists.

### Analysis: what is happening in the music?

Combine measured timing, component activity, repetition, energy and timbral changes
with candidate sections and structural interpretations. Keep observations distinct
from interpretations and attach evidence/provenance and uncertainty. Use whole-song
context even when rendering a short passage; a repeated idea or return cannot be
understood from isolated onsets alone. Do not require perfect note transcription.

"Surprise" is a change relative to an established musical expectation, not simply
a loud onset or a section label. Possible evidence includes a new layer, a missing
expected hit, a break in repetition, a sudden thinning, or a changed texture.
These are hypotheses to test, not implemented reliable surprise detectors.

#### Structure is multiscale, not one partition

The user explicitly clarified that verse/chorus/bridge labels can be useful
identity anchors, but a single discrete section level is not the desired model.
Smaller and larger transitions and degrees of change/novelty can coexist and
overlap. Preserve time-varying evidence at multiple scales and optional event
intervals alongside identity hypotheses; do not reduce all of it to one cut list.

Keep duration/scale, change magnitude, familiarity relative to history, and
importance for visual direction separate. A large return may be familiar;
a small deviation may be unexpected. Do not replace a single section hierarchy
with a single all-purpose "novelty" scalar, or infer perceptual intensity from
uncalibrated acoustic scores. A hierarchy may be one useful view, not a required
partition that every signal or user annotation must fit.

The local-structure experiments retain curves as well as thresholded proposals.
Those peaks are a diagnostic selection, not the whole musical representation.
Reliable overlapping transition extents, graded perceptual salience and their
consumption by the director remain open work. The current comparison does not
claim to implement them.

### Director: what should the viewer notice, and how?

Choose primary focus, secondary context and deliberate omissions over time.
Choose or vary treatments for those layers, composition, palette, visual density,
and transitions. An instrument may be hidden, foregrounded, or shown differently
in different passages. A stable motif can connect repeated musical ideas; changes
should have musical reasons rather than enforcing novelty at every boundary.
Decisions may occur inside a section as well as at section changes.
Continuous changes in emphasis or visual parameters should also be possible
without requiring a discrete section boundary or treating every peak as a cut.

An LLM is a possible planner over structured evidence and relevant diagnostic
images, not the musical ground truth or a per-frame controller. A deterministic
policy can exercise the same plan contract and provide a comparison. The intended
product makes its own direction decisions; hand-authored examples may help define
an experiment but must be labeled and are not proof of automated understanding.

The first experimental version-1 plan contract is now implemented in
`songviz/direction.py`, with a deterministic planner and plan-driven renderer.
See `10_directing_prototype.md` for exact supported fields and limitations. The
broader target contract remains:

- Absolute song-time spans and event references, with the evidence behind a choice.
- Visible/hidden layers, primary focus and optional supporting layers.
- Treatment identifiers and bounded parameters, including composition and palette.
- Motif identifiers, reuse/variation intent, transition timing and behavior.
- Concise rationale, uncertainty, planner/configuration versions and input hashes.

Use a small vocabulary of reusable, configurable visual treatments initially.
This is not a permanent one-treatment-per-instrument map. Validate plan ranges,
available layers, supported treatments and parameters before rendering. Missing
or uncertain evidence should allow restraint or retaining the current focus, not
inventing an instrument or forcing a dramatic event. The exact schema, conflict
rules and fallback policy beyond this bounded experiment are still design work.

### Renderer: execute the direction precisely

Render a validated plan deterministically for fixed inputs/configuration/seed.
Measured event times drive local animation within the chosen treatment; the
planner must not fabricate new hit timing to make its story appear correct.
Keep plan interpretation separate from frame generation. Cache any model-produced
plan so reviewing or rerendering it does not require another model call.

### Implemented structural timing foundation

`compute_story` accepts an explicit beat grid for structural synchronization and
records requested/effective times, hashes and fallback provenance. The isolated
same-code comparison is documented in `12_structure_grid.md`. Default production
callers still track internally; reviewed pulses are not yet a global timing or
downbeat authority.

Fresh structural runs now use one-to-one SSM/energy boundary fusion and
history-masked, unrescaled lag-cosine novelty. `songviz/recurrence.py` provides
role-independent ordered phrase comparisons for the isolated listening review;
it does not yet replace role-derived section letters or drive the director.
See `13_structure_review.md` for contracts, controlled comparisons and limits.

## Open design questions

These are deliberately unresolved. Working approaches below are proposals for
the next experiment, not additional user requirements or implemented behavior.

| Question | Working approach | How to resolve it |
| --- | --- | --- |
| How much should the director decide per section, phrase, or event? | Plan a few focus states using whole-song context and evidence-linked transitions | Review one passage spanning a meaningful change before fixing the plan schema |
| What counts as a meaningful surprise? | Inspect changes relative to repetition, activity and texture; allow unknown | Compare concrete moments with listening feedback; do not equate novelty scores with perceived surprise |
| Which visual treatments and aesthetic controls are needed? | Small reusable vocabulary; selective visibility; consistent motifs | Review directed examples before expanding styles or asking the user to specify every effect |
| Should planning use rules, an LLM, or a hybrid? | Shared plan contract; deterministic comparison; optional model proposal | Compare decision quality, repeatability, latency and cost on the same evidence |
| What are the model/provider, cost ceiling and data-sharing settings? | No provider requirement and no new paid calls/uploads for this documentation change | Resolve configuration and external-data scope before enabling model-backed execution |
| How much user control is exposed? | Inspectable saved plan first; style/seed/focus overrides remain candidates | Learn which corrections recur in short reviews; exact CLI and editing UX remain open |
| How do we evaluate direction separately from timing? | Compare a directed sequence with an always-on fixed mapping using identical musical evidence | Test focus/omission timing automatically; ask whether emphasis, continuity and surprises follow the music |

No additional user choices are needed to document this direction. Do not block a
local plan-contract prototype on choosing a provider or a final art style. Revisit
these questions when an experiment or external execution actually requires them.

## Current implementation boundary

The existing story analysis produces heuristic sections, roles, repetition and
tension signals, including role-derived `visual_behavior` labels. Existing
renderers use reactive signals and section palettes/boundary effects. Those
pieces are useful inputs, but the production pipeline does not yet implement the
full director above. The separate directing prototype now implements saved plans,
activity-based focus selection and deterministic execution for a cached passage;
its artistic quality is awaiting review. The earlier opening study used fixed
component effects and did not validate automatic narrative decisions.

## Key concepts

- **Story**: the structural narrative of a song — section boundaries and roles (intro, build, payoff, valley, contrast, outro), tension arc, repetition patterns, energy dynamics. Not a text narrative; a model of how the song's structure unfolds over time.
- **Separation**: splitting audio into stems (vocals, bass, drums, other) so each musical layer can be analyzed independently. A means to better analysis, not an end in itself.
- **Features**: per-stem signals extracted from audio — vocal pitch track, bass pitch track, drum band energy, chroma. Currently frame-level (one value per ~23ms hop). Used today to drive the renderer.
- **Reduced representation**: discrete musical events derived from features. Drum, bass and vocal paths are implemented; chord/harmony detail remains future work. The goal is a simplified structural skeleton useful for analysis, not a prerequisite for every visual decision. See `docs/06_reduced_representation.md` for historical design.
- **Timbre**: *how* notes sound (tone color, texture, instrument character). Deliberately excluded from the reduced representation. Can be layered back from original audio later as a separate descriptor.

## Existing pipeline

The experimental structural evaluation contract is separate from legacy
`story.sections`: exact annotated intervals and layer-scoped positive identity
groups, optional attributed arrangement/transition interpretations, and independent
acoustic pattern/level and local/historical context evidence. Unknown is a valid
state; role-derived letters are not musical identity, and an interval is not just
a cut. See `16_structural_evaluation.md` for schemas, evidence timing, commands and
limits. Automatic identity/transition inference and consumption by the director
remain future integration work, not implicit behavior of the pipeline below.

1) **ingest**
   - normalize input path
   - hash file contents into a stable `song_id`
   - decode to mono WAV at 22.05 kHz
2) **separate** (optional)
   - stems: vocals/drums/bass/other via Demucs
   - drum sub-components: kick/snare/toms/hh/ride/crash via DrumSep (optional post-pass)
3) **analyze**
   - beats + tempo
   - envelopes (RMS loudness + normalized onset strength)
   - per-stem features: vocal pitch, bass pitch, drum band energy, other chroma
   - structural segmentation (SSM + checkerboard novelty; role-based labels)
   - story signals: tension curve, drop candidates, buildup windows
4) **reduce** (implemented for drums, bass and vocals)
   - convert frame-level features into discrete musical events
   - drum hits and bass/vocal note events; chord labels are not implemented here
   - beat-quantized summaries for section comparison
5) **lyrics** (optional)
   - query LRCLIB for human-verified synced lyrics
   - refine word timestamps via Whisper/stable-ts/whisperx backend
   - write `outputs/<song_id>/lyrics/alignment.json`
6) **render**
   - generate frames at configured fps/resolution
   - section-aware background gradients (crossfade at boundaries)
   - beat flash, tension buildup bar, drop strobe
   - optional lyric word overlay (mix: bottom-centre; stems4: vocals quadrant)
   - mux original audio into MP4 via ffmpeg

## Output structure
```
outputs/<song_id>/
  analysis/
    analysis.json
    story.json
    overview.png          # signal envelopes + section timeline (requires [viz])
    stems_overview.png    # per-stem RMS heatmap (requires [viz] + stems)
    README.md             # auto-generated human-readable summary
  video.mp4
  stems/                  # optional; drums.wav, bass.wav, vocals.wav, other.wav
  lyrics/
    alignment.json        # optional; word-level timestamps
```

`song_id` is the first 16 hex chars of the SHA-256 of the file contents (stable across machines).

## analysis.json schema
Top-level:
- `meta`: song_id, duration_s, sample_rate, created_at
- `beats`: tempo_bpm, beat_times_s
- `envelopes`: hop_s, times_s, loudness (normalized), onset_strength (normalized)
- `story`: sections, tension, events — see story.json

## CLI commands
- `songviz analyze <audio>` — write `analysis/*.json` without rendering
- `songviz render <audio>` — analyze + render + mux into `video.mp4`
- `songviz stems <audio>` — run Demucs and dump WAV stems
- `songviz lyrics <audio>` — run lyrics alignment pipeline
- `songviz ui` — interactive terminal picker
- `songviz tidy` — move legacy output files into hidden subfolders

Full option reference: `docs/04_repo_reference.md`.
Lyrics pipeline contract: `docs/05_lyrics_playbook.md`.
