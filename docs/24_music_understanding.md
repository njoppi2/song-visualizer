# Music understanding experiment design

Design recorded by Team 1 on 2026-09-13. Current assignments and runnable
requests live only in [CONTINUE.md](../CONTINUE.md).

## User direction and question

The user explicitly identified this account as Team 1 and prioritized understanding
the song before improving artistic visualization. Text, simple graphics and
metrics changing with playback are acceptable development outputs. The user
requested a Team 1/Team 2 work split and Terra/Luna assistance. This supersedes
the pending visual-preference dependency; the frozen emphasis comparison remains
available without requiring a response before analysis proceeds.

The question is: can we explain what continues, changes and returns in the song,
with evidence a listener can inspect? Track these dimensions separately:

- Presence and acoustic contribution of each separated stem.
- Local changes and their before/after support at multiple scales.
- Recurring musical material and arrangement differences on returns.
- Unfamiliar material relative to earlier, nonoverlapping audio.
- Vocal behavior, foreground/support roles and transition development, where
  these remain hypotheses until checked against listening evidence.

The first two deliverables are a learned-representation comparison and an
audio-synchronized evidence page. Better artistic rendering is deferred. We can
assess progress without perfect transcription or a complete musical theory model.

## Reusable baseline

Read [16](16_structural_evaluation.md), [20](20_listening_feedback.md) and
[23](23_role_context.md) for existing definitions, evidence and limitations.
The frozen inputs are `structure-evaluation-03`, `role-context-02`,
`listening-examples-01`, the original full-song WAV in `structure-review-03`,
and the two raw feedback exports listed in CONTINUE. Verify the manifests and
consumed files before using them; this planning turn did not reverify their hashes.

Existing code already computes ordered 16/32-beat acoustic recurrence and
separate local/history evidence (`songviz/recurrence.py`), plus continuous
2/4/8-beat stem context (`songviz/role_context.py`). Reuse these as baselines.
The `allin1`, LinkSeg and SongFormer entries in
`experiments/build_structure_tool_diagnostics.py` are marked **not run yet**;
they are not existing learned-model results. The source search found no MuQ,
MERT or Flamingo integration in the active analysis code.

## First Team 1 deliverable: MuQ feasibility and comparison

Start with the original mix and one pinned MuQ checkpoint. The official
[MuQ repository](https://github.com/tencent-ailab/MuQ) supplies frame features,
requires 24 kHz input, recommends fp32, and warns that its released checkpoint
may perform differently from paper results. Its representations are candidates
for useful similarity measurements; neither pretrained status nor agreement with
MERT would establish musical identity or importance.

1. Inspect the available runtime and resolve checkpoint revision, dependency
   versions, size and memory needs. Use an isolated environment. Record a short
   real inference with elapsed time, memory, feature shape and finite-value checks
   before scheduling full-song extraction. If GPU access fails, measure a bounded
   CPU run and report feasibility; do not repair system drivers or provision
   paid remote compute as a routine experiment step.
2. Record the extraction configuration before evaluating listening labels:
   chunk length/overlap, padding, resampling, frame-to-source timing, chosen layer
   and pooling. Preserve the entire encoder input context for each feature:
   a frame timestamp alone does not describe the context of an attention encoder.
   Check chunk seams using shifted chunk placement. State offline availability.
3. Extract a full-song feature sequence if feasible. Compare adjacent windows
   at 2/4/8 beats per side and ordered recurrence at 16/32 beats, matching existing
   baseline support where possible. Preserve temporal order for phrase comparison;
   a single pooled vector cannot establish matching musical sequences. Historical
   reference windows must end before the target starts; account separately for
   encoder context that reaches outside those windows. Missing history is unknown.
4. Produce a saved feature artifact, source-time curves, comparison table and
   readable report in a fresh package. Record full cosine-distance definitions
   and normalization, with no per-song rescaling presented as calibrated confidence.
   If mix/stem or representation support differs from the baseline, report the
   confound explicitly instead of attributing all differences to the model.

Evaluation uses the four existing excerpts (16–26, 57–70, 74–85, 119–132s), all
eligible existing positive motif-return examples, and their support limitations.
Show whether MuQ adds explanatory evidence or merely responds to acoustic change
already captured by the baseline. In particular, retain the 21s `none` case and
the 124s continued-voice case even if they disfavor the model. Inspect rank/order
and full curves; do not tune thresholds to these four cases. Different motif
names and unlabeled times are not negative labels. Findings apply to development
music; broader accuracy needs new held-out listening material after this review.

Acceptance requires real inference and inspected numeric outputs, honest failures,
verified source/model/config provenance, timing and chunk-seam checks, saved-output
replay, and comparison against the baseline. Synthetic tests alone cannot certify
that a model ran. If infeasible, a measured feasibility report is the handoff;
feature extraction and model benefit remain unfinished.

## First Team 2 deliverable: synchronized evidence page

Build a full-song diagnostic page from the existing frozen baseline packages.
It must work independently of Team 1's model run. Provide native audio controls,
a shared playback cursor, click-to-seek plots, excerpt shortcuts, and a small
text panel explaining the selected window. Start with stem RMS/activity/share
and selectable local contrasts, then provide existing recurrence/history evidence
with both compared passages available for listening. Plot labels must explain
units, scale, support and missing data. Keep the initial view readable; expose
the detailed metrics through selection rather than displaying every curve at once.

Use deterministic factual text tied to the selected measurements, e.g. voice
activity persists while RMS and share fall. Show user notes with their source,
and label analyst interpretations separately. Do not generate a musical-role
label from RMS, or turn a threshold crossing into an asserted musical event.
At source times unsupported by a measure, show unavailable rather than silently
holding the last value. Display window support when snapping to an anchor.

This first page needs no new annotations, model downloads, LLM, source separation,
director changes or decorative visual treatments. All existing packages stay
frozen. Team 2 owns a small adapter to the existing package schemas. Team 1's
future model package will have its own versioned schema; adding it to the page
is a subsequent integration task after both outputs are inspected. Neither team
depends on an invented shared schema in this first pass.

Acceptance: verify the consumed input fingerprints, compare displayed numbers
and text with their actual source records, and inspect playback/seek behavior,
paired-passage listening, unsupported edges, desktop/mobile readability and
audio failure/retry. Build into a new directory with source/output manifests.
Use focused tests for mapping/timing and actual browser checks for interaction.

## Subsequent model questions

Music Flamingo is the planned semantic probe once local/remote execution
feasibility is established. The official
[checkpoint card](https://huggingface.co/nvidia/music-flamingo-2601-hf) is the
starting reference. Ask neutral questions about what is audible and what changes,
without providing the song title, our annotations, or the expected laughter/role
answer. Save the initial answer before more focused prompts. Test original-mix
excerpts first; isolated stems and different context lengths are sensitivity
checks with different input distributions. Save exact prompts, decoding settings,
raw replies, source intervals and uncertainty. Repeated deterministic answers
are reproducibility evidence, not independent confirmation.

Evaluate individual claims against audio and existing notes, including invented
events and incorrect timing. A fluent caption or uncalibrated self-reported
confidence cannot establish improvement. Audio Flamingo 3 can later test broader
sound/vocal-event questions. MERT can later test representation sensitivity;
MuQ-MuLan can test controlled text/audio associations. Their use is conditional
on a concrete unanswered question, not a requirement to install every model.

## Planning evidence and delegation

This turn inspected the entry documents, current sources and baseline experiment
definitions, with a Terra worker auditing reuse and task independence read-only.
No model inference, downloads, new audio package or application tests were run.
The local `nvidia-smi` query failed with **Driver/library version mismatch**
(NVML 580.178); this does not prove PyTorch CUDA inference will fail, which still
needs checking. The filesystem reported 43 GB available. No driver changes were
made. The plan therefore includes a measured feasibility step.

For implementation, the lead fixes the experiment and reviews outputs. Use one
Terra worker per bounded implementation initially; add a worker only for useful
independent work in disjoint files. Luna is suitable for checked inventory or
documentation tasks. The separate Team 2 account receives its request through
CONTINUE and the human, not through an assumed live agent channel.

## Fixed first-run protocol — 2026-09-13

Recorded before model outputs are evaluated. Use MuQ's final hidden layer,
fp32 evaluation mode and the original mono mix resampled to 24 kHz. Begin with
a two-second CPU feasibility probe, since the existing PyTorch CUDA query
failed with error 804. If practical, extract 10-second input chunks at five-second
hops; assign frames to the closest chunk center (earlier center wins ties), with
edge ownership covering the available source. A second placement shifted by
2.5 seconds measures context/chunk sensitivity across the song. Record exact
chunk intervals, real/padded input extent and model-derived feature centers.
Checkpoint inspection may resolve clock details, but musical labels do not
select the layer, chunk length, placement or comparison settings.

Pool frame centers within each existing beat interval by arithmetic mean;
retain frame counts and the union of their full encoder input support. Local
2/4/8-beat comparisons use cosine distance between means of the left and right
beat vectors. Ordered 16/32-beat recurrence uses concatenated L2-normalized beat
vectors, four-beat stride, and cosine similarity; preserve the sequence rather
than averaging a whole phrase. Cosine similarity lies in [-1, 1] and distance
in [0, 2]; these are uncalibrated geometric measurements. Zero vectors or missing
beats are unknown. No per-song rescaling or event threshold is introduced.

Retain both nominal nonoverlapping historical comparisons and the subset whose
full encoder contexts are also temporally separate. Label the first as potentially
sharing audio context. The encoder can see future audio within its chunk, so
even the strict subset is offline evidence, not an instantaneous novelty signal.
The existing baseline also has different preprocessing/support and stem-based
inputs; matching beat windows cannot isolate a pure model effect.

Use all existing fixed neighboring anchors for the four listening cases and all
eligible positive separated-return pairs. Compare full base/shifted curves and
their paired changes, including per-beat embedding differences. Do not select
the best placement per case. A high response in the `none` example remains a
failure of salience inference, even if it is stable under the shift.

Before implementation, lead verified 34 fingerprint records in
`structure-evaluation-03`, 78 in `role-context-02`, and 15 in
`listening-examples-01`. Baseline feature cache contains 499 beat endpoints and
four 84-by-498 log-CQT arrays plus four 498-value RMS arrays. Extraction itself
will not receive feedback labels. Model execution/results are recorded below
once observed; this protocol is not evidence that the model has run.

## Completed Team 1 experiment — 2026-09-14

**Decision:** local CPU inference and a full-song learned-representation comparison
are feasible and verified. Retain MuQ as development evidence, not a promoted
identity, salience, vocal-function or director policy. It responds strongly to
the known drum entry, which the baseline already explains, but also responds in
the user's `none` example. No advantage in semantic understanding is established.

The final readable [report](../outputs/reviews/music-representation-02/report.md)
and [full-song curves](../outputs/reviews/music-representation-02/local-curves.svg)
are in `outputs/reviews/music-representation-02/`. The report includes all fixed
central cases, neighboring-anchor ranges, positive-return summaries and placement
sensitivity. All eight numeric/review JSON files are unchanged from version 01;
version 02 only clarifies the plot's time ticks and anchor-line explanation.
The graph is a static diagnostic, not the synchronized Team 2 interface.

### Runtime and pinned provenance

- Checkpoint: `OpenMuQ/MuQ-large-msd-iter`, revision
  `0562a57814f6f8bbd9fdea0a25921a2fce1a841a`.
- Weights: 1,333,825,096 bytes; SHA-256
  `273febab2be02872c37d2c37e48a9d6c52c1c9392f3eeeabd498efa281ccb7a6`.
  Config SHA-256:
  `237335ee27d8fb951ce778701a12a79e06c51ae636dd786f97e45f51ce532543`.
- Local model: `.songviz/representation-models/muq-large-msd-iter-0562a57814f6/`;
  isolated interpreter: `.songviz/representation-venv/bin/python`. The environment
  inherits existing site packages read-only; separate dependencies were installed
  there without modifying the existing analysis environment.
- Actual architecture: 333,401,472 parameters, encoder depth 12, dimension 1024.
  The outer depth overrides the nested config's depth 24. Final hidden state,
  fp32 evaluation, four CPU threads. Recorded versions include MuQ 0.1.0,
  torch/torchaudio 2.10.0, transformers 4.57.6 and NumPy 2.2.4; full versions
  and MuQ library-source snapshots are in each runtime manifest.
- Original FLAC SHA-256:
  `657af9333edafbb6cf8cc573446650773d1169b0117c784106d1895cf802cb44`;
  9,753,744 stereo frames at 44,100 Hz, duration 221.17333333333335s.
  Each chunk is cut from source PCM before stereo float32 averaging and
  torchaudio resampling to 24 kHz. Full supplied chunk support is retained.
- GPU access failed with NVML driver/library mismatch and PyTorch CUDA error
  804. No drivers, paid compute or external audio service were used.

Preliminary feasibility records in `music-representation-feasibility-01/` are
`muq-2s-probe.json` (frozen original WAV, 16–18s, 50x1024 features, 0.5016s
model-only inference) and `muq-10s-probe-flac.json` (original FLAC, 250x1024,
1.2637s). Peak process RSS was about 3.36 GiB. These preliminary probes do not
have the final runtime's complete source-snapshot coverage; the full extractions
below are the reproducible evidence.

### Frame clock, coverage and failures resolved

The inspected model uses a 240-sample mel hop and two stride-2 convolutions,
giving a nominal 25 Hz clock. Preserve each chunk's frame index rather than
rounding timestamps onto a global grid. Centered mel drops its last frame and
padded convolutions determine the final count; geometry is checked against the
actual extraction. There is no additional authored source padding.

Final ownership is the closest **eligible containing** chunk center, earlier
chunk on exact ties. The shifted pass includes an initial boundary chunk at 0s,
then starts at 2.5, 7.5, … seconds. Both passes have 45 chunks, 5,530 finite
1024-dimensional retained frames, no duplicate/out-of-order timestamps, and
maximum inter-frame gap 0.04s. Coverage begins at 0s and ends within one nominal
frame period of the source end. Every frame retains its full encoder support.
The 157 common retained features from the identical initial chunk are byte-identical.

Two implementation defects were found before accepting the full results:
global rounding at the shifted half-frame phase discarded frames, and assigning
tail ownership only by center midpoints created a 0.68s coverage gap. Per-chunk
indices and eligible-center selection fix these, with dedicated regression tests.
`music-representation-runtime-01/` is preserved as a rejected pre-hardening
development artifact and must not be reused. The first comparison build also
exposed a positional/keyword mismatch for `shifted_result`; it was fixed before
the package was emitted. Team 2 subsequently saw three synthetic-fixture failures
against the hardened clock contract; those fixtures now match the contract and
the full current suite passes. None of these failures was hidden by overwriting
a frozen package.

Base and shifted full-song runs used 51.0912s and 54.2661s of summed model
inference respectively, excluding loading/preprocessing/serialization. Peak RSS
was approximately 3.35 GiB. The shift changes both context and frame-grid phase
(20 ms); it is placement sensitivity, not an isolated seam-effect experiment.

### Findings and limits

All 498 pooled beats have support. Each pass retains 495/491/483 local contrasts
at 2/4/8 beats per side (1,469 total), and 6,903/5,995 ordered recurrence pairs
over 121/117 spans at 16/32 beats (12,898 total). Nominal historical support is
available for 117/109 spans; strict encoder-independent history for 114/104 in
base and 112/105 in shifted. Strict history requires the complete prior encoder
context to end before the target context starts. Both remain offline evidence.

All 60 fixed listening anchor/scale joins and 616 eligible explicit positive
return pairs are retained. No unlabeled time or differently named motif is
treated as a negative. At the fixed central anchors:

| Existing case | Observed MuQ evidence | Interpretation limit |
| --- | --- | --- |
| Drum entry, 79.1034s | Base distances 0.7492/0.7913/0.6989; descriptive within-scale midranks about 96/97/94%. | Strong response agrees with the already-visible drum RMS/share/activity rise; not unique added understanding. |
| Within-passage `none`, 20.6300s | Distances 0.5071/0.5432/0.2870; substantial short-scale responses persist after shifting. | Exceeds the verse-ending response at 2/4 beats in both passes; magnitude is not perceived importance. |
| Verse ending, 124.5827s | Distances 0.4310/0.5164/0.3541. Baseline vocal activity stays 1 while RMS and share fall at each scale. | Embeddings do not identify laughter, vocal function, or foreground-to-support role. |
| Breakdown/transition, 63.0774s | Distances 0.3733/0.3767/0.5687; about 10/10/84% within-scale midranks. | Broader context responds more strongly, without establishing physical transition endpoints. |

These midranks use the whole curve for description only, not calibration or
threshold selection. All five neighboring anchors remain in the report and JSON.
Positive-return median MuQ cosine is about 0.34–0.39 across the four motif/scale
groups (chorus and verse 2). Baseline pattern medians are about 0.59–0.67 and
arrangement medians 0.67–0.87. Higher values are not evidence of higher accuracy:
the metrics and full-mix versus separated-stem inputs differ. The current return
join uses layer/motif text; an independent check confirmed matching identity IDs
for every actual pair. A future schema allowing identical motif text with distinct
identity IDs needs an explicit ID-aware join before reuse.

Placement changes are material: median absolute local-distance differences are
0.0504/0.0561/0.0441 at 2/4/8 beats, maxima 0.2381/0.2391/0.1747, and full-curve
Pearson correlations 0.8114/0.7991/0.9022. Median pooled-beat base/shift cosine
distance is 0.0988. Both passes are retained; no best placement or scale was
selected. One familiar development song and positive-only recurrence annotations
cannot establish general quality or a semantic-model advantage.

### Verification, artifacts and reproduction

Terra workers implemented bounded runtime, numerical and package/test tasks;
the lead inspected actual code, hardened frame/provenance checks and reviewed
the real outputs. An independent Terra result reviewer checked all 60 joins,
616 positive-return rows, pair universes and strict-history conditions without
a blocking finding. At final handoff:

- Lead focused tests: **33 passed, 1 warning** (0.24s).
- Fresh Terra shared-worktree full suite: **731 passed, 123 warnings** (20.18s).
  The previously reported three Team 1 fixture failures are resolved. This full
  suite does not constitute independent browser acceptance of Team 2's page.
- Lead reverified **161 dependencies, 140 snapshots and 10 outputs per final
  package**, including source/model/config bindings. Builders reconstruct exact
  expected frame clocks and supports and recheck dependencies before manifesting.
- Final/replay **all 10 outputs match byte-for-byte**, including curves and
  report. This replays analysis from saved real features, not a second model
  extraction. Eight JSON outputs also match comparison version 01 exactly.
- Lead rendered and inspected the final standalone SVG with Playwright;
  capture `/tmp/songviz-muq-review.EZPKT1/local-curves-final.png`. Final time
  labels no longer overlap. `git diff --check` passed.

Manifest SHA-256 values:

| Package under `outputs/reviews/` | SHA-256 of `manifest.json` |
| --- | --- |
| `music-representation-runtime-02` | `1994c45a3d61a31913287587504b5ffedc81755cecfd808fea62e8e7579c6c2a` |
| `music-representation-runtime-shifted-02` | `bdd9fa71304dbbd3b7a9bbea7cb9649c248540c1bac4052ed203e44cb09afd94` |
| `music-representation-02` | `ed22cac082710fc13b0943bcca11bda8c1f552b42832bb654926cecd24bc7dd6` |
| `music-representation-replay-02` | `e6028dd9cfb7aa9351f4094d7488c53bc6818c215c3ca21c7aecef738af41180` |

`review.json` retains pooling, both full local/recurrence results, joins and raw
notes. Separate local/recurrence/evaluation/shift-difference JSONs support focused
inspection. `inputs/` snapshots include extracted frames and analysis/library
code; audio and model weights remain fingerprinted external dependencies, not
duplicated gigabyte payloads. Existing output directories are immutable.

Commands actually run for final validation/replay:

```bash
.songviz/venv/bin/python -m pytest -q tests/test_music_representation.py \
  tests/test_music_representation_probe.py tests/test_music_representation_builder.py \
  --disable-warnings
.songviz/venv/bin/python -m pytest -q --disable-warnings
OPENBLAS_NUM_THREADS=1 .songviz/venv/bin/python experiments/build_music_representation_review.py \
  --runtime outputs/reviews/music-representation-runtime-02 \
  --shifted-runtime outputs/reviews/music-representation-runtime-shifted-02 \
  --out outputs/reviews/music-representation-replay-02
git diff --check
```

For another analysis replay, replace `--out` with a verified-unused directory;
do not rerun the command into the frozen replay above. No new inference is
needed. If an explicitly requested new extraction is later required, the
isolated interpreter runs `experiments/probe_music_representation.py` with
`--audio`, `--model-dir`, `--full-output-dir`, `--offset-s 0` or `2.5`, and
`--threads 4`; use fresh directories and verify the pins first.

The bounded MuQ deliverable is complete. The subsequent integration review is
recorded below. Adding MuQ to the evidence page remains a separate reviewed
schema/display request.

## Team 1 integration review — 2026-09-14

**Decisions:** return `evidence-timeline-05` for interaction/support repairs;
accept doc 26's current no-run conclusion with the hardware qualification below;
return doc 27 for scoring/control revisions before preregistration. Neither
semantic execution nor model/page integration is ready. The runnable counterpart
repair requests and current ownership are in `CONTINUE.md`.

Reviewed inputs (SHA-256):

| Input | Digest |
| --- | --- |
| `evidence-timeline-05/manifest.json` | `53fa424c5ffe0ce7f3fcbd9009a1ed72e78170c72f46ee846240905d4b2db6c6` |
| `docs/26_semantic_runtime_feasibility.md` | `91cde75fa4f7a1ebe89d0c4e09187a369ebf63335b8efe6a781eadd302e112d5` |
| `docs/27_semantic_experiment_design.md` | `88939f3c473c35ad359afc2c2114a5686cc2551cf63c17bb496d7d5a8592c363` |

### Evidence page: verified data, returned interface

A Terra worker verified all **103 sources + 8 snapshots + 2 outputs** and their
byte counts; current builder/template/checker match their snapshots. The lead
inspected the verifier and generated results. Adapted local context exactly
preserves 499 anchors and 495/491/483 supported records; recurrence selection,
24 saved listening pairs and 16 unavailable probes match the frozen inputs.
Exact HTML derivation, feedback/excerpt parity and original-audio binding were
also checked. Focused builder tests passed **3 tests, 1 warning**. The existing
browser checker passed using the repository's ranged review server, including
its error/retry and unsupported-local-edge checks. Its paired-listening check
only seeks to interval starts; it does not prove playback advances or stops.
An additional Terra browser audit checked **96 combinations** at 124s (three
scales, four stems, eight metrics): selected-source numbers matched factual text
and table values. All were finite at that cursor; a separate unavailable-edge
probe verified null/unsupported text. The lead inspected its script and result.

The lead additionally clicked both real paired-listening buttons and observed
audio advancement; a short `playRange(57,58)` stopped paused at exactly 58s.
Desktop and mobile screenshots were inspected, and 320/375/768px had no page
overflow. Desktop explanations and numeric tables are legible. Mobile requires
substantial scrolling and the SVG's scaled text is small; layout success is not
evidence that the user now understands the graph.

Lead browser findings in the frozen template:

| ID | Reproduction / evidence | Required repair |
| --- | --- | --- |
| P1 | `chart()` draws time with `x=40+830*t/duration` in a 900-unit SVG, but clicks use the entire SVG width. Clicking the plotted 60s coordinate seeks to **65.317269s**. | Invert the actual plot transform, including margins and clamping; check known ticks and endpoints at desktop/mobile widths. |
| P2 | After seeking to 40s and playing, audio reaches **40.717079s** while `#cursor` stays **40**. `timeupdate` redraws the graph but never updates the slider. | Keep the slider and native audio synchronized during playback and native seeking. |
| P3 | `playRange(16,26)` followed by `seek(100)` lands at **26s, paused**, on the next update. `seek()` retains the previous `stopAt`. | A new ordinary seek must cancel or explicitly replace the old bounded audition; preserve normal range stopping. Cover plot, slider and native seeks. |
| P4 | At song end (**221.173s**), local support is unavailable but history still shows **0.28247 / 117 prior passages**, selected from the last record at 220.305853s. `renderPairs()` always selects the nearest record and displays neither that record's span nor its availability time. The independent pair selection can refer to another passage. | State the exact history source window and its relationship to the cursor/selected pair; define support selection and show unavailable outside it. A nearest saved record must be explicitly identified, never look like evidence at the unsupported cursor. |
| P5 | The guide says the graph "started at the first supported anchor" but interpolates current playback time; after the tick click it names **1:05.317** as that starting anchor. | Store the true initial anchor or describe current time accurately. |

These are returned findings, not permission to overwrite version 05. Team 2
should repair its source/checker and produce a fresh candidate, retaining all
numeric evidence and frozen packages. P1–P4 block interface acceptance; P5 is a
small factual-text repair in the same pass.

Diagnostic evidence is in
`outputs/reviews/music-representation-integration-review-01/`: `verify_page.py`,
`audit.json`, `verify_display.cjs`, `display.json`, `browser_review.cjs`,
`browser/observations.json`, and captured desktop/mobile PNGs. These scripts
capture review observations; the known bad
values in the JSON are defects, not expected behavior for a repaired page.

Reproduction (run from repository root; choose new diagnostic output paths):

```bash
.songviz/venv/bin/python -m songviz.review_server
.songviz/venv/bin/python -m pytest -q tests/test_evidence_timeline_builder.py --disable-warnings
SONGVIZ_PLAYWRIGHT_MODULE=/home/njoppi2/.npm/_npx/04489aa25f0b609a/node_modules/playwright \
  node experiments/check_evidence_timeline.cjs http://127.0.0.1:8770/evidence-timeline-05/
.songviz/venv/bin/python outputs/reviews/music-representation-integration-review-01/verify_page.py \
  --out /tmp/songviz-page-audit-NEW.json
SONGVIZ_PLAYWRIGHT_MODULE=/home/njoppi2/.npm/_npx/04489aa25f0b609a/node_modules/playwright \
  node outputs/reviews/music-representation-integration-review-01/verify_display.cjs \
  http://127.0.0.1:8770/evidence-timeline-05/ --out /tmp/songviz-display-NEW.json
SONGVIZ_PLAYWRIGHT_MODULE=/home/njoppi2/.npm/_npx/04489aa25f0b609a/node_modules/playwright \
  node outputs/reviews/music-representation-integration-review-01/browser_review.cjs \
  /tmp/songviz-page-browser-NEW http://127.0.0.1:8770/evidence-timeline-05/
```

The first exploratory run used plain `http.server` on 8771, which cannot serve
the required byte ranges and caused seek/playback failure. That run is discarded;
the findings above were reproduced using `songviz.review_server`. No full-suite
rerun was warranted for this review-only work; the earlier 731-pass result is
historical, not a new validation claim.

### Runtime audit: accept current stop, qualify proposed recovery

The lead rechecked the [Hub model identity](https://huggingface.co/api/models/nvidia/music-flamingo-2601-hf):
revision `6b5be086d52f65a1e204cb0faf70bf54e2741ecd`, ungated, last modified
2026-04-09. The [publisher card](https://huggingface.co/nvidia/music-flamingo-2601-hf)
and [upstream runtime documentation](https://huggingface.co/docs/transformers/main/en/model_doc/musicflamingo)
support the model/API and audio-context constraints in doc 26. Its
16,534,531,504-byte weight size equals 15.39898 GiB; a two-copy envelope is
30.79796 GiB. These are storage/loading arithmetic, not measured peak inference.

A Luna worker reproduced unavailable CUDA and incompatible installed packages.
Kernel NVRM is 580.173.02 while the NVML user library reports 580.178; both
CUDA-bearing project environments expose zero devices and error 804. The lead
read `/proc/driver/nvidia/gpus/0000:01:00.0/information` directly: **NVIDIA GeForce
RTX 4050 Laptop GPU**. [NVIDIA's specification table](https://www.nvidia.com/en-au/geforce/laptops/compare/)
lists a standard **6 GB** memory configuration for that GPU. That is published
capacity, not a successful measurement of this machine's free VRAM. It is already
below the raw BF16 weights, so repairing the driver does not make the proposed
all-on-GPU, unquantized, no-offload smoke test viable.

Lead `free -b` snapshot: total RAM 33,343,787,008 bytes (~31.05 GiB), available
11,787,624,448 bytes (~10.98 GiB); swap 2,147,479,552 bytes with only 61,440 free.
Available memory varies with concurrent processes. It limits the current run;
it does not prove permanent CPU infeasibility with another loading scheme. Disk
remained about 40 GiB free. No weights were downloaded or inference run.

Team 2 should amend doc 26 to distinguish driver restoration from a feasible
compute route and report **no demonstrated route under current constraints**.
A different memory/quantization/offload/hardware plan needs its own bounded
feasibility evidence. No repair, installation, remote resource, or paid compute
is assigned by this review. A systems smoke test must also be reconciled with
the frozen scientific protocol: doc 26's illustrative 20s verse excerpt and
prompt are not doc 27's scored 18-clip run.

### Semantic design: useful structure, revisions required

The lead reviewed doc 27 against the exact raw notes in
`benchmark/feedback/listening-examples-01.json`, doc 20, the completed MuQ report
and doc 24's constraints. Retain the claim-level audit, neutral first prompt,
the `none` control, original-mix core, familiarity warning, and held-out-only
generalization boundary. The following issues prevent preregistration:

1. **S1 — decision precedence is contradictory (§5).** Verse + another AVU
   obtained only in Stage B satisfies SUPPORT and UNRESOLVED. Zero AVUs caused
   by missing timestamps or scorer disagreement satisfies REFUTE and UNRESOLVED.
   Require one ordered, exhaustive decision procedure: establish scoreability
   and control validity before interpreting absent benefit. Include worked
   outcomes for Stage-B-only success, all abstentions, missing timestamps,
   scorer disagreement, and complete valid failure.
2. **S2 — smaller-run and padding rules disagree (§3.2, §4.3).** Dropping
   post-padding makes the required "both padded clips" criterion impossible;
   dropping drum ablations leaves T2 requirements undefined. Fix the minimum
   run before output inspection, or declare omitted controls unresolved.
   Real pre/post padding adds new music: score claims about the shared core
   interval and match the same event/transition across contexts. Changes in
   caption scope or interval centres must not automatically count as invented
   audio. Define event matching, timing tolerance and missing-time handling.
3. **S3 — manipulation validity is assumed (§4.2 E2, §4.3.4).** Removing a
   separated vocal stem does not establish that laughter disappeared from all
   retained stems. A model's "faint" wording cannot validate bleed, and a
   confident claim is not necessarily false. Thresholded frozen activity also
   cannot prove auditory absence. Require independent evidence that the intended
   ablation worked, or mark the control unverified and withhold its AVU. Keep
   proxy conflict separate from a demonstrated contradiction. Existing notes
   do not provide counterfactual listening labels.
4. **S4 — the always-continuation safeguard leaks (§4.3.5).** T2 is continuity,
   yet it is listed as evidence of "some change" that permits T3 credit. A
   model saying "same part/no change" everywhere can satisfy that clause.
   Require an explicit supported entry/drop/change proposition for the positive
   check; demonstrate the constant-continuation counterexample earns no credit.
5. **S5 — targets and execution details remain underspecified.** T5 broadens the
   note's laughter into any "non-melodic vocal sound"; define partial matches so
   generic vocalization cannot earn the same credit as the specific behavior.
   T6 must distinguish loss of perceived importance in the note from an analyst's
   inference of a supporting role. Freeze seed/order, token budget, exact
   format-retry prompt/eligibility, stage aggregation and segmentation/mapping
   procedure before outputs are viewed. A "preregistered" heading alone is not
   a completed preregistration.

No fresh annotations or model outputs were needed to find these defects. Team 3
owns the design revision; it can make an untestable control explicitly unresolved
using current evidence. Agreement between delegated scorers remains scoring
consistency, never new auditory ground truth.

### Reconciled next action

Return page/runtime repairs to Team 2 and scoring/control revisions to Team 3,
independently. Team 1 reviews their next durable handoffs and only then defines
a single execution request if a valid design and a feasible compute route both
exist. Current evidence supports neither a semantic-model result nor a model
download. The checkpoint/protocol were refreshed to distinguish completed work
awaiting integration from accepted work and to require explicit completion
status in each team's owned deliverable.

## Team 1 revision review — 2026-09-14

This review follows the P1–P5/S1–S5 handoffs above; earlier findings and results
remain historical. Reviewed document hashes:

| Input | SHA-256 |
| --- | --- |
| doc 25 repair handoff | `8ec3156063ec808d75fd5006d5775303c5edb1d477f1bcd9bab5225f1e6442ea` |
| doc 26 corrected runtime audit | `80b56a23391519f53cd442680d43ced3ae6d91ad35113bda434a4653e88adc9f` |
| doc 27 revision 2 | `f13be1df36a3dbc74ce870528eab205f0e93a6c346e71a678813324ce1236119` |

### Page and runtime decisions

**Accept `evidence-timeline-10` as the frozen baseline evidence interface.**
Manifest: `a2b2b55fbb00b0bdd8859c4bd88be25336fa433590ef72010bed611a04c91fa1`.
Its adapted timeline is byte-identical to version 05:
`aad36cd2828c95aba7ce7675aeb878422f8f7704dcd94ca94d068a46994848fe`.
No measurements or source audio changed. This accepts the scoped diagnostic
interface, not musical interpretations or proof of user comprehension.

The lead inspected the actual frozen-05-to-10 template changes, the expanded
checker, delegated verification scripts/results, and desktop/mobile captures.
A Terra worker verified all **113** bound source/snapshot/output records, exact
current source/snapshot equality, and frozen-source adapter parity. Focused
builder tests passed **3 tests, 1 warning**. The lead independently verified
exact HTML derivation, raw feedback, excerpt and original-WAV hash binding.

Lead browser observations against version 10:

- Plotted 60s click → **60.069248s**, resolving the former 65.317269s result.
  The small residual comes from browser click coordinates; the handler now
  inverts the actual SVG transform and plot margins.
- Audio **40.702961s**, slider **40.479s** during playback: within one coarse
  native `timeupdate` interval, resolving the permanently stale slider.
- Both real paired buttons advanced playback. A bounded 57–58s audition paused
  exactly at **58s**. `playRange(16,26)` followed by `seek(100)` continued to
  **100.460105s, unpaused**, instead of jumping back to 26s.
- The worker's additional end-to-end paired-button check stopped paused at the
  saved endpoints **14.132971s** and **27.993333s**. The expanded browser checker
  passed plot/slider/native seek replacement, history support, retry and layouts.
- Song end explicitly shows history unavailable, its last saved window end,
  and no carried-forward value. Supported history discloses the selected source
  span, availability time, and relationship to the independent listening pair.
- The opening-anchor guide stays fixed during seeking. Desktop/mobile captures
  were inspected; 320/375/768px had no page overflow. Mobile still involves a
  long page and small plot text.

Two nonblocking maintenance caveats remain. The guide stores the opening
2-beat anchor (6.336518s) but calls it the data start even after selecting 4/8
beats, whose first anchors are 7.202790/8.935336s; wording should identify the
opening view or follow the selected scale in a future maintenance pass. At a
bounded endpoint, the correct pause/time is followed by a `seeking` event that
replaces "Trecho concluído" with "Cursor movido". Neither changes the underlying
measurements, disclosed local support or verified stopping behavior. Acceptance
is with these recorded caveats; no extra rebuild is assigned for them now.

Diagnostic package: `outputs/reviews/music-representation-integration-review-02/`.
`worker/verify_candidate.py` and `worker/audit-10.json` record provenance/parity;
`worker/report.md` and `worker/transport-buttons-10.json` record full paired
playback and the nonblocking caveats;
`lead-browser/observations.json` and PNGs record the lead browser checks using
the previous review's `browser_review.cjs` against the new URL. Scripts require
fresh output paths. No extraction or full-suite rerun was necessary.

**Accept doc 26's corrected feasibility conclusion as a completed audit.** It
now distinguishes the observed driver mismatch, the GPU's published capacity,
the weight-size arithmetic, and unknown peak runtime needs. It explicitly states
that driver restoration alone is insufficient and that no viable execution
route is demonstrated. Its conditional smoke-test illustration is not an
execution request or a preregistered semantic run. Historical RAM/swap readings
are snapshots from the audit, not fresh measurements in this revision review.

### Semantic design decision and remaining findings

**Return doc 27 revision 2 for three bounded scoring corrections (R1–R3).**
Its ordered gates remove the original overlapping outcomes; the fixed eight
required clips resolve the reduced-run contradiction. The positive control now
requires entry/drop evidence rather than continuation. Ablation validity is
explicitly unverified without a separate pre-output listening check. These
changes address substantial parts of S1–S4.

Team 1 accepts the narrower development question: test the specific vocal-manner
transition, with the other cases serving as controls/descriptive evidence. The
removed second-AVU requirement is acceptable for deciding whether to design a
future held-out study. It does not establish general semantic understanding.
The manipulation check is a declared unmet condition, not permission to collect
new labels or an assertion that the current stems remove all audible laughter.

The lead and a bounded Terra reviewer traced new counterexamples against the
actual rules, without treating their agreement as auditory truth:

| ID | Counterexample and applicable rules | Required correction |
| --- | --- | --- |
| R1 — prompted completion earns unprompted credit | Core A: "At 6s a man laughs." Core B: "At 0s the man raps before laughing at 6s." A alone is T5-partial. Sections 4.2–4.3 combine stages, add the missing lead/ordering predicates from B, and mark the match unprompted because the laughter claim was in A. It can pass G6 and ultimately SUPPORT under otherwise valid controls. | Keep a separate Stage-A-only full-target match. Every predicate needed for the credited transition must be present in A; B must not upgrade partial A into unprompted-full. A may contain separate atomic claims connected by valid ordering. |
| R2 — assertion polarity/modality and ablation exception | G10b says "no laughter-term claim" but gives no positive/negative assertion rule. "No laughter is audible" can be confused with an audible-laughter assertion. "Unsure whether the man laughs" is also not explicitly excluded from targets. Conversely, positive "a laughter sample is audible" is exempted by G10b even though the manipulation check asks whether any laughter is audible. | Record presence/absence/uncertainty explicitly. Only affirmative assertions earn target credit; explicit absence must not become an ablation conflict. Define uncertainty's gate outcome. Align G10b with the manipulation check: an affirmative audible-laughter claim cannot escape solely by being called a sample/effect. |
| R3 — invalid timestamps can satisfy anchoring | Section 4.3 parses times and section 4.4 tests source overlap, without validating both endpoints. For example, with core laughter at source 132s, post-pad interval [12,20] maps to [131,139], lasts 8s and contains 132: it passes the containment alternative despite exceeding the 17s clip. | Validate finite point/interval times and ordered endpoints inside the actual clip before mapping. Invalid time must not satisfy G7/G8 or turn into a timed mismatch; preserve the semantic claim but mark timing unverified. Define multiple-claim selection so a valid later candidate cannot inherit an invalid earlier time. |

R1 is a direct false-credit path under the written rule. R2/R3 are missing
decision semantics, not evidence of actual model failures. Worked cases must
show the exact resulting gate/reason for positive, negated, uncertain, prompted,
out-of-bounds and reversed-time inputs. Keep the existing eight-clip question
and remaining protocol stable while closing these findings. No further broad
redesign, model survey or new musical annotation is requested.

Current execution gates remain closed: no demonstrated feasible compute route,
unverified manipulation control, and R1–R3 unresolved. Team 2's page/runtime
repair assignment is complete; Team 3 owns the bounded revision. Team 1 reviews
that handoff next. The live request and acceptance states are in `CONTINUE.md`.

## Team 1 design closure — 2026-09-14

**Accept doc 27 revision 3 as the bounded development-experiment design.**
Reviewed SHA-256:
`a0d7bb953cdeccad95457dd2a5af480ec25d2d98f6150e232be8a9a7b31453c9`.
R1–R3 are closed. Team 3's assignment is complete; no further design repair is
requested. This accepts the rules for the eight-clip vocal-manner test, not a
completed preregistration, demonstrated model benefit, or authorization to run it.

The lead read revision 3 and traced its rules against the returned counterexamples.
A bounded Terra reviewer independently traced **all 11 new rows** (W13, W14,
W15, W15b, W16, W17, W18, W19, W20, W21, W21b), including boundary cases.
Its review did not claim a fresh trace of the 16 older rows. Exact closure:

| Finding | Accepted rule | Checked outcomes |
| --- | --- | --- |
| R1 | Section 4.2's T5-full-A requires the laughter, lead-delivery and ordering evidence from Stage A alone; G6 requires that grade. | A laughter-only plus B preceding rap reaches UNRESOLVED(prompted-only); the whole transition in A can pass. |
| R2 | Section 4.3 records assertion polarity; only affirmative claims earn credit. G4u/G10c handle uncertainty, and G10a/G10b align with the source-agnostic audibility check. | Explicit absence in ablation passes; uncertain core/ablation claims reach the stated UNRESOLVED reasons; affirmative audible laughter attributed to a sample still fails the valid ablation. |
| R3 | Section 4.3 validates finite, ordered times inside the actual clip; sections 4.2/4.4 retain separate candidate times; G7/G8 use only valid candidates. | Out-of-bounds padded time is unverified; reversed core time cannot pass G7; a valid candidate survives alongside an invalid one; invalid padding cannot rescue a valid mismatch. |

The fixed English cue lists and ambiguous source nouns remain declared scoring
limitations. Outcome-changing ambiguity goes to the protocol's scorer-resolution
rule; neither model nor scorer agreement supplies auditory ground truth. The
scoped R1–R3 review found no remaining consequential defect in these fixes.

**Nonblocking validation-accounting caveat:** doc 27 section 8 says "all 22
scenarios" while the actual table has **27 unique rows** (16 older + 11 new).
The lead counted the row IDs directly; Team 1 does not adopt the unsupported
22/22 coverage claim. Its independently checked closure evidence is the 11 new
rows above. The owner can correct the historical accounting on future maintenance;
this does not justify reopening the design or inventing unrecorded test coverage.

### Reconciled execution readiness

The three requested deliverables are now accepted within scope: MuQ comparison,
baseline evidence page 10, and semantic design; doc 26 is accepted as the runtime
feasibility audit with a no-route result. None of these is evidence that the
semantic model has run. The remaining gates are:

1. **Compute route:** doc 26 establishes no feasible execution route for the
   current unquantized, all-on-GPU proposal. Driver repair alone cannot solve
   its capacity mismatch. Another precision/offload/hardware route needs a
   separately scoped feasibility decision and evidence before setup or download.
2. **Manipulation validity:** no pre-output human check of the re-summed and
   no-vocals clips exists. The accepted protocol therefore cannot yield SUPPORT
   yet. No clips were exported and no new listening labels were requested here.
3. **Preregistration:** before any future generation, bind the accepted design,
   model/runtime, exact clip samples, order and check procedure as G0 requires.
   Doc 26's illustrative systems smoke prompt is not a substitute for this test.

There is no outstanding counterpart review or repair to run now. The next
decision is which compute route, if any, to investigate in a separate bounded
task; do that before spending on setup or preparing the manipulation audition.
No paid resources, audio transfer, driver changes, environments, downloads or
inference are assigned by this closure. `CONTINUE.md` records all assignments
closed and the unresolved execution gates.

Validation this turn: read-only rule/counterexample review, scenario-ID count,
unchanged Team 2 handoff hashes and `git diff --check`. No application tests,
browser rerun, audio/model execution or new experimental outputs were needed;
the earlier page/MuQ validation remains historical.

## Team 1 lower-memory feasibility screen — 2026-09-14

**Metadata/source review complete; CPU-only 4-bit is a candidate for an isolated
synthetic compatibility test, not a demonstrated Music Flamingo execution route.**
The user asked Team 1 to continue the lower-memory investigation. This screen
does not reopen accepted Team 2/3 handoffs. No packages were installed, model
tensor payload downloaded, drivers changed, audio transferred or inference run.

Evidence is in `outputs/reviews/music-representation-lowmem-audit-01/`:
`worker/` retains the original hardware/header audit; `lead/metadata/` contains
eight public package-metadata/source snapshots and their URL/size/SHA manifest;
`linear-review/` records the corrected tensor classification. Original worker
artifacts remain unchanged; their hypothetical all-parameter estimates must not
be used as an executable bitsandbytes memory budget.

### Corrected memory accounting

Pinned model: `nvidia/music-flamingo-2601-hf`, revision
`6b5be086d52f65a1e204cb0faf70bf54e2741ecd`. The header reports 830 BF16 tensors,
8,267,215,360 elements and 16,534,430,720 tensor bytes. Adding the 100,776-byte
header and eight-byte prefix reproduces the 16,534,531,504-byte file size.

The proposed conservative route quantizes **only the decoder-block Linear
weights**. It explicitly preserves the audio tower, projector, token embeddings,
untied LM head, norms and biases in BF16. Transformers' replacement operates on
Linear modules, not every parameter. Explicitly exclude the LM head along with
audio/projector modules: supplying custom exclusions need not retain default
exclusions. Modern model module paths also differ from some checkpoint keys;
actual replacement and checkpoint-renaming coverage must be checked in setup.
[Pinned replacement implementation](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/integrations/bitsandbytes.py),
[exclusion handling](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/quantizers/base.py).

| Selected storage assumption | Weight-only GiB |
| --- | ---: |
| All BF16 | 15.3989 |
| Decoder Linear weights at ideal packed 4-bit, everything else BF16 | 6.2832 |
| Same, with FP32 scales per 64 values | 6.6630 |
| Same, approximate nested scales (one byte/64 plus four bytes/16,384) | 6.3796 |

These formulas use 6,525,288,448 quantized elements and 3,483,853,824 fixed BF16
bytes (3.2446 GiB). The two vocabulary tensors alone account for 2.0250 GiB.
Codebooks, rounding, metadata, conversion temporaries, activations, attention/KV
cache, framework memory and allocator reserve are excluded. This corrects the
original worker's 4.9855-GiB hypothetical decoder estimate, which quantized
embeddings/head/norms/biases too and assumed FP16 rather than FP32 ordinary
scales. It was not a supported module conversion plan.

Even ideal packed weights for this chosen route exceed a nominal 6-GiB GPU;
driver repair cannot make this all-on-GPU variant fit. This does not rule out
every possible, more aggressive quantization/offload scheme. CPU steady storage
could fit the audit's 10.24-GiB available RAM, but loading/inference peaks remain
unknown. Host totals from byte counts are 30.32 GiB RAM and 39.55 GiB free disk;
the earlier approximate 31-GiB RAM wording is not an exact binary-unit count.
Swap was almost exhausted. Do not load a full BF16 state dict or rely on swap.

### Why a CPU test is worth considering

The current bitsandbytes installation guide supports Linux x86-64 AVX2 CPUs and
glibc >=2.24. This host reports AVX2 and glibc 2.35. Public package metadata for
Transformers **5.17.0**, bitsandbytes **0.50.2** and Accelerate **1.15.0** allows
Python 3.10; these are inspected candidate versions, **not installed versions**.
[CPU platform requirements](https://huggingface.co/docs/bitsandbytes/main/en/installation#cpu).

Transformers 5.17.0 contains the Music Flamingo class and its 4-bit quantizer
explicitly permits an all-CPU device map. This differs from mixed CPU/GPU
offload, whose exclusions can leave offloaded modules unquantized. Source
presence and declared requirements are not an import/load test.
[Pinned quantizer](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/quantizers/quantizer_bnb_4bit.py).

This CPU lacks AVX512-BF16, so the inspected fused CPU 4-bit path is unavailable.
The default matrix path dequantizes a layer before its floating-point linear
operation. The largest selected layer is 67,895,296 values: 129.5 MiB BF16 or
259 MiB FP32 when expanded, before other temporaries. Memory savings therefore
do not imply usable generation speed; no latency estimate is established.
[CPU dispatch](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/backends/cpu/ops.py),
[default dequantization/matrix path](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/backends/default/ops.py).

### Next bounded task — needs setup authorization

Team 1 should ask the human to authorize **an isolated CPU-only environment and
synthetic runtime tests only**, before any model-weight download. Proposed new
paths are `.songviz/semantic-cpu-probe-venv/`,
`.songviz/semantic-cpu-probe-cache/`, and a new
`outputs/reviews/music-representation-cpu-smoke-01/` evidence package. Preserve
both existing project environments. No global installs, system packages, driver
repair, paid compute or external audio are part of this proposal.

After authorization, resolve and record CPU-only PyTorch plus the candidate pins,
package hashes and disk use; stop if installation requires a source build or
system changes. Use offline model access during synthetic tests. Import the exact
class, verify the intended replacement/exclusion mapping with a tiny local
configuration, and test random representative Linear weights with NF4 and nested
scales. Compare BF16/FP32 compute, recording finite outputs, numerical deviation,
conversion/forward time and peak RSS. Start small, then test the actual largest
decoder-layer shape in an isolated process with a 120-second timeout and 2-GiB
RSS stop threshold. These limits are proposed test guards, not measured demands.
Do not instantiate the full model merely to inspect module names.

Only a successful synthetic screen should trigger a separate decision about the
15.40-GiB checkpoint download and measured full-loader smoke. A tiny test cannot
prove full-model loading capacity, acceptable song-output latency or semantic
quality. Quantization must be registered as a different runtime condition, not
silently treated as the publisher's BF16 result. Doc 27's G0 and pre-output human
manipulation check remain required before semantic study outputs; no new
listening task is needed for this synthetic screen.

## Team 1 CPU synthetic compatibility screen — 2026-09-14

**Complete.** With the user's setup authorization, Team 1 created only the
proposed Python-3.10 virtual environment and cache, installed binary CPU wheels,
and ran an offline synthetic screen. No model tensor payload, song/stem/audio
input, driver/system change, remote service, or paid resource was used. Evidence:
`outputs/reviews/music-representation-cpu-smoke-01/`.

The exact installed core set is Torch `2.10.0+cpu`, Transformers `5.17.0`,
bitsandbytes `0.50.2`, and Accelerate `1.15.0`; `pip check` passed. The harness
confirms imports of both Music Flamingo symbols and verifies a tiny
path-mirroring module: decoder Q/K/gate projections become `Linear4bit`, while
the LM head, audio tower, and multimodal projector remain ordinary Linear
modules. This is module-path/replacement evidence only, not a constructed Music
Flamingo model.

Both deterministic random-weight NF4/nested-scale cases produced finite outputs.
The largest real decoder Linear shape (`18944 × 3584`, output by input) finished
within the externally monitored 120-second / 2-GiB-RSS guard: 4.679s wall,
1,811,001,344-byte peak RSS (1.687 GiB). It is therefore close to the guard,
not evidence of headroom. The harness, result and report SHA-256 values are in
the package report; it never calls `from_pretrained` and forces offline mode.
Host Datadog instrumentation injects paths even through `env -i`; the result
records that state, so this is not presented as an instrumentation-free process.

This closes the isolated synthetic task, but does not establish full checkpoint
load capacity in the observed roughly 10–11 GiB available RAM, full-model peak
RSS, usable song latency, output quality, audibility, G0 preregistration, or
semantic validity. The only plausible next compute action is a separately
authorized checkpoint download and bounded full-loader smoke; it is not assigned
by this result, and must not include song inference.

## Team 1 full-loader smoke — 2026-09-14

**Checkpoint downloaded, verified and CPU-loaded; no forward executed.** The
user authorized the CPU full-loader feasibility task. Team 1 downloaded the
pinned public checkpoint only into `.songviz/semantic-cpu-probe-cache/hf/` and
verified `model.safetensors` at 16,534,531,504 bytes with SHA-256
`4da26452d3978503fb064023a69f7867a1f235a2a5a89d3d9146ebf2e04eb15b`.
The final staged snapshot has seven model/config/tokenizer files. No source audio,
stem, prompt, processor call, model forward, generation, CUDA/driver work or
remote resource was used.

An independent Terra preflight fixed fail-closed loader gates: at least 12 GiB
`MemAvailable`, 1 GiB free swap and 21 GiB disk remaining after download, then a
fresh offline CPU-only subprocess with 8.5-GiB RSS / 10-minute guards. After the
user freed memory, the host passed at 21.2 GiB available and 1.30 GiB swap free.
The loader completed in 3.823s wall / 0.345s model-load time with a 417,705,984
byte peak RSS; all 830 parameter tensors were CPU BF16, no parameter remained
meta, and the count was 8,267,215,360. It immediately exited without processor,
audio, prompt, forward or generation. Exact result and hashes are in
`outputs/reviews/music-representation-full-loader-01/`.

The low RSS is consistent with safetensors memory mapping/lazy page residency;
this is evidence that the loader resolves the checkpoint, not that all 15.40 GiB
of weights can become resident or that an audio forward fits/has useful latency.
It does not authorize audio inference or the semantic study. A first-forward
feasibility probe remains a separate decision.

## Team 1 synthetic first-forward feasibility — 2026-09-14

**Complete — the CPU BF16 route did not finish a full synthetic forward within
the declared RSS guard.** Team 1's Terra worker owned the new package
`outputs/reviews/music-representation-first-forward-01/`; the lead independently
reviewed its report, JSON invariants, compilation and hashes. This task used only
the already verified checkpoint with offline CPU flags. It did not use source
audio/stems/user audio, a processor/tokenizer, text prompt, label, generation,
CUDA or network access.

The final valid deterministic call used all-zero BF16 features `[1,128,2999]`,
an all-one feature mask, and 750 copies of the configured audio-placeholder ID
151667. The shape is required by the actual encoder's 1,500-position table and
matches its 750 pooled audio rows; the smaller rejected shape is preserved as
development evidence, not silently replaced. The child loaded 830 CPU BF16
parameter tensors (8,267,215,360 parameters; no meta tensors), completed the
audio tower in about 43.367s and the projector in about 0.383s, then entered the
language model.

The independent 0.05s monitor sent SIGKILL at the 8.5-GiB RSS bound. It stopped
at 136.314111s and 9,128,382,464 bytes, 1,576,960 bytes over the exact
9,126,805,504-byte threshold; return code was -9, not a timeout. The language
model, LM head and outer forward did not return. There are no logits, loss,
caption, song claim or semantic result. This is a measured negative feasibility
result for the pinned BF16 CPU route, not proof that every possible quantized,
offloaded or different-hardware route fails. It does not authorize an audio run
or the semantic design. Any alternative runtime route needs a new bounded
decision rather than a retry at a larger guard.

## Team 1 private Modal serverless route — planned 2026-09-14

**Status: private health/authentication and L4 checkpoint loading are verified;
the latest synthetic POST is transport-inconclusive.** The user selected a
private Modal serverless endpoint as the next candidate runtime condition. The
first authorized, $1-capped deploy exposed a worker source-import defect; a
second, FastAPI-only-health diagnostic corrected it and returned authenticated
health in four seconds while unauthenticated access was rejected. The first L4
probe exposed a missing-librosa audio dependency; the next explicitly authorized
configuration installed it and loaded all checkpoint shards, but initially
returned an empty HTTP 303 result handoff whose `Location` was not captured
before teardown. A subsequent separately authorized one-POST silent-WAV run
returned terminal HTTP 200 after the client transport safeguard was tested. This
is deliberately not a claim that model output will be semantically useful or
that approved song audio has been evaluated. It
supersedes neither the pinned CPU failure nor the semantic-study gates. Evidence
is in `outputs/reviews/music-flamingo-modal-01/`,
`outputs/reviews/music-flamingo-modal-health-02/`,
`outputs/reviews/music-flamingo-modal-l4-03/` and
`outputs/reviews/music-flamingo-modal-l4-librosa-04/`, with completion evidence
in `outputs/reviews/music-flamingo-modal-result-05/`.

### Objective and non-objective

The bounded first outcome is a reproducible, private HTTP wrapper around the
pinned `nvidia/music-flamingo-2601-hf` revision
`6b5be086d52f65a1e204cb0faf70bf54e2741ecd`, with a local contract test and a
later, separately authorized remote feasibility package. The wrapper will accept
one audio payload plus an explicit analysis prompt, and return structured JSON
containing the model response and machine-readable runtime metadata.

It is not a public product endpoint, a production service, a permission to send
the repository's source song or stems to an external provider, a semantic-study
run, or a replacement for G0/pre-output audibility requirements. The model card
marks this checkpoint `License: other` and includes an NVIDIA one-way
noncommercial license file; before any product/commercial use, the applicable
license must be read and accepted separately.

### Team 1 scope and delegation

Team 1 owns the endpoint design, cost/security gates, evidence review and
integration. It may delegate ordinary wrapper implementation and local tests to
a Terra worker within the exclusive new paths below; the Team 1 lead must inspect
the diff and results. Team 2 and Team 3 have no task in this route and must not
duplicate its wrapper, change its model contract, or start a remote service.

Allowed new paths for the implementation are:

- `experiments/modal_music_flamingo/` — Modal app, request/response schema,
  local helper and deployment/teardown documentation;
- `tests/test_modal_music_flamingo*.py` — no-credential contract and safety
  tests; and
- `outputs/reviews/music-flamingo-modal-01/` — immutable evidence from the
  first authorized remote feasibility decision.

No token, account identifier, audio payload, model-generated text from private
audio, or Modal secret may be committed. Secrets must come from the user's
Modal/Hugging Face configuration at runtime only.

### Endpoint contract and safety defaults

The initial API must be private/authenticated, accept one bounded audio file and
one explicit prompt, allow one concurrent worker, enforce a request timeout, and
expose a health/readiness route that returns no model or credential data. It must
reject unsupported file size/duration/type before work is scheduled. The client
must retry a cold-start response only within an explicit overall deadline; it
must not issue duplicate analysis calls.

Use one GPU worker at most and configure a short scale-down window. Modal bills
GPU/CPU/RAM usage while the container runs and provides scale-to-zero behavior;
the initial implementation must make that lifecycle explicit rather than rely on
the user to remember to stop a pod. Do not create a permanent warm worker,
public URL, background queue, multi-worker autoscaling rule, persistent cached
model volume, or external log sink in the first route. A later cache decision
needs measured cold-start and storage-cost evidence.

The first GPU candidate may be a 24-GiB NVIDIA L4/A10-class worker because it is
the lowest-cost compatible class under consideration, but it is only a bounded
feasibility attempt. An out-of-memory result is a valid stop result. Moving to a
larger GPU, enabling quantization/offload, changing the model revision, or
raising a timeout is a new decision—not an automatic retry.

### Sequenced execution and gates

1. **Local implementation only.** Build the private Modal app, schema,
   authentication/secrets plumbing and scale-down/one-worker configuration.
   Run syntax, request-validation and no-credential tests only. No `modal deploy`,
   `modal run`, model download, account login, cloud API call or paid GPU is
   allowed in this phase.
2. **Lead review.** Inspect the worker's actual patch and local tests. Verify
   that no secret is present, the pinned revision is explicit, unsupported input
   fails closed, and no route can become public accidentally. Record commands
   and a deliberate scale-to-zero/teardown command.
3. **Human remote authorization.** Before deployment/invocation, obtain the
   user's Modal access method and a stated small spending limit. Record the
   chosen GPU type, region if exposed, per-second rate visible in the account,
   request timeout and cost-stop threshold in a fresh evidence package. Cloud
   dashboard budget alerts are useful but not assumed to be a hard stop; the app
   limits are the primary guard.
4. **Remote feasibility only.** First verify private authentication, readiness,
   exact model revision and a deterministic synthetic/health path. Capture
   cold-start time, model-load result, peak GPU memory if exposed, request
   runtime, structured output validity, scale-to-zero behavior and actual billed
   cost. Stop on load/import/VRAM failure, timeout or cost threshold; preserve
   the failure without silently altering configuration.
5. **Separate audio decision.** Only after the feasibility package is reviewed
   may the user explicitly authorize upload of a bounded source-song excerpt.
   Store only provenance, hashes, timings and derived evaluation results in the
   repository unless the user expressly authorizes retaining output text/audio.
   No semantic claim or visualization policy is promoted from one response.

### Local wrapper implementation — complete 2026-09-14

Terra implemented the bounded code in `experiments/modal_music_flamingo/` and
`tests/test_modal_music_flamingo.py`; the Team 1 lead inspected the actual
implementation and requested a correction before acceptance. The accepted
wrapper is import-inert unless `SONGVIZ_MODAL_ENABLE_APP=1` is deliberately set.
It neither imports the Modal SDK nor imports/downloads model weights during local
tests. The deployment definition uses `requires_proxy_auth=True`, one L4 worker
at most, zero minimum/buffer workers, a 60-second scale-down window, a 300-second
   request/startup ceiling and a pinned public revision. The checkpoint was
   previously downloaded from its public repository without an HF token, so this
   first route deliberately has no Hugging Face account/secret requirement.

The pure-Python schema accepts exactly one base64-encoded, uncompressed PCM WAV
(`audio/wav`), at most 20 MiB/60 seconds, one or two channels and 8–48 kHz, plus
one nonempty prompt of at most 1,000 characters. Unknown fields and malformed or
out-of-bound inputs fail before inference. The authenticated health route is
intentionally not a model-ready claim. The caller helper retries only health
503s within a deadline and submits an analysis POST exactly once.

The model/processor are now lazy-cached once per warm Modal container: the first
authorized analysis call records a cache miss and the elapsed wall time; later
calls in that same 60-second container reuse the in-memory runtime. The cache is
not a Volume and disappears on scale-down. This correction prevents silently
reloading roughly 16.5 GB of weights for every request while retaining no
cross-container model state.

Lead validation executed:

```bash
.songviz/representation-venv/bin/python -m pytest -q tests/test_modal_music_flamingo.py
.songviz/representation-venv/bin/python -m py_compile experiments/modal_music_flamingo/*.py
git diff --check
```

The focused suite reported **13 passed, 1 existing ddtrace pytest warning**;
compilation and whitespace checks passed. The offline tests use a fake Modal
surface and injected fake model loader to prove private endpoint options,
revision pinning, scale settings, input rejection, inert import, no-duplicate
POST behavior, and cache reuse. They do not prove current SDK deployment,
provider authentication, image build, model download, GPU memory fit, model
output, cold-start time, scale-down observation, or billed cost. No Modal
SDK/account/login/deployment, GPU, model load/download, secret, real audio or
network action occurred during this work.

### Acceptance for the first remote package

`outputs/reviews/music-flamingo-modal-01/` must contain the pinned revision and
package versions; redacted deployment configuration; input/output schema;
executed commands; timestamps; chosen GPU; exact request and lifecycle limits;
observed load/forward/scale-down outcomes; actual cost; and hashes of source
files. It must clearly distinguish synthetic, health, and any later authorized
audio calls. A successful HTTP response alone is insufficient: a valid result
must be attributable to the exact revision and request, preserve uncertainty,
and remain separate from semantic validity.

Official references for the intended platform behavior are Modal's
[pricing](https://modal.com/pricing) and
[GPU guide](https://modal.com/docs/guide/gpu). These were checked for planning;
the account-visible configuration/rate at the time of authorization is the
authoritative cost evidence.

### First authorized remote feasibility — stopped 2026-09-14

The Team 1 lead ran the authorized private deployment with the account's $1
usage limit intact. The current Modal SDK first rejected the image before any
remote worker because FastAPI was no longer implicitly installed for a
`fastapi_endpoint`. A narrowly scoped Terra implementation added the explicit
`fastapi==0.115.14` image dependency; lead inspection then reran the focused
suite (**13 passed, 1 existing ddtrace warning**), compilation and whitespace
check. The corrected image deployed as one private app with `analyze` and
`health` endpoints, the pinned L4 settings, no volume and no warm worker.

The endpoint was never made public. An unauthenticated route attempt was denied
for missing proxy authorization. An ephemeral proxy token was generated only to
test health, held in a shell process, and revoked automatically at exit. The
redirect-following authenticated health call did not return a health payload
within more than three minutes and was cancelled; because health has neither a
GPU option nor a model call, this observes only the image/cold-start/request
path. There was no `POST /analyze`, no synthetic/source/user audio, no remote
checkpoint download, no model load/generation and no GPU use. The lead stopped
the exact app and verified zero tasks and no remaining proxy tokens.

Modal's observed billing report at teardown listed only CPU/memory:
`$0.00858076` total across the two app objects, no GPU resource. The Dashboard
rounded the result to `$0.01 / $1` with `$0.99` credits remaining. This may
settle later and is cost evidence, not a bill guarantee. The frozen evidence
contains image/app IDs, redacted commands, source hashes and stopping result:
`outputs/reviews/music-flamingo-modal-01/report.md`. Do not redeploy or alter
hardware/model configuration automatically. A future attempt requires new
human approval for a distinct bounded diagnostic, ideally private health on a
small image before any L4/model call.

### Health-isolation diagnostic — complete 2026-09-14

The human authorized that new private, capped diagnostic. Team 1 separated
health into a FastAPI-only image (no Torch/Transformers/SoundFile/model/GPU
runtime) and kept `analyze` on its existing L4 image. The first call to the
small route still timed out, but the stopped-app logs established the cause:
Modal's worker re-imports mounted source without the deploy shell's
`SONGVIZ_MODAL_ENABLE_APP` variable, so `health` had not been defined. The
source now uses Modal's worker-only container-arguments marker to define the
global app/functions on worker import while remaining inert for normal local
imports. Focused tests now include that import path: **14 passed, 1 existing
ddtrace warning**; compilation, opt-in endpoint-global import and whitespace
checks passed.

The corrected private deployment returned authenticated `GET /health` in four
seconds with `{"status":"ok","ready":false,"schema_version":"songviz.music_flamingo.v1"}`.
A no-credential request received HTTP 401 before endpoint execution. The exact
app was stopped afterward; Modal subsequently reported zero tasks, zero
containers and zero retained proxy tokens. No L4/GPU resource, model load,
checkpoint download, `POST /analyze`, synthetic/source/user audio or model
output occurred. The successful app's observed CPU+memory cost was
`$0.00003072`; all four diagnostic apps totalled `$0.00862300` at report time,
with no GPU resource. Evidence/provenance: 
`outputs/reviews/music-flamingo-modal-health-02/report.md`.

The health/authentication gate is now closed. Do not automatically invoke the
L4 endpoint: its one deterministic silent-WAV request, model-load/VRAM
observation and immediate teardown require distinct human approval. Source-song
upload remains a separate approval after that feasibility result.

### One-request L4 synthetic probe — stopped 2026-09-14

The human then authorized exactly that L4 feasibility call. Team 1 generated a
one-second 16-kHz mono 16-bit PCM silent WAV in memory, used the non-semantic
prompt `Reply with exactly: OK`, issued one private POST without redirect or
retry, and stopped the app immediately after the response. There was no source
song/stem/user audio and no retained model response. The call returned HTTP 500
after 99 seconds (98.1 seconds in Modal's log). All **830** checkpoint shards
loaded on the L4, so this is positive evidence for this pinned remote loader;
it is not a successful audio/model forward.

The precise exception arose at `processor.apply_chat_template`, before feature
construction or `model.generate`: Transformers required `librosa` to load the
WAV, but the pinned image lacks that package. No OOM, model generation, logits,
caption or semantic claim occurred. The generic response body and its hash,
full redacted provenance, hardware/cost and exact stop verification are in
`outputs/reviews/music-flamingo-modal-l4-03/report.md`. The observed L4 cost
was `$0.02444455`; this app totalled `$0.03103089` and all diagnostic apps
totalled `$0.03965389` at teardown, with the $1 cap unchanged. Zero tasks,
containers and proxy tokens remained.

This closes the one-request configuration. Do not silently add `librosa` or
retry. A new explicit decision must choose/pin the compatible audio dependency,
add an image-contract test, and authorize exactly one new synthetic L4 call;
source-song upload remains separately prohibited.

### One-request L4 + librosa synthetic probe — inconclusive 2026-09-14

The human explicitly authorized the next configuration: Team 1 pinned
`librosa==0.11.0` only in the heavyweight ML image, added the fake-image
contract that forbids it in the private health image, and passed the focused
suite (**14 passed, 1 existing ddtrace warning**) plus compilation/import/diff
checks. One generated silent-WAV L4 POST then ran directly with no redirect
following or retry. The updated image installed librosa and the stopped-app log
shows all 830 checkpoint shards loaded, with no repeat of the former audio
loader exception.

The HTTP client received an empty **303** after 151 seconds. Modal documents
that Web Functions return a 303 after their 150-second HTTP wait limit, with a
`Location` that refers to the original request's result and may be fetched with
GET without reissuing the POST ([Modal request timeouts](https://modal.com/docs/guide/webhook-timeouts)).
The first client did not retain that `Location`, and the app was stopped rather
than guessing at a redirect target. The logs contain no generation completion,
output, logits, audio feature evidence or terminal model exception before
teardown. Therefore this is not accepted as a successful end-to-end forward; it
is an inconclusive transport/result condition after successful image installation
and model load.
No source/song/user audio or model text was retained. The app later reported
zero tasks/containers/tokens. L4 cost was `$0.03688908`; this app totalled
`$0.04772916`, observed diagnostics `$0.08738305`. Full evidence:
`outputs/reviews/music-flamingo-modal-l4-librosa-04/report.md`.

### Private L4 synthetic completion — passed 2026-09-14

The human authorized one final bounded synthetic feasibility attempt after the
transport contract was tested. The client first received private health HTTP
200, submitted exactly one one-second silent-WAV private POST with `Reply with
exactly: OK`, and received terminal HTTP 200 directly. The response was 350
bytes and its SHA-256 was retained, but neither response text nor any source or
user audio was retained. The client made no POST retry and no result-handle GET
was necessary.

The stopped-app log records all 830 weight shards loaded and `POST / -> 200 OK`
at 122.3 seconds total / 117.6 seconds execution. The app then stopped with
zero tasks, containers and proxy tokens. This is positive feasibility evidence
for the pinned endpoint/image/L4 condition, including preprocessing and a
terminal model response. It says nothing about semantic validity, the quality
of the response, musical-role understanding, source-song behavior, or the
visualization. The run cost `$0.03538923`; observed diagnostics totalled
`$0.12277228` at teardown. Full bounded evidence:
`outputs/reviews/music-flamingo-modal-result-05/report.md`.

The remote endpoint feasibility gate is now closed. Do not send source-song
audio or additional requests automatically. Any semantic evaluation needs a
separate audio-approval and predeclared evaluation decision.

### Semantic-probe preregistration prepared locally — 2026-09-14

Team 1 then prepared the no-cost, no-run package
`outputs/reviews/semantic-probe-01/`. It binds the accepted doc-27 revision,
the original-mix and four-stem input fingerprints, the eight required R clips,
their deterministic `random.Random(20260914)` file order, fixed Stage-A/B
prompts, greedy decoding/retry rule and blank independent-scorer records. With
the explicit local-only clip mode, it generated eight 44.1-kHz stereo PCM-24
WAV files with recorded SHA-256/frame bounds: original-mix clips are exact
integer-frame cuts, and the two ablation clips are exact sums of their declared
stems. There was no upload, network request, model execution, model text,
source mutation or new listener label.

Lead replay checked every generated WAV's hash, format, frame count and sample
values against its declared source recipe; all eight matched. The focused builder
suite passed **6 tests** (plus the existing ddtrace warning); compilation and
`git diff --check` passed. The preparation is not a semantic result and does not
complete G0 yet: a locally random, recorded blind listener order now exists, but
the listener must answer `yes` / `no` / `unsure` to “Can you hear laughter?” for
each of its two files before any model response is viewed. After that, a separate decision must authorize
external transfer of the eight bounded clips and specify the total spend cap.

### Blind vocal-ablation check — control invalid 2026-09-14

Before any model output, the listener played the recorded blind order from
`semantic-probe-01`: `clip_01.wav` then `clip_04.wav`. Their raw observations,
preserved in the package's manipulation-check record, were laughter after about
3 seconds in the first clip and a quieter but audible laugh after about 7
seconds in the second. The internal mapping is resum first and no-vocals second.
Thus the normalized answers are `resum=yes` and `novocals=yes`; doc 27 requires
`resum=yes` **and** `novocals=no` for a valid control.

This does not identify which stem carries the residual audible cue or prove a
source-separation defect. It does establish that the specified no-vocals
ablation cannot test the intended counterfactual for this listener. Under the
frozen gate G9, a model run using this package cannot reach SUPPORT and would be
`UNRESOLVED(control-invalid)`. Team 1 therefore closed the external-upload and
paid-inference gate without sending any clip. Any replacement control needs a
separate design decision and a new preregistration package; it must not overwrite
this result.

### Independent vocal-ablation candidate — 2026-09-15

To test a replacement control without remote audio transfer, Team 1 did a
fresh, bounded local separation of the already-selected `119–132s` excerpt.
This is not a rerun of the old historical comparison and not a full-song model
replacement. The candidate is Kimberley Jensen's two-stem Mel-Band RoFormer
vocal model, selected because its author-published checkpoint record gives a
clearer license/provenance chain than the previously tested Viperx and
Bleedless community checkpoints. The exact source checkpoint was downloaded
from `KimberleyJSN/melbandroformer`, verified as
`87201f4d31afb5bc79993230fc49446918425574db48c01c405e44f365c7559e`, and
paired with the recorded MSST-compatible configuration
`f63f38eb1e6e40a7db0dade714a5ae257555dd8748f4e774eae8679275a81926`.

The local run used `audio-separator 0.41.1`, CPU only, four numerical threads,
MDXC overlap 2 and segment size 256. The 13-second PCM-24 input and both
outputs are precisely 573,300 frames at 44.1 kHz stereo. It completed model
separation in 22 seconds (28.12 seconds wall including startup) at 2,596,092
KiB peak RSS. The output hashes, all command settings, source/model/config
fingerprints, first failed CLI invocation (`--log_level 20` is not a valid
level) and successful attempt log are preserved in
`outputs/reviews/stem-separation-control-01/manifest.json` and its sibling
logs. The model does not identify a truth stem: its `other` output is only a
candidate independent no-vocals control.

`outputs/reviews/stem-separation-control-01/index.html` is a small labelled
diagnostic page with the original excerpt, the known-invalid four-stem Demucs
no-vocals clip, and the candidate `other` and vocals outputs. It is deliberately
not the acceptance test: labels make it useful for diagnosis but invalidate it
as a blind manipulation check. A new randomized listener order must retain the
original/resum positive condition and ask whether laughter is audible in the
candidate `other` clip. Only `resum=yes` and `candidate-other=no` validate this
replacement control. A pass would justify a separate, new preregistration
package; it would not retroactively repair semantic-probe-01. A failure leaves
Music Flamingo semantic probing closed for this design. No source audio was
uploaded, no remote model request was made, and no full-song separation was
performed.

That blind check was prepared as `blind/index.html` with neutral clip names and
a one-byte `/dev/urandom` draw recorded in
`candidate-manipulation-check.json`. The draw placed `candidate-other` first
(`clip_a`) and the existing resum-positive condition second (`clip_b`).

### Replacement-control blind check — passed 2026-09-15

The listener then reported exactly `A: no, b: yes`. The recorded mapping makes
that `candidate-other=no` and `resum-positive=yes`; no time estimates were
provided. This satisfies the predeclared polarity rule in
`candidate-manipulation-check.json`, which now preserves both raw and
normalized observations and records `valid: true`.

The result validates the replacement listener-facing vocal-ablation control
for this bounded passage. It does **not** validate source-separation accuracy,
prove that all vocal material is absent, establish a musical interpretation,
change the frozen invalid result in semantic-probe-01, or authorize a Music
Flamingo call. If semantic probing is resumed, Team 1 must build and review a
separate, fresh `semantic-probe-02` package whose declared no-vocals condition
is this exact hash-bound `candidate-other` WAV; only a later explicit decision
can authorize its external transfer and spend.

### Semantic-probe-02 local preregistration — 2026-09-15

Team 1 completed that fresh package at
`outputs/reviews/semantic-probe-02/`, governed by the new narrow amendment
[`docs/28_semantic_probe_02_amendment.md`](28_semantic_probe_02_amendment.md).
It incorporates doc 27 unchanged except for the one declared stimulus change:
`verse-ending.novocals` is now the exact passed candidate `other` output rather
than the invalid Demucs bass+drums+other sum. The amendment, candidate audio and
completed blind-control record are individually fingerprinted in the package
manifest. All prompts, the eight ids, randomized order, scoring template,
decoding/retry rules and decision gates remain the doc-27 values.

The builder writes new PCM-24 WAV files in a new directory. Lead verification
confirmed all eight output hashes/formats/frame counts; each of the seven
unaffected clips is sample-identical to its semantic-probe-01 antecedent, and
the replacement package clip is sample-identical to the passed candidate WAV
(its WAV byte hash differs only because it is newly written under the package
filename). The focused builder suite is **8 passed, 1 existing ddtrace warning**;
the updated builder compiled and `git diff --check` passed. No source or derived
audio was uploaded and no Music Flamingo request/model output was generated.

This makes the local preregistration prerequisite complete; it does not itself
authorize external transfer or paid execution. A later explicit human decision
must either set a bounded spend/transfer authorization for this exact package or
stop semantic probing here.

### Semantic-probe-02 execution — refuted 2026-09-15

The human authorized the exact package with a $1 total cap. Team 1 deployed the
private, scale-to-zero L4 endpoint, created an ephemeral Modal Proxy Token, and
verified private health before sending source-derived audio. The runner checked
all eight package clip hashes first; it then made Stage A followed by the
explicit reconstructed Stage-B conversation for every clip. All initial pairs
triggered doc 27's mechanical retry criterion, so every clip consumed its one
allowed complete retry pair: 32 analysis POSTs total, with no individual POST
replay or unplanned prompt/clip. The app is now stopped with zero tasks and the
temporary Proxy Token is deleted.

The result is **REFUTE(positive-control)**. The selected retry Stage-B response
to `drum-entry.core` says “No change,” and the selected retry Stage-B response
to `transition-extent.core` likewise says “No change.” Either failure fails G3
under doc 27 before a verse-ending target may be credited. Each selected Stage-A
reply also exhausted `max_new_tokens`, while Stage B ended normally; no further
retry is permitted. The raw output includes inconsistent, unsupported genre,
tempo, instrumentation and vocal claims, so it is retained only as evidence.

Modal's immediate itemized report was L4 `$0.07777775`, CPU `$0.00912075`,
memory `$0.00137926`, total **$0.08827776**—below the authorized $1 cap, though
billing can settle later. Full response/provenance and teardown record:
`outputs/reviews/music-flamingo-semantic-01/report.md` and `run.json`. This
refutes useful semantic evidence from this model/protocol on this song; it does
not prove that every audio model is incapable of understanding music. Do not
retry or spend further on this model/protocol merely to obtain a better answer.

### Postmortem qualification — protocol fidelity unresolved 2026-09-15

At the user's request, Astra performed a read-only postmortem. Lead inspection
confirmed its two concrete findings: `app.py` used `MAX_NEW_TOKENS = 256` even
though the frozen package requires 400 per stage, and the retry call repeated
the unchanged Stage-A prompt instead of appending the package's required
timestamp-format suffix. Every initial/retry Stage-A and Stage-B pair in
`run.json` is byte-identical. The run record also does not bind a snapshot of
the effective deployed source/configuration, so current source can identify the
discrepancy but cannot by itself prove deployed bytes.

The generic, inconsistent outputs and both positive-control misses remain
observed evidence and are not suitable for SongViz decisions. But the earlier
strict `REFUTE(positive-control)` label is now qualified as
**UNRESOLVED(protocol-fidelity)**: the prescribed retry/decoding condition did
not occur, and a corrected retry might differ. This is not a reason to spend or
rerun automatically.

### Execution-contract audit complete — 2026-09-15

The promised local-only audit is complete. No endpoint was deployed or called;
no model, audio transfer, checkpoint download, or paid resource was used.
`app.py` now fixes the production generation limit at the frozen 400 tokens per
stage. The runner leaves an initial Stage-A prompt byte-for-byte unchanged and
constructs a retry as that prompt followed once by the frozen timestamp-format
suffix. Its Stage-B request carries the exact Stage-A prompt and returned text,
which the app reconstructs as user/assistant/user transcript. The runner now
also writes an `execution_contract` into any future `run.json`: fingerprints of
the package manifest/order/prompts and local runner/app/schema sources, the
pinned model/revision, deterministic decoding settings, retry details, and the
declared transcript shape.

Focused offline validation (`tests/test_modal_music_flamingo.py` and
`tests/test_run_semantic_music_flamingo.py`) passed **32 tests** with **one
existing ddtrace warning**; `py_compile` on the four changed/test modules and
`git diff --check` passed. The regression test states the historical unchanged
retry predicate and fails it; the transcript tests check both initial and retry
turns. Exact current source/package hashes and commands are retained in
`outputs/reviews/music-flamingo-execution-contract-01/report.md`.

This is a forward execution contract, not recovered remote provenance. The
historical `run.json` did not retain deployed source/config bytes, so the audit
cannot prove what was deployed on 2026-09-15 and cannot upgrade the historical
run from **UNRESOLVED(protocol-fidelity)**. A corrected paid run remained a
separate human decision; the human subsequently made that authorization, and
its distinct result is recorded below.

### Corrected semantic-probe-02 execution — refuted 2026-09-15

The human explicitly authorized one fresh semantic-probe-02 execution with a
$1 cap. Team 1 deployed the private scale-to-zero L4 app from the audited local
sources, created an ephemeral proxy token only in the invoking shell, and sent
the hash-bound package through the module runner. Two failed setup attempts
sent **no health request or audio**: one used obsolete proxy-token JSON field
names and one invoked the runner as a direct script rather than a package
module. Both temporary apps were stopped; their tokens were deleted. The
successful app was `ap-1B5UqeqA1IpZllQkjMO6yT`.

The successful `run.json` is SHA-256
`2351d927752c164e809119ef2c7de9a965834e883d15d949e3ce9092df6f5778`.
It retains eight R clips, the frozen package/source hashes, the pinned model
revision, greedy decoding with `max_new_tokens_per_stage=400`, the retry prompt
hash, and the explicit Stage-B transcript shape. Every initial pair was
mechanically retry-eligible; exactly one fresh complete retry was made for every
clip, yielding 32 POSTs. Initial and retry reply pairs differ for all eight
clips, unlike the earlier invalid execution. Several selected retry Stage-A
replies still stop at the correct 400-token maximum; this itself triggered the
allowed retry but does not authorize another one.

The two frozen G3 controls both fail on their literal selected Stage-B text.
`drum-entry.core` says, “No change. The arrangement stays constant,” and
`transition-extent.core` says, “No change. The arrangement stays constant,”
then calls the clip continuous with no transition. Neither answer affirms T1
(an entry) or T7 (a drop/breakdown); `no change` is the frozen explicit failure
case. G3 therefore yields **REFUTE(positive-control)** before the verse-ending
responses may earn any target credit. This is a result for this pinned
Music-Flamingo model/protocol on this song only; it does not establish that all
audio models lack musical understanding.

The immediate Modal report itemized the successful app as L4 `$0.13644458`,
CPU `$0.01243177`, memory `$0.00245573`: **$0.15133208**. The same-day report
for all SongViz Music-Flamingo apps was **$0.40735652**, below the authorized
$1 cap (billing can settle). At teardown all four historical/current apps were
stopped with zero tasks, and the workspace proxy-token list was empty. Full raw
answers, contract and cost/teardown record are retained in
`outputs/reviews/music-flamingo-semantic-02/`. The frozen stopping rule applies:
do not issue another semantic request for this model/protocol on this song.

### Text-first baseline-explanation candidate — 2026-09-15

With semantic probing stopped, Team 1 prepared a small no-cost comprehension
aid from already accepted evidence. The four cards in
`outputs/reviews/baseline-explanation-01/report.md` each separate: (1) what the
existing listener reported, (2) a narrow source-linked acoustic observation,
and (3) what that evidence does not establish. They cover the existing drum
entry, perceptual non-change, verse ending, and broad breakdown excerpts.

This is deliberately text-first: it neither creates another visual policy nor
asks the user to repeat the four song judgments. It makes no new musical label
or semantic claim. The next review is only whether the cards help distinguish
what changes, what continues, and what is unknown; that would be feedback about
the explanatory aid, not a new annotation of the song.

First review feedback was inconclusive. The user found the three categories “a
little clear,” but was unsure why the file was necessary or whether it compared
one thing against another. The candidate therefore has not demonstrated useful
comprehension support and must not be promoted into an interactive page.

#### Strategy reset following read-only Astra review

Historical decision, now followed by the authorized consolidation and benchmark
in doc 29. The direct four-case walkthrough was presented in the conversation;
the human authorized proceeding. Do not ask for that same decision again.

The next bounded action is not a framing-only card revision. The product goal
remains an automatic visual director that follows musical development; its
intermediate requirement is an inspectable account of what continues, changes,
and returns before any translation into visual choices. The present evidence
does not establish an automatic directing policy: the `none` case has stable
acoustic changes, transition windows mix disappearance and recovery, MuQ adds
no salience/function/identity advantage, and the corrected Music Flamingo
protocol failed its positive controls.

Before another artifact, model, or visual pass, Team 1 will present one direct
current-versus-target walkthrough in the conversation. For the four existing
examples it will show the desired human-level understanding, the system's
current measurement, and the missing capability. The proposed first bottleneck
is distinguishing meaningful musical development from ordinary within-passage
variation. The stop condition is the user's agreement or disagreement with
that destination—not another annotation or an implementation decision.

#### Historical three-lead proposal — superseded 2026-09-15

This proposal was not launched and is retained as decision history. The user
subsequently authorized one active Team 1 lead with bounded Terra/Luna workers,
merging the evidence inventory and development benchmark. Broad research is
deferred; a second strong model may review one concrete experiment. Current
design and validation are in [doc 29](29_development_benchmark.md), and the live
assignment is in `CONTINUE.md`.

The original proposal assumed two Astra leads and one Claude Opus lead. The work is
parallelizable only if each lead owns a different decision:

| Lead | Question / deliverable | Acceptance boundary |
| --- | --- | --- |
| Astra A | What is already known about the song? Build a source-linked temporal knowledge map from existing human annotations, recurrence, stems and local evidence. | Every statement has evidence or is marked unknown; no new label/model claim. |
| Astra B | What must a useful method distinguish? Define a development benchmark from the four existing listening cases plus retained section/motif records. | Tests meaningful development versus ordinary acoustic variation without treating the four cases as training truth. |
| Opus | What missing capability is worth testing next? Compare specific methods for vocal behavior, arrangement change, transition extent and importance. | Ranked bounded experiments with input/provenance, local/remote cost, acceptance and stopping rules; no installation or execution. |

Each lead can use direct Terra/Luna/Sonnet/Haiku workers for ordinary inventory,
implementation and test cycles. Terra should not be a fourth orchestration
layer: the work is presently research/design-heavy rather than a settled
implementation backlog. These historical workstreams are superseded; do not
launch them as active tasks.
