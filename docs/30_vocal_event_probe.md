# Local vocal-event probe

**Status correction — 2026-09-20:** the human challenged laughter recognition as
too song-specific for the project's next milestone. The lead agrees. The completed
probe and blank reference page remain useful diagnostics, but their follow-up is
parked. No timing labels are required to continue the project. Historical requests
for those labels below describe the previous handoff, not the current queue.
The general problem is understanding arrangement, continuity and changing musical
roles across songs; an audible laughter label establishes none of those by itself.
Current priorities are maintained only in `CONTINUE.md`.

## Decision — 2026-09-15

The human asked Team 1 to continue autonomously until completion or a real need
for human input. This extends the completed benchmark task to a bounded local
method screen, including an isolated runtime and public model download. No paid
service, remote audio transfer, system/GPU changes or production promotion is
part of this experiment. Live ownership remains in `CONTINUE.md`.

The question is whether a small audio-event classifier supplies temporal evidence
for the reported change in vocal behavior at 119–132s, beyond the known drop in
vocal RMS while activity continues. Speech/singing/laughter detection cannot by
itself establish musical leadership, importance, section identity or verse ending.

## Two-method comparison

| Method | Relevant capability and local cost | Decision |
| --- | --- | --- |
| Google YAMNet, official MediaPipe `float32/1` TFLite asset | Fixed audio-event classes include speech, singing, rapping and laughter. Official asset HEAD is 4,126,810 bytes; 15,600-sample mono input. Existing TFLite metadata is present locally, but NumPy 2 compatibility is uncertain. | First screen in a fresh NumPy-1.26/TFLite-2.14 CPU environment; no full TensorFlow installation. |
| PANNs `Cnn14_DecisionLevelMax` | Official implementation offers framewise sound-event outputs. The author's Zenodo v3 lists a 327.4-MB checkpoint. Existing PyTorch could support a separate route, but dependencies and CPU runtime are untested. | Reserve only if the first screen establishes a concrete model limitation worth comparing; do not install both by default. |

Primary sources consulted before setup:

- [Google's audio-classifier model and input contract](https://developers.google.com/edge/mediapipe/solutions/audio/audio_classifier).
- [YAMNet model/preprocessing description](https://github.com/tensorflow/models/blob/c14bf9ad91962cf189f9f58db2132c06247fcd53/research/audioset/yamnet/README.md).
- [Exact class vocabulary](https://github.com/tensorflow/models/blob/c14bf9ad91962cf189f9f58db2132c06247fcd53/research/audioset/yamnet/yamnet_class_map.csv).
- [PANNs author's implementation](https://github.com/qiuqiangkong/audioset_tagging_cnn) and [checkpoint record v3](https://zenodo.org/records/3987831).

This compares task fit and acquisition requirements, not accuracy on SongViz.
The TFLite variant is separately hash-bound; do not assume numerical parity with
the upstream Keras HDF5 weights merely because both are called YAMNet.

## Fixed first experiment, before song inference

Use the frozen four-case benchmark as evaluation-only reference. The classifier
receives waveforms only, never notes, names, target descriptions or prompts.
All 521 output scores are retained, so the report cannot select a favorable
class after seeing results. Save numeric inputs, model/class/source fingerprints,
environment versions, full scores and the exact runner used.

Eight predeclared conditions:

| Neutral ID | Input recipe | Purpose |
| --- | --- | --- |
| input-01 | Original mix, 16–26s | Existing within-passage case; no new event labels. |
| input-02 | Original mix, 74–85s | Existing drum-entry case; classifier does not certify continuity. |
| input-03 | Original mix, 119–132s | Primary vocal-behavior input. |
| input-04 | Original mix, 57–70s | Existing breakdown case; no exact endpoint target. |
| input-05 | Existing Demucs vocal stem, 119–132s | Source-separation sensitivity. |
| input-06 | Existing Mel-Band RoFormer vocals, 119–132s | Independent separation sensitivity; use the already-saved candidate output. |
| input-07 | Existing Mel-Band RoFormer instrumental, 119–132s | Passed listener control: laughter was not heard. |
| input-08 | Exactly 0.5 times input-03 | Gain sensitivity with the same musical content. |

Do not rerun separation. Verify the original FLAC, Demucs vocals, both RoFormer
outputs, completed listener-control record and their parent manifests before
reuse. Preserve originals and all earlier review packages. A new package gets
new output paths; existing paths are refused.

Decode source PCM, arithmetic-average channels, resample to 16 kHz with fixed
`scipy.signal.resample_poly` settings, and save float32 arrays without loudness
normalization or clipping. Reject nonfinite or out-of-range input instead of
silently changing it. Input-08 is derived from the final input-03 float32 array.
Use 15,600-sample windows (0.975s waveform support) at a fixed 7,680-sample hop
(0.48s). Only complete windows are evaluated; retain explicit uncovered tail
length. YAMNet's 0.96s mel-patch convention is not the full waveform support.
Model-window bounds refer to the resampled waveform; resampling and whole-clip
extraction are offline and do not establish physical onset or streaming latency.

## Predeclared evidence screen

Report individual scores for Speech (0), Singing (24), Rapping (31), Laughter
(13), and each laughter subtype (14–18), with the complete 521-class array
retained. Verify these IDs against the exact class map before execution. For
the following operational checks only, define `speech/singing family` as the maximum
of 0/24/31 and `laughter family` as the maximum of 13–18. These maxima are
descriptive summaries, not calibrated joint probabilities or musical-role labels.

For target inputs 03/05/06/07/08, use early [119,123.72] and late [125.45,132]
evaluation supports from the existing focus band. Include only model windows
fully contained in each support; discard straddlers from these summaries but
retain their scores. These supports are not new onset/offset annotations.

A fixed score of 0.5 is an **operational screening threshold**, not a calibrated
audibility probability. A target condition provides narrow behavior support
only if all three hold:

1. At least two consecutive early windows have speech/singing-family scores >= 0.5.
2. At least two consecutive late windows have laughter-family scores >= 0.5.
3. The late fraction of laughter-positive windows exceeds its early fraction,
   and the late fraction of speech/singing-positive windows is below its early fraction.

The instrumental condition fails the operational control criterion if it has
two consecutive late laughter-positive windows. Its listener check is not
ground-truth event absence, and passing this check does not validate classifier
specificity; high scores could also reflect separation artifacts. Compare primary and half-gain conditions under the
same rule; disagreement flags gain fragility. Report both separated-vocal
conditions without selecting only the better separator. A full-mix failure with
stem success is at most separator-dependent evidence, not primary-input success.
Neither separator independently validates event presence or absence.

Enumerate every model window's exact resampled sample bounds, source-relative
time bounds, early/late membership and per-support counts before gate interpretation.
Do not interpolate missing support or count uncovered tails as negative windows.

These are deterministic evidence gates, **not** automatic passes on the four
musical-development requirements. Threshold failures mean this configuration
did not provide the predeclared evidence, not that laughter/voice is absent or
that a different model cannot work. The other three cases remain descriptive
controls because no frame-level vocal-event negatives were annotated there.
No threshold, class family, interval or input recipe may change after scores
are seen. Any later alternative needs a new explicitly exploratory experiment.

## Execution and stopping

First inspect the downloaded model contract and run only synthetic silence,
seeded noise and a sine. Use two CPU threads, no accelerator and a 2-GiB RSS guard.
Each synthetic subprocess has a 120-second bound; the full eight-condition song
screen has one 120-second total bound. An interrupted/partial screen is incomplete
and cannot produce an accepted complete-screen conclusion. Runtime metadata
is not semantic evidence. Only after model/label/shape/provenance checks pass
may the frozen eight-condition song screen run locally.

Generate a saved-score report and simple static curves, comparing event scores
with RMS from the same model support. Preserve the older stem RMS/activity
baseline as separate evidence; do not pretend it was a semantic classifier.
Review actual score patterns and the fixed gates. Stop this screen after one
valid execution; no silent threshold changes, favorable reruns, full-song
extension or model promotion. A result may justify a specific follow-up, but
does not by itself justify another model search.

## Work and validation record

### Runtime accepted; implementation resumed — 2026-09-16

The isolated runtime completed before a temporary account usage limit interrupted
the runner implementation. The human confirmed the limit reset and requested
continuation. No song/stem inference occurred before that interruption, and the
fixed experiment above remains unchanged.

Runtime evidence: `outputs/reviews/vocal-event-runtime-01/`. Python 3.10.12,
NumPy 1.26.4, TFLite Runtime 2.14.0, SciPy 1.15.3 and SoundFile 0.13.1 are
installed in `.songviz/vocal-event-venv/`, without system site packages. Exact
transitive versions are in the saved freeze. The 240-MB environment has no
TensorFlow or PyTorch installation. The official model has float32 `[15600]`
input and float32 `[1,521]` output; internal tensors include int8, so its URL's
`float32` label is not a claim about every stored weight.

Silence, seeded noise and a 440-Hz sine produced finite scores in [0,1]. Two-thread
invokes took approximately 0.029/0.004/0.004 seconds; peak RSS was 90,423,296 bytes.
The synthetic wrapper imposed a 120-second timeout and a 2-GiB virtual-memory
limit, then checked peak RSS. This runtime evidence establishes interface
compatibility only. The song execution will use an external RSS monitor as well.

The lead independently verified downloaded-source/model, synthetic-result/log,
script and freeze hashes. All 521 embedded `yamnet_label_list.txt` strings match
the pinned CSV display names exactly in output order. Accepted hashes:

- Model: `4d8b4a53282dc83ef04e3e7dbc4fbc98082e34e44ed798e16c3a0cdd4c584faf`.
- Class map: `cdf24d193e196d9e95912a2667051ae203e92a2ba09449218ccb40ef787c6df2`.
- Runtime manifest: `dcef6475cf23a7861268ecfada5a847dfcebd8d7fb4dcf61d1beac75b6a7a646`.
- Runtime upstream record: `b2ee1ee69283e1140ada66eeb4c361575d0bb52df297d66228552b0a8523b9b9`.

Luna's pre-output design review required explicit per-window bounds/support counts
and a narrower instrumental-control interpretation. Both corrections are included
above: the control is listener-qualified, not ground-truth event absence; every
used and excluded window must remain inspectable. Implementation and its focused
tests were in progress at this runtime checkpoint; the completed result follows.

### Eight-condition screen complete — 2026-09-16

**Negative result: no target condition supplies the predeclared behavior evidence.**
One valid execution completed all eight conditions and 196 windows in 2.043451s,
with 157,691,904-byte peak monitored RSS (150.4 MiB), two CPU threads and no guard
stop. No threshold, class family, input recipe or evaluation interval changed
after outputs were seen. The frozen protocol and runner are in
`outputs/reviews/vocal-event-execution-01/`; inference arrays and per-window
records are in `outputs/reviews/vocal-event-probe-01/`.

| Target input | Maximum speech/singing-family score, whole clip | Maximum laughter-family score, whole clip | Early / late speech-positive windows | Behavior screen |
| --- | ---: | ---: | --- | --- |
| Original mix | 0.031250 | 0 | 0/8; 0/12 | Fail |
| Demucs vocals | 0.968750 | 0.148438 | 7/8; 1/12 | Fail |
| RoFormer vocals | 0.968750 | 0.199219 | 7/8; 1/12 | Fail |
| RoFormer instrumental | 0 | 0 | 0/8; 0/12 | No laughter-positive control run |
| Original mix at half gain | 0.019531 | 0.003906 | 0/8; 0/12 | Fail |

All target conditions have eight fully contained early windows and twelve late
windows; six other windows remain saved but excluded from those summaries.
Both vocal stems supply the required early speech run, followed by lower speech
scores, but neither supplies a laughter-positive window anywhere in the excerpt.
Their small laughter maxima occur earlier than the designated late region.
This reinforces the distinction between prior focus bands and actual event
annotations; it does not justify changing this experiment's intervals.

The listener-qualified instrumental control passes its operational check, but
cannot establish specificity when the positive condition fails. Matching failures
of original and half-gain inputs are not evidence of gain robustness. The other
three benchmark excerpts are descriptive only, without frame-level vocal-event
reference labels. There is no demonstrated improvement in musical leadership,
importance, section identity or automatic directing.

Saved-score interpretation and inspected PNG/SVG curves are in
`outputs/reviews/vocal-event-analysis-01/` ([report](../outputs/reviews/vocal-event-analysis-01/report.md),
[target curves](../outputs/reviews/vocal-event-analysis-01/target-curves.png)).
These plots compare scores with RMS over exactly the same waveform windows;
their builder performs no inference. All 521 scores per window remain available.

Validation actually performed:

- Terra implemented the runner and focused tests; lead review required strict
  waveform/score validity, full source/runtime pins and live environment checks
  before the valid run. Final focused tests: **9 passed, 1 existing ddtrace warning**.
  The runtime-tamper test uses synthetic local fixtures and does not depend on
  ignored model downloads. Compilation and `git diff --check` passed.
- Lead verified all 20 probe output fingerprints, protocol/runner preservation,
  and all seven analysis fingerprints. Independent decoding/resampling reproduced
  all eight stored model inputs sample-for-sample, including exact half gain;
  every window's recorded RMS was independently reproduced.
- Luna independently recomputed the fixed gates from saved arrays. The lead
  checked the actual counts: 20 probe fingerprints, and early/late membership
  only for inputs 03/05/06/07/08. Input 04 is the descriptive breakdown excerpt.
- Lead inspected both plot sets. No model rerun or broad regression suite was
  needed for this isolated experiment.

Accepted SHA-256 records:

| Record | SHA-256 |
| --- | --- |
| Pre-execution registration | `998c810826216cb845c9bd4c033bc80b4a829420745e10ad11f16297343b54e9` |
| Frozen protocol | `3d3bcc678d1e0ce044637f3f790a974adbde4f42b418776ebcae413cad27b54d` |
| Frozen runner | `bb171708685feb6635ae91ca0dc9914ff56130016239b4401125ae1502aac51f` |
| Guarded execution result | `3ade86c440c0176e807b74351043884f9b23cb8d10062b437bf547c4f25a0515` |
| Probe manifest | `85dc9734ec5a8a34625879a0b8ca5be50d3c9b31e2fbf5502ee56fa74e510434` |
| Analysis manifest | `ab1e0391179768569b73fec00967037a4feae9ada4d6be811f59e951712f978e` |

The executed command, retained for audit rather than assigned to run again, was:

```bash
python outputs/reviews/vocal-event-execution-01/run_guard.py \
  .songviz/vocal-event-venv/bin/python \
  outputs/reviews/vocal-event-execution-01/runner.py \
  --repo . --runtime outputs/reviews/vocal-event-runtime-01 \
  --protocol outputs/reviews/vocal-event-execution-01/protocol.md \
  --output outputs/reviews/vocal-event-probe-01
```

Focused checks need no model execution:

```bash
.songviz/venv/bin/python -m pytest -q tests/test_vocal_event_probe.py
.songviz/venv/bin/python -m py_compile experiments/probe_vocal_events.py tests/test_vocal_event_probe.py
git diff --check
```

### Follow-up decision: defer a second model

Terra reviewed PANNs read-only at author commit
`d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4`; the lead inspected the relevant source.
The [decision-level network](https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4/pytorch/models.py#L2781)
has five temporal halvings. Its [output utility](https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4/pytorch/pytorch_utils.py#L94)
repeats each segment score 32 times. At the official 10ms feature hop, distinct
segment positions are 0.32s apart, not independent 10ms predictions. From its
convolutions and pooling, we derive approximately 3.5s interior waveform support;
padding affects boundary support. This is an architectural calculation, not
measured event-localization accuracy.

That does **not** establish that PANNs would fail broad speech/laughter recognition.
It does rule out treating nominal framewise resolution as sufficient reason for
another run. Its 327,428,481-byte [author checkpoint](https://zenodo.org/records/3987831)
and missing `torchlibrosa` dependency would need a separate compatibility check.
No PANNs weight download, installation or inference was performed.

The completed YAMNet screen stops here. Next concrete step: define a small
vocal-behavior reference with explicit speech/singing, laughter, overlap and
uncertain time spans, starting with the existing 119–132s excerpt. This is new
temporal information, not a request to repeat the four musical judgments. Keep
these development labels separate from any later held-out example and from
musical leadership. Recognition and timing can then be evaluated separately
before selecting one further method. Do not relabel the frozen screen as a pass,
tune its threshold on this example, or launch a broad model search.

### Temporal reference preparation — 2026-09-16

The human requested continuation with Terra after the usage limit reset. Prepare
one local listening page for the original 119–132s excerpt. This collects missing
event timing, not a new musical-importance rating. Team 1 retains experimental
design/review; Terra implements the bounded builder and page.

Before labels: verify the original FLAC's pinned hash, extract exactly 119–132s
at its native rate/channel count with no normalization, save lossless PCM WAV,
and bind source/audio/builder/template hashes in a fresh package. Refuse existing
destinations. The page starts with no marks, preselected labels, model scores or
suggested boundaries. Show both elapsed and absolute song times clearly.

Users can play the full excerpt, choose and replay an interval, mark its start/end
from the player or numeric controls, and add/remove spans. Labels are speech/rap,
singing, laughter, other vocal sound, and uncertain; multiple labels and overlapping
intervals are allowed. Timing certainty is explicit. A note is optional. Unmarked
regions remain unknown, including after the user indicates finishing the review;
that flag is not exhaustive class absence. Label definitions remain visible and
do not equate audible voice with musical leadership.

Export a JSON record binding the exact source and excerpt, in absolute song
seconds, with blank spans by default and an explicit review-completed flag.
Keep local drafts scoped to the package/audio identity, validate them on restore,
and expose download failure or unavailable draft storage without losing editable
marks. This is an offline browser workflow; it does not write raw feedback in
the repository. Human exports require a later validated intake as new feedback,
never an overwrite of the four existing musical judgments.

Acceptance: focused builder checks for exact PCM extraction, source mismatch and
overwrite refusal; actual browser playback/seek/interval endpoint, mark validation,
overlap, export, draft reload and mobile layout. Lead verifies hashes and decoded
PCM independently. No model installation/inference or new event labels are part
of preparation. Stop at the concrete listening handoff if human labels are absent.

These labels will be collected after the YAMNet result was inspected. They are a
development reference, not a blind held-out assessment or a way to retroactively
change its frozen outcome. On receipt, preserve the raw export byte-for-byte as
new feedback and validate its package/source/audio identities, finite in-bounds
times, allowed labels and completion flag before creating any separate normalized
reference. Preserve overlaps, uncertain timing and unmarked unknown regions.
Do not infer event absence from omitted labels or from a completed-review flag.

### Temporal reference page accepted; human labels pending

Accepted page: [vocal-behavior-reference-03](http://127.0.0.1:8770/vocal-behavior-reference-03/),
stored in `outputs/reviews/vocal-behavior-reference-03/`. The page begins with blank
time inputs, no labels/spans and an unchecked completion flag. It plays a native
44,100-Hz stereo PCM-24 WAV containing exactly 573,300 frames from the original
119–132s excerpt. It shows song and excerpt time, replays selected intervals,
allows overlapping/multiple labels, preserves uncertainty and exports a record
bound to source, excerpt, builder and template identities. Local drafts use that
same identity. No feedback file is silently written into the repository.

Terra implemented the bounded builder/template and three self-contained tests.
Lead review rejected draft 01's default times, permissive labels and incomplete
draft/replay checks. Draft 02 passed the core annotation flow but failed a real
aborted audio request: the source-element error did not expose retry, and the
initial metadata message stayed stale. Version 03 repairs both and binds the UI
source hashes into exported identity. Versions 01/02 remain preserved.

Validation actually performed:

- **3 focused tests passed, 1 existing ddtrace warning**, covering native PCM
  extraction, source mismatch and overwrite refusal; compilation/diff checks pass.
- Lead independently verified the source hash, three output fingerprints, two
  snapshots, identity derivation and exact generated HTML. All 573,300 decoded
  stereo PCM frames equal a separately decoded original-source slice. Current
  builder/template hashes match the frozen snapshots and exported identity.
- Actual Chromium checks passed native playback/seek/marking, a clamped interval
  endpoint and resumed full playback, empty/invalid mark rejection, overlapping
  labels, literal user-note rendering, exact JSON export, draft reload/delete,
  mismatched-identity rejection, storage/download failures preserving editable
  marks, and real audio-request failure/retry retaining marks.
- Layout passed at 320/375/768/1100 pixels; desktop and mobile captures were
  inspected. Browser test marks were synthetic, in isolated browser contexts;
  they are not human feedback. No model or broad regression suite was rerun.

Independent evidence/scripts/screenshots:
`outputs/reviews/vocal-behavior-reference-validation-01/`. Accepted fingerprints:

| Record | SHA-256 |
| --- | --- |
| Page package manifest | `66471289b7bde9745970c0870e8ac0f79bf17f42732fd7e869094abbb2077504` |
| Excerpt WAV | `69f464bd7d911ce3a9f012e498a4f4271c9fb1031b5ad72fe94c7d055e3bb32e` |
| Builder | `591f0aa1d5da2c6c8745391d681ba91641b007b27e45df79bfc6ef60b3ea11b4` |
| Template | `0068175a2dd52f65a271fd7af3e036edc372739378750b56dabbe291bece5704` |
| Validation manifest | `55bd1a012555ee5a0b11c5a6a3b32ce889555a266bbf3e432c41cb06c36b2fb7` |

Commands performed for the final package:

```bash
.songviz/venv/bin/python -m pytest -q tests/test_vocal_behavior_reference.py
.songviz/venv/bin/python experiments/build_vocal_behavior_reference.py \
  --repo . --output outputs/reviews/vocal-behavior-reference-03
.songviz/venv/bin/python outputs/reviews/vocal-behavior-reference-validation-01/check_package.py \
  outputs/reviews/vocal-behavior-reference-03
node outputs/reviews/vocal-behavior-reference-validation-01/check_browser.cjs \
  http://127.0.0.1:8770/vocal-behavior-reference-03/
```

The builder refuses the now-existing destination. These are recorded acceptance
commands, not a request to rebuild the page or overwrite frozen validation files.
The standard review server was started locally at port 8770 for this handoff.
If it is no longer running, use `.songviz/venv/bin/python -m songviz.review_server`.

**Next step needs human input:** listen to the 13-second excerpt, mark the audible
speech/rap, singing, laughter or other/uncertain vocal spans, and return the JSON
export. Approximate times and overlaps are valid. A text reply with approximate
song-time spans is also usable and must be preserved as such rather than presented
as a page export. Preparation is complete; no event-timing observations have yet
been received, so there is no accepted temporal reference or next model run.
