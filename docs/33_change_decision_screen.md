# Fixed local distribution-contrast screen

Team 1, 2026-09-23. Follow-up to doc 32 using six already-seen listening
references across three tracks. The human requests continued independent work.
No new listening request, model search, source extraction or director integration.

## Acceptance criteria remain separate

The original four benchmark rubrics remain authoritative. Acoustic evidence or
abstention alone does not become a semantic pass.

| Reference | Required musical distinction | What must remain uncertain |
| --- | --- | --- |
| Feel Good Inc, 16–26s | No perceived meaningful development despite acoustic variation | Absence of one detector signal does not prove no musical change |
| Feel Good Inc, 74–85s | Added drums within continuing material | Bass entry and verse/chorus naming |
| Feel Good Inc, 119–132s | Vocal behavior changes while voice continues | Behavior/role cannot be inferred from RMS alone; laughter branch stays parked |
| Feel Good Inc, 57–70s | Broad breakdown/recovery rather than a single cut | Actual transition extent and precise endpoints |
| Agnes, original 24s excerpt | Listener reports a new part around elapsed 12s | Exact onset and tentative verse/pre-chorus identity |
| Arctic Monkeys, original 24s excerpt | More elements around elapsed 17s within mostly the same idea | Added instruments, exact onset and visual importance |

The new signal screen below tests an acoustic prerequisite. It does not satisfy
these musical criteria by renaming a score or returning unknown. All automatic
musical-continuity, section-identity, vocal-behavior and importance fields remain
explicitly unknown. Human judgments remain external references.

## One fixed hypothesis

Retain the doc-31 **level-only** rule unconditionally: existing per-stem median
relative change >=0.50 and at least 75% strict midpoint support on each side;
same stem/direction at half-windows 4 and 8 beats and at anchor k and k+1.
Reuse the existing sustained-activity numerical evidence; never invoke the
failed continuity gate or suppress a level proposal based on pattern evidence.

Add a local per-stem checkerboard distribution contrast. Reverse cached log1p
CQT with expm1, then normalize each usable beat's vector to unit L2 norm. A beat
is usable only above the existing whole-track RMS floor and with nonzero finite
spectral norm. Each side must have >=75% usable beats and at least two. Otherwise
the score is unknown. For h in {4,8}, compute:

`S = (within_left + within_right)/2 - cross`

Within-side terms are mean cosine similarities over distinct beat pairs, excluding
self-pairs. Cross is the mean of every usable left/right pair. Preserve negative
scores; negative is not semantic stability. Declare a persistent pattern signal
only when the same stem has S>=0.20 in all four comparisons (both scales and
both adjacent anchors). Keep each score, usability and the full four-window
support union. These comparisons are correlated, not independent confirmations.

The 0.20 threshold is a conservative fixed engineering choice, not calibrated by
its numerical reuse from the old pattern score. No tuning after this run. The
statistic is related to the existing story checkerboard, not a newly discovered
kind of evidence: this version uses local per-stem normalized CQT distributions,
excludes self-similarity, retains negative values and has no path enhancement,
track-maximum normalization or section partition.

This can reject an alternating/repeating pattern whose distributions are alike
on both sides, unlike a side-mean spectral contrast. It ignores temporal order
and may miss rhythmic reordering or gradual changes. Phrase phase, separator
leakage, articulation, gain-dependent usability and grid error remain confounds.
Shared silence must never count as continuity. Whole-track floors and future
windows keep this an offline experiment.

## Frozen evaluation and stopping rule

Only the verified saved CQT/RMS caches are input to prediction. Predict before
joining feedback; all six cases are already seen, so this is not a blinded or
held-out quality claim. Preserve the previous full reserve bounds and require
the full [k-8,k+9] support union inside each 24s clip. Development excerpts are
reported by anchor location with their full support retained.

Probe five anchors (-2,-1,0,+1,+2 beats) around each existing fixed benchmark
anchor, or the nearest beat to each new approximate human time. Probe coverage
is an engineering diagnostic, not a chosen onset tolerance or precision/recall.
Keep unavailable anchors unavailable. Report all proposal counts, support times,
per-stem scores and entire-excerpt candidate burden; adjacent anchors are not
independent events.

The narrow screen requires all of:

1. Exact level-only parity against every eligible frozen doc-31 development and
   Agnes anchor; no changed-layer suppression.
2. Retain a drum-increase signal in the drum probe and an increase signal in the
   Arctic probe.
3. Add persistent pattern evidence in the Agnes probe.
4. Add no persistent pattern proposal in the original non-change probe.

All six musical rubrics are assessed separately; semantic unknown is unresolved,
not pass. Broad extent and vocal behavior cannot be supplied by this rule alone.
If the narrow screen fails, reject this candidate without another threshold
trial. If it passes, it remains a development candidate requiring independent
validation, not production promotion. Report any additional non-change excerpt
burden explicitly even if its central probe is clear.

Required checks: analytical constant/repeating-pattern and orthogonal-step cases,
gain invariance when CQT magnitudes and RMS are scaled together above the absolute
floor, silence/zero-norm unknowns, exact support and input preservation, and old
level parity. New code/output directories only; all frozen sources remain intact.

## Completed result: reject the new rule

Package: `outputs/reviews/change-decision-01/`. The fixed narrow screen fails.
The new rule adds bass-pattern evidence near the Agnes reference but also in
the original non-change probe. It does not solve the intended distinction.
No threshold retuning, recurrence gate, semantic classifier or promotion follows.

| Case | Eligible excerpt anchors | Level anchors | Pattern anchors | Probe level / pattern |
| --- | ---: | ---: | ---: | --- |
| Original non-change | 23 | 0 | 3 | 0 / 2 |
| Drum addition | 25 | 2 | 0 | 2 / 0 |
| Vocal ending | 30 | 0 | 3 | 0 / 1 |
| Broad breakdown | 30 | 8 | 1 | 2 / 0 |
| Agnes new-part reference | 38 | 1 | 5 | 0 / 2 |
| Arctic layer addition | 17 | 2 | 0 | 2 / 0 |

The ±2-beat probes are fixed inspection neighborhoods, not exact event-matching
tolerances. The Arctic five-index probe has only three eligible anchors after
the full-support restriction; unavailable anchors are not counted as negatives.
Adjacent qualifying anchors are overlapping observations, not separate events.

All **520** old eligible level records (482 development, 38 Agnes) match exactly.
The known drum increase is retained at 78.670/79.103s; Arctic's `other` increase
is retained at elapsed 16.609/17.329s. The new Agnes pattern proposals in its
probe are song 104.513/104.954s (elapsed 11.219/11.660s). Non-change proposals
are 21.063/21.496s, also in bass. All six musical acceptance results remain
**unresolved**, including the non-change case: returning unknown is not success.

Independent reviewer comparison of the strongest four-window minimum per probe:

| Probe | Anchor | Minimum S across all four comparisons |
| --- | ---: | ---: |
| Non-change, bass | 21.496288s | 0.273847 |
| Agnes, bass | 104.513016s | 0.306901 |

Both pass the predeclared 0.20 threshold. Their small post hoc difference supplies
no validated decision boundary and is not a license to choose a separating
threshold. The failure survives the added requirement of sustained spectral
redistribution: a persistent acoustic change can still occur within material the
listener experiences as unchanged.

A further read-only check of the same saved RMS arrays also gives no simple
energy-based explanation: at these qualifying probe anchors, bass accounts for
roughly 34–55% of the summed stem-RMS power proxy across the original non-change
windows, versus roughly 8–11% in Agnes. That proxy is not perceptual prominence;
weighting by it would not straightforwardly favor the listener's new-part case.
No energy threshold or new candidate policy was tested.

### Validation and limits

**6 focused tests passed**, with the existing ddtrace warning. Lead review fixed
the worker's accidental `new_experiments/` location and a cancellation issue for
tiny nonzero spectra before execution; the added tiny-spectrum regression passes.
The executed normalization uses an expm1-equivalent factorization, preserving
small positive values while avoiding large intermediate exponentials.

The lead verified **77 input hashes, 14 output hashes, 4,352 per-stem score
records (942 unknown), and all 537 reported anchors**. Independent arithmetic
used sums of unit vectors and the off-diagonal norm identity, rather than the
implementation's explicit pair loop. Support bounds, usable counts, negative
scores, qualification intersections and semantic-null fields all matched. A
separate Terra reviewer verified output/prediction fingerprints, old-level
parity, the fixed-rule implementation and the negative-result interpretation.
`both` means level and pattern signals co-occur at an anchor; they need not
involve the same stem. It never means two independent semantic confirmations.

| Artifact | SHA-256 |
| --- | --- |
| Frozen design | `f83fa1700190bbef93cb4e5f4ec5abfb308236688f21785bc1875734690b35ba` |
| Numerical module | `b91d74bd22478561c900775246ed1d0f4c80ff2427148e06b098180e8b85a354` |
| Manifest | `14340ca7f967b841f2dc21b49c2fa53cce14e06044a26655b48822618e54a184` |
| Evaluation | `3ad56443a0a4202a2ab463e1b87900a299e4b1a6779b066790da4bec0282b776` |

```bash
.songviz/venv/bin/python -m pytest -q tests/test_change_decision.py
env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 timeout 120s \
  .songviz/venv/bin/python experiments/run_change_decision.py --repo . \
  --output outputs/reviews/change-decision-01
```

The completed output is frozen. No source extraction, learned inference or
detector threshold sweep occurred. These are already-seen development cases,
not a generalization estimate.

## Cue clarification requested and received

The remaining ambiguity is the musical cue behind the Agnes new-part judgment.
The current note says *where* it feels different and tentatively names the parts,
but does not say what changes in the vocal line, accompaniment, rhythm, or their
relationship. Persistent bass contrast alone also occurs in the non-change case.
Before choosing another representation or model, ask for a short description of
what makes the Agnes passage around elapsed 12s feel like a new part. No exact
timestamp, instrument taxonomy, new clip or detailed annotation is required.
Preserve the answer as a new raw intake; do not rewrite the earlier judgment.

### Completed signed-level diagnostic — 2026-09-23

The human answered: `i gets more "quiet" i think with less instruments, and give me the feeling that something is comming`.
The new raw intake preserves perceived quietness, tentative thinning and felt
anticipation separately. Timing near elapsed 12s is inherited from the earlier
exchange, not an exact onset supplied in this answer. Neither instrument count
nor an actual subsequent arrival is established.

Package: `outputs/reviews/change-decision-agnes-cue-01/`. Its protocol was fixed
before calculating new summaries: nearest reference beat +/-2, 2/4/8 beats on
each side, only comparisons fully inside the six existing excerpts. Original
source RMS uses native sample frames and both channels. Cached stem means,
medians and inherited activity-floor fractions are reported separately. No
new detector, perceptual threshold or semantic label was introduced.

At the nearest Agnes beat (elapsed 12.101s):

| Beats on each side | Original mix RMS change | Drum mean-RMS change | Other mean-RMS change |
| --- | ---: | ---: | ---: |
| 2 | -0.875 dB | -1.209 dB | -0.704 dB |
| 4 | -0.653 dB | -1.406 dB | +1.457 dB |
| 8 | -0.847 dB | -1.244 dB | +5.582 dB |

Across the 15 overlapping Agnes comparisons, mix RMS decreases in 14 and drum
mean/median RMS decreases in 14. Other-stem mean/median RMS increases in 13;
bass and vocals have mixed directions. All four groups remain above the
inherited activity floor in all compared intervals. These observations support
a modest signal-level reduction and receding drum energy near the reference;
they neither prove nor disprove fewer instruments within the separation groups.
The comparisons overlap and are not 15 independent confirmations. Unweighted
RMS is not perceptual loudness, and anticipation remains a human observation.

The non-change comparison is useful: other-stem mean/median RMS falls in all
15 comparisons there, while the original mix rises in 14. A declining component
alone therefore remains insufficient evidence of a meaningful transition.
The broad breakdown's mix falls in all 15 comparisons. The known drum addition
and Arctic other-layer addition retain their positive central level evidence.
Arctic has 14 supported comparisons and one unavailable 8-beat comparison;
unavailable support is not a negative result. This diagnostic does not establish
a transferable quietness, thinning, anticipation or section classifier.

Lead validation independently checked all **89 available source-window pairs**
using direct native-frame reads and vector norms, **356 stem comparisons**,
support availability and semantic nulls, plus **77 parent input hashes and all
six package-file hashes**. A read-only Terra audit independently confirmed the
Agnes stem directions and non-change counterexample. The shared-axis plot was
visually checked; use the table for the small Agnes differences. No production
code changed and no model or failed detector was rerun.

| Artifact | SHA-256 |
| --- | --- |
| Cue manifest | `3ff65e3607d54ed029bb2e5429023e98b76cd50e3d133f26f001bf035e1ebeba` |
| Signed results | `e46b6a8b41d0e30c2c69d09363eb6671d7cc6b565f8e94f3254baf966452918f` |
| Raw clarification | `fa8847bc71ae98d4f69be4617ebd92fcca1ce11bdb278374d4239ff46a41db5d` |

### Next reference: selection fixed before signal inspection

The next listening sample will be the central 24 seconds of the first filename
in Python's sorted `songs/*.flac` list after excluding the three tracks already
used in this comparison (Agnes, Arctic Monkeys and Gorillaz). This selects
**Castlecomer — Move**. Use native source frames, stereo PCM-24, and verify
exact decoded equality. No feature extraction or detector-result inspection
for this new passage before the human's description. Historical exposure is
unknown; no certified holdout claim is made.

Purpose: acquire a separate musical reference before designing a withdrawal or
anticipation rule around Agnes. A response that the music simply continues is
equally useful. Ask neutrally whether the idea continues or changes, roughly
when and how; do not suggest quietness, anticipation or an expected event time.
This is a new listening intake after the completed diagnostic, not a retroactive
extension of the rejected fixed-rule experiment or a promised positive example.

The package is now ready at `outputs/reviews/change-decision-listening-02/`:
song **77.633333–101.633333s**, exactly 24 seconds. Native PCM-24 round-trip
equality, source/WAV/snapshot hashes, HTTP payload equality and byte-range
response passed. Actual Chromium audio playback advanced with duration 24s
and no media error. Manifest SHA-256:
`23eca7e38138a3e2770b9e49156f4af281494717a9c8de4d1d301c8b9a848732`.

Playback verification initially found the old server had stopped. A transient
user service now runs the unchanged localhost-only review server independently
of the chat process: `songviz-review-8770.service`. The first service command
incorrectly supplied an unsupported `--host` argument; it was corrected to
`--port 8770` (the server already binds only 127.0.0.1). A browser check then
timed out because it assumed the browser's direct-WAV document contained an
`audio` tag. The corrected check inserted an explicit native audio element and
verified actual playback. Neither failure required a source or product change.

Inspect with `systemctl --user status songviz-review-8770`; stop with
`systemctl --user stop songviz-review-8770`. This transient service is not
installed for automatic startup after reboot. The human's neutral listening
description is the next required input; do not run a new feature/model search
while awaiting it.

## Castlecomer judgment received — 2026-09-23

The listener reports a noticeable change around elapsed **16–17s**, tentatively
verse to chorus. Raw text is preserved verbatim in
`outputs/reviews/change-decision-castle-intake-01/human-feedback.json`. This maps
to approximate song **93.633333–94.633333s**; the interval is not an exact onset,
event extent or agreed matching tolerance. The answer supplies no instrument,
loudness-direction or anticipation annotation. In particular, do not transfer
the Agnes quietness/anticipation description to this song.

A Terra reviewer independently verified source and excerpt hashes and exact
decoded PCM equality, and searched the cache inventory. Castlecomer has no
source-bound stems, beat analysis or feature cache. Existing per-stem methods
are therefore **unavailable**, not tested misses; this input gap must be filled
before an unchanged-method comparison is possible.

The intake's fixed protocol was saved before calculating a bounded original-mix
comparison. It uses 1/2/4 **seconds** on each side at elapsed 16/16.5/17s, and the
six prior reference anchors +/-0.5s with identical seconds windows. These are
not the prior beat-window methods. At the calculation midpoint, which is not
a revised human onset:

| Seconds on each side | Castlecomer original-mix RMS change |
| --- | ---: |
| 1 | +0.912 dB |
| 2 | +0.901 dB |
| 4 | +0.914 dB |

Eight of nine overlapping Castlecomer comparisons rise, with a full range
**-0.066 to +2.020 dB**. Agnes falls in seven of nine comparable fixed-seconds
windows. The non-change passage rises in all nine (+0.097 to +1.662 dB), while
the drum and Arctic additions also rise in all nine. These are descriptive
directions, not independent trials or perceptual thresholds. Level direction
alone cannot distinguish a new part, addition within a continuing idea, and
ordinary variation. No automatically detected chorus or section boundary is
claimed, and no new rule is selected from these values.

Lead independently checked **63 source-window pairs** with direct native-frame
reads/vector norms, all support bounds, summary signs and semantic-null fields,
**four source hashes and four package-file hashes**. There were no unavailable
windows. The comparison needed no model inference or product-code change.

| Artifact | SHA-256 |
| --- | --- |
| Intake manifest | `66157bbe84be7dd01df302d42477e6dc0b38266da71c8afbdbc769c24c8eb97a` |
| Source comparison | `5acb5f771759085921087d520a2a5addad46455f38f881f9854d3eec5b476316` |
| Raw feedback | `90456dedf5d0f4b3f6b4d616121340346fb7c460d34be6c5a328c7adb97cd677` |

The next technical prerequisite was to check feasibility of obtaining source-bound
Castlecomer stems and a beat grid using the existing local pipeline, isolated
outputs and cached models. Keep the human judgment frozen and the old policies
unchanged. This is input preparation, not a new model-selection experiment.

### Completed input preparation and unchanged-method comparison

The local prerequisite check passed. A separate fixed protocol and output
package, `outputs/reviews/change-decision-castle-methods-01/`, was created before
stem extraction or per-stem predictions. The installed Demucs 4.0.1 htdemucs
checkpoint was already cached and its full hash recorded (matching its published
filename prefix `8726e21a`). Execution used CPU, two numeric threads, `unshare
-Urn` to disable networking, and a 600-second timeout with a 10-second forced
termination grace. No model download, external audio upload, package changes,
GPU work or existing-cache overwrite occurred.

The source-bound stems and original-mix analysis now exist within that package.
All four stems are 44.1 kHz stereo PCM-16 and match the 179.266667s source duration.
The unchanged local `analyze_audio` produces 320 beat boundaries at 22.05 kHz /
hop512. Its source hash is bound explicitly; beat accuracy and separated-source
quality have not been independently musically validated. The existing CQT/RMS
extractor and default control, separate-channel and sustained-activity policies
were reused. Full predictions were saved before joining the reference guides.

| Policy | Fully supported anchors | Reported changes | Reported dip intervals |
| --- | ---: | ---: | ---: |
| Control | 28 | 0 | 0 |
| Separate channels | 28 | 0 | 0 |
| Sustained activity | 28 | 0 | 0 |

The nearest guide anchors are elapsed **16.059, 16.593 and 17.151s**, all eligible
under the fixed +/-8-beat support rule. No raw candidate is present around these
guides either, so filtering does not explain the absence there. The unfiltered
sustained policy has one anchor elsewhere inside the clip, at elapsed **0.502s**,
whose support starts **3.771s before the clip**; it is correctly excluded. Do not
describe every unfiltered full-clip result as empty.

At elapsed 16.593s, vocal pattern contrast is **0.501** at two beats per side.
The two-largest-stem aggregate pattern contrast is **0.331**, below the existing
whole-track adaptive threshold **0.532**. Other-stem median RMS increases from
0.05219 to 0.09969 at four beats per side (relative-change value **0.476**, below
the unchanged 0.50 activity minimum), and its eight-beat relative change is
0.403. These are measurable acoustic changes that the fixed policies do not
select. They do not establish a vocal melody change or instrument identity.
Do not lower thresholds to admit this familiar reference: earlier non-change
counterexamples remain unresolved. The listener's new-part judgment stands;
the methods have not demonstrated that distinction.

Execution completed in **120.55s**: separation 115.18s, analysis 2.91s, features
1.28s and policies 0.79s. Peak RSS was 1,111,888 KiB for the driver and 2,224,656
KiB for its child process. TorchCodec emitted warnings about encoding and bit
depth arguments; actual WAV headers and duration parity were checked afterward.
The CLI implementation worker's nested invocation hit an unsupported
`--full-auto` option and then completed the bounded driver directly; it did not
run inference. Lead review added checkpoint-prefix validation, pre-execution
code pins, stage timings, explicit cache location and stem/grid bounds checks
before the only actual preparation run. No production source was changed.

Lead independently verified **24 fingerprints**, **336 per-stem contrast
records** and **224 per-stem activity records** at all eligible anchors, support
masks, guide selection and candidate filtering. Contrast validation used dot
products of normalized window-sum vectors; activity validation recomputed
medians, strict midpoint support and qualification from the saved arrays.
A Terra reviewer independently checked fingerprints, default configs, source
bindings and the no-candidate result near the reference. The extractor and
detector source hashes match the frozen Arctic comparison; the reused masking
and guide helper ASTs match too (the runner's previous plot-only repair changed
its file hash). No model rerun or historical test-suite rerun was needed.

| Artifact | SHA-256 |
| --- | --- |
| Method manifest | `83dfdab9751ca86c00594a2739a54f4ad8e2d82cb4b32395732dc98deba10ba2` |
| Reported results | `ba4837428d38bc382463480961f7444cd432660f2651ae8ee5d003e545ff9f43` |
| Features | `e7c4265c009e6dff409e5ad30d7a06074b189eaf91eb41590cf704f433ea6595` |

Next input requested: what makes the Castlecomer 16–17s change feel like a
chorus, in the listener's own words. The existing note identifies the perceived
part change but not its audible cue. Preserve any answer as a new clarification
linked to the original reference. Do not choose another detector or representation
from the section name or the near-threshold value alone. No new clip is needed.

## Strategy correction after the user's challenge — 2026-09-23

The user questioned whether repeated failed experiments and follow-up listening
questions constitute a useful path toward the product. Lead review agrees that
the recent workstream has not demonstrated improved musical understanding.
The negative results are valid; continuing essentially the same diagnostic
loop is not justified by their careful numerical validation.

### What went wrong

- Local level/spectral changes repeatedly failed to distinguish meaningful
  development from variation, yet the queue stayed centered on that distinction
  using similar inputs and increasingly specific descriptions of individual clips.
- Source provenance, exact arithmetic and reproducible extraction are necessary
  engineering assets. They are not evidence of improved musical capability.
- The lead made the listener's ability to verbalize an audible cue a blocking
  dependency. The existing approximate new-part judgments already permit a
  diagnostic boundary comparison. A listener need not explain the mechanism
  before a task-trained model can be tested. Exact-onset accuracy and confident
  section naming remain unvalidated, but that does not prevent useful work.
- The repository still lists task-trained structure models as **not run yet**
  in `experiments/build_structure_tool_diagnostics.py`. Generic embeddings and
  local contrast rules did not establish a strong task-specific baseline first.

The independent Terra reviewer agreed on the measurement/capability gap but
suggested another cue-gated casebook. The lead rejects that recommendation:
doc 24 already records an explanation-card/reset loop without demonstrated
comprehension benefit. Another report requiring a more articulate listener
description would preserve the present dependency.

### Revised next deliverable

One **full-song music-structure baseline comparison**, starting with All-In-One
as the selected candidate for a bounded feasibility check. The authors' current
[repository](https://github.com/mir-aidj/all-in-one) and
[paper](https://arxiv.org/abs/2307.16425) describe supervised functional boundaries
and labels alongside beats/downbeats. This is a better-aligned hypothesis for
the missing coarse-boundary capability; it is **not evidence of success on our
songs**. SongFormer was also checked in its
[official repository](https://github.com/ASLP-lab/SongFormer); it is not a parallel
assignment or automatic fallback. This review did not download or run either.

Preserve the user's multiscale intent. A coarse section sequence is one evidence
layer, not the entire musical model, a verified identity/importance model or a
mandatory visual cut list. Existing arrangement additions, local changes and
transition intervals remain separately represented. The system still needs to
show what continues and returns; equal predicted section labels alone do not
prove musical recurrence. Artistic rendering remains deferred as requested.

The first implementation deliverable should make the existing analysis and the
new model's full-song output directly comparable with original-audio playback,
including all seven existing reference passages. Reuse the existing playback
components rather than building another diagnostic framework. Keep human notes
on the evaluation side. A side-by-side report is sufficient before any new UI.
The user should judge a concrete result after this comparison, not specify
which acoustic representation to engineer beforehand.

### Scope and stopping rule

1. Inspect the selected model's actual checkpoint/dependencies, license and local
   CPU feasibility; pin an official revision and preprocessing requirements.
   Bound setup to one isolated environment, at most one compatibility repair,
   and 60 minutes of active setup. Preserve the current working environments.
   No paid compute, system-driver changes or source-audio upload is authorized.
2. Before inference, register the published configuration, full-song input
   context, fixed evaluation/reporting rules and measured execution budget.
   Inspect one full-song feasibility run before scheduling the other tracks.
   Use the same configuration for the four existing songs; no song-specific
   tuning or selection among folds using our listening answers. Reuse separated
   audio only if the new model's preprocessing contract matches it.
3. Compare the Agnes and Castlecomer new-part references, the original non-change
   passage, the two additions within continuing material and the broad breakdown.
   Keep the vocal-ending behavior reference explicitly out of scope for a
   section model rather than treating it as automatically passed. Report all
   nearby proposals and uncertainty; new numerical matching rules must be fixed
   before results and cannot turn approximate prose into exact ground truth.
4. The deliverable is evidence of improved coarse musical-change recognition
   across the existing cases, including the negative reference, or a clear
   negative result. Runnable installation and plausible labels alone are not
   success. These already-seen cases cannot establish generalization; reserve
   additional validation only after the development comparison merits it.
5. A failed comparison stops this selected baseline without another threshold
   sweep, model-shopping cycle, or automatic request for another short listening
   explanation. Decide the next scope from the complete comparison and its
   resource cost. Keep all previous packages frozen.

The former Castlecomer cue question is **optional and no longer blocking**.
No further human annotation is required to begin this revised workstream.
This turn completed the strategy review and corrected the queue; it did not
claim to have implemented or validated the replacement approach.

### Opus 5.5 strategy review and fixed comparison criteria

At the user's request, the lead also consulted Claude Opus 5.5 via the isolated
`claude2` account (`Claude Code 2.1.280`; CLI model identifier
`claude-opus-5-5`; high effort; shell/file tools disabled). The independent
review agrees All-In-One is aligned as a task-specific hypothesis, subject to
keeping its flat output as one coarse signal. It pointed out that subthreshold
local evidence may indicate contextual rather than spike-like change. This is a
plausible interpretation to test, not a conclusion already established.

The review recommended fixing scoring before running and treating small-sample
results as go/no-go only. The lead adopts the frozen comparisons below, with two
qualifications: approximate prose does not prove a calibrated tolerance, and
the current reference songs are **not certified training-disjoint** from the
model. Do not describe them as held-out or out-of-sample. A circular-shift null
is deferred: with only a few imprecise, overlapping references it could lend
false precision, and the first requirement is a readable fixed comparison.

#### Frozen boundary task

- Run the official `harmonix-all` default checkpoint with published/default
  inference and peak-picking. Pin the official source revision, checkpoint,
  model dependencies and input settings before any song output is opened.
  Capture both final segments and frame activations. Final segments alone are
  scored; activations are diagnostic, with no new thresholding or label remap.
- Score a transition-boundary proposal within **elapsed 15–18s** in the
  Castlecomer clip (user says around 16 or 17) and **elapsed 11–13s** in Agnes
  (user says around 12). These fixed two-second neighborhoods are review windows,
  not calibrated tolerances or exact ground truth. Report strict 16–17s and
  12-second center distances descriptively, without using them to change the
  decision.
- Require **no model boundary anywhere in the Feel Good Inc `none` passage at
  16–26s** as the negative comparison. The drum-entry and Arctic additions are
  arrangement events in largely continuing material, so do not score them as
  positive section boundaries. Report model output there as scope checks.
- Inspect the broad breakdown and vocal-ending references, but keep their
  acceptance unresolved: neither prose provides exact section onsets suitable
  for this boundary score, and vocal behavior is outside a section model's
  remit. Report all default boundaries, labels and support across the four
  complete songs; do not selectively crop inference to favorable snippets.
- A possible **go to one coarse evidence-layer integration** requires both
  positive references inside their fixed windows and a clean `none` passage.
  A model label change to `chorus` at Castlecomer is useful supporting evidence,
  not a prerequisite: the listener's identity wording is tentative. For mixed
  results or any failure, stop this candidate with a no-go/ambiguous report.
  Do not switch folds, lower thresholds or select different boundaries after
  listening to the outputs.

The small sample cannot establish performance on unseen songs. Report the
boundary task separately from the model's functional label names and from local
arrangement events. If the test passes, integration remains exploratory: retain
continuous local, arrangement and transition evidence, and never derive musical
importance or visual emphasis directly from section labels.

#### Feasibility and failure reporting

The setup clock is **60 minutes of active setup**, ending at the first complete
full-song prediction. Try one isolated environment using the existing Python
and cached Demucs installation. All-In-One depends on source separation and
NATTEN/Madmom support; verify whether saved stems meet its exact preprocessing
contract. Avoid a large environment rebuild. Permit at most one small
compatibility repair, with a new isolated environment and preserved logs. No
paid compute, driver changes or external audio upload. If installation remains
blocked, report an **infrastructure no-go**, not a musical-model failure.

If one-song inference works, freeze the package/config and run the same model on
the four songs in isolated outputs. The standard research setting uses an
eight-fold ensemble, so measure CPU time/disk for one full song before running
four. If the ensemble cannot fit the already bounded local resource budget, stop
and report measured feasibility rather than silently changing to one fold. If
full comparison finishes, distinguish no-go on the fixed boundary task from
infra failure. Do not use annotations as model inputs; the model receives only
the source songs.

Two max-effort CLI attempts did not return advice: one reported transient OAuth
refresh contention, and a later process produced no output before the lead
terminated it after about 160 seconds. One high-effort Opus 5.5 review completed.
No project files, audio or model output were given to that review; it ran with
tools disabled and no session persistence. The model/version were confirmed
separately as Opus 5.5.

After the All-In-One dependency failure, the lead requested one more strategic
review with only a concise text summary, again with tools disabled and no session
persistence. An initial invocation accidentally used `--bare`, which disables
OAuth; it exited before sending the prompt. The corrected normal OAuth invocation
completed as Claude Opus 5.5 at high effort. Its full prompt and response are
preserved in `outputs/reviews/all-in-one-feasibility-01/opus-next-step.md`.
Opus recommends keeping the multiscale research goal, stopping repeated local
rule tuning, and prioritizing an end-to-end musical artifact. It advises getting
explicit scope authorization before another dependency attempt because the
previously stated setup cap has been reached. This is strategic advice, not a
finding about the model or a requirement imposed by the CLI.

## All-In-One feasibility — infrastructure no-go

The candidate is the official All-In-One v1.1.0 package. The isolated setup
installed it, Demucs 4.0.1, Hydra 1.3.2, and the documented Git Madmom at pinned
commit `27f032e8947204902c675e5e341a3faf5dc86dae`. All-In-One source tag v1.1.0
resolves to commit `c04f37609e2c7ba5d3b333d6d69a7e3c429dafc9`. On this host, the
resolver selected Torch 2.14.0+cu130 and TorchAudio 2.11.0+cu130; CUDA 13.0 was
available on the RTX 4050 Laptop GPU. Demucs and Madmom imported successfully.

Current NATTEN 0.21.7 initially attempted a CUDA kernel build and stopped because
CMake is unavailable. One compatibility repair hid the GPU for the package
build, producing the PyTorch-only NATTEN package. The All-In-One import then
failed because its model code imports `natten1dav`, `natten1dqkrpb`, `natten2dav`
and `natten2dqkrpb`, which are absent from NATTEN 0.21.7. Its attention wrappers
also call these names using an older signature. The project README says to
install NATTEN on Linux, while its package metadata only adds NATTEN under a
macOS platform marker; the current NATTEN PyPI backend does not supply this
legacy interface. See the [All-In-One README](https://github.com/mir-aidj/all-in-one/blob/main/README.md),
[package metadata](https://github.com/mir-aidj/all-in-one/blob/main/pyproject.toml),
and [NATTEN installation guide](https://github.com/SHI-Labs/NATTEN/blob/main/docs/install.md).

This is a dependency/runtime infrastructure no-go, not a model-quality result.
No NATTEN source patch, framework rollback, CMake/toolkit installation, or
alternate model was attempted. No checkpoint was downloaded, no track audio was
given to All-In-One, and no inference output exists. The frozen musical criteria
remain untouched. The isolated environment and package diagnostics were captured
in `outputs/reviews/all-in-one-feasibility-01/`. The environment still occupies
5.8 GiB and remains in place with the failing import state. No pre-existing
environment, source audio or cache was modified.

The setup cap is reached: one isolated environment and one compatibility repair.
The follow-up strategic review favors an end-to-end product artifact over more
narrow diagnostics and recommends explicit authorization before any additional
environment attempt. The next action therefore requires the human to decide
whether to grant **one timeboxed, isolated search for an upstream-compatible
All-In-One/NATTEN version pair**. If that requires editing model code or a custom
toolchain build, stop and amend the plan before running another named section
model or building a pure-PyTorch shim. A successful section model would remain
only one coarse layer; it would not establish arrangement understanding,
returns, or musical importance.

Detailed setup evidence and the full Opus response:
[`setup-result.md`](../outputs/reviews/all-in-one-feasibility-01/setup-result.md)
and [`opus-next-step.md`](../outputs/reviews/all-in-one-feasibility-01/opus-next-step.md).

## Second All-In-One attempt — infrastructure no-go

The human authorized one additional, timeboxed attempt to use an upstream
All-In-One/NATTEN dependency pair. This supersedes only the earlier request for
authorization; the frozen musical criteria and all stop conditions remain in
force. The first environment, first-attempt logs and no-go result remain
preserved in `outputs/reviews/all-in-one-feasibility-01/`.

### Candidate pinned from upstream evidence

- All-In-One remains pinned at v1.1.0 / commit
  `c04f37609e2c7ba5d3b333d6d69a7e3c429dafc9`. Its upstream `dinat.py` imports
  `natten1dav`, `natten1dqkrpb`, `natten2dav` and `natten2dqkrpb` from
  `natten.functional` and passes kernel size and dilation to the attention
  wrappers.
- NATTEN v0.15.1's official source defines the legacy QK wrappers with
  `(query, key, rpb, kernel_size, dilation)` and the AV aliases with
  `(attn, value, kernel_size, dilation)`, matching the All-In-One calls. The
  official release includes the exact CPython 3.10 Linux x86_64 wheel
  `natten-0.15.1+torch200cu118-cp310-cp310-linux_x86_64.whl`.
- A newer option, NATTEN v0.17.4 with Torch 2.5.0/cu118, also exposes those
  names, but its own source marks them deprecated wrappers over its refactored
  backend. NATTEN v0.17.5 removes them. After comparing the two, Claude Opus
  5.5 recommends the Torch 2.0.0/NATTEN 0.15.1 pair because it keeps the
  legacy functions as the primary API and avoids changing the framework at the
  exact compatibility boundary. This is a risk choice, not a confirmed All-In-One
  development environment.
- PyTorch's official previous-version matrix pairs Torch 2.0.0, TorchVision
  0.15.1 and TorchAudio 2.0.1 on CUDA 11.8. TorchVision will be omitted unless
  inspection proves All-In-One needs it.
- Host: Python 3.10.12, NVIDIA RTX 4050 Laptop GPU (compute capability 8.9),
  driver 580.178.04. Source documentation does not establish the NATTEN
  wheel's compiled GPU architectures or runtime behavior on this GPU; those
  are smoke-test questions, not assumed compatibility.

Primary references: [All-In-One v1.1.0 `dinat.py`](https://github.com/mir-aidj/all-in-one/blob/v1.1.0/src/allin1/models/dinat.py),
[NATTEN v0.15.1 functional API](https://github.com/SHI-Labs/NATTEN/blob/v0.15.1/src/natten/functional.py),
[NATTEN v0.15.1 release and assets](https://github.com/SHI-Labs/NATTEN/releases/tag/v0.15.1),
[NATTEN changelog](https://github.com/SHI-Labs/NATTEN/blob/main/CHANGELOG.md),
[official legacy-wheel index](https://whl.natten.org/old/),
and [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/).

Claude Opus 5.5 (`claude2`, model `claude-opus-5-5`, high effort, tools off,
no session persistence) reviewed only text summaries, not project files or
audio. It compared the Torch 2.0.0/NATTEN 0.15.1 and Torch 2.5.0/NATTEN 0.17.4
pairs and selected 0.15.1 because its old wrappers remain primary API rather
than deprecated shims. Exact GPU execution and full-model behavior remain
unverified. Its prompt and findings are recorded in
`outputs/reviews/all-in-one-feasibility-02/opus-compatibility-audit.md`.

### Scope and gate before audio

Preserve `.songviz/allin1-venv/` and create a fresh
`.songviz/allin1-venv-02/` with Python 3.10.12. Candidate pins are Torch
2.0.0+cu118, TorchAudio 2.0.1+cu118, NATTEN 0.15.1+torch200cu118,
`numpy<2`, All-In-One 1.1.0, Demucs 4.0.1, Hydra 1.3.2, and the already used
Madmom upstream commit `27f032e8947204902c675e5e341a3faf5dc86dae`. Use
constraints so later package installation cannot replace Torch or NATTEN. No
model source patch, custom attention/kernel build, CUDA toolkit or driver
change, remote inference, paid compute or source-audio upload is allowed.

The existing pinned Madmom dependency is an upstream Cython source package. Its
build metadata requests NumPy>2, while the Torch 2.0 runtime must use NumPy 1.x.
Install `numpy==1.26.4` and `Cython==0.29.37`, then build only that pinned
Madmom commit with `--no-build-isolation` so its extension uses the runtime
NumPy ABI. This is the project's ordinary upstream package build; it does not
patch model code or build a custom attention kernel. If it fails once, stop.

The legacy `shi-labs.com/natten/wheels` host currently presents a TLS
certificate expired on 2025-12-03; do not disable certificate verification to
use it. The official v0.15.1 GitHub release page links the wheel as a GitHub
release asset. Attempt 02 will fetch only through that official endpoint after
confirming its TLS chain. The release API publishes its 89,829,851-byte asset
digest as
`b198acfd271e72f18475614ca5c0c3677e5b6efa20eab98e5e2cd564cdbfd221`; verify it
before installation.

Before any checkpoint download or audio inference, require all of these:

1. The new environment resolves and imports the pinned packages; all four
   legacy NATTEN names exist at the expected module path and their inspected
   signatures accept the arguments used in All-In-One v1.1.0.
2. CUDA is available on the RTX 4050 and synthetic 1D and 2D NATTEN operations
   execute on GPU. Where CPU kernels are available, compare deterministic CPU
   and GPU outputs with a fixed tolerance. No song audio or checkpoint is used.
3. `allin1`, Demucs, Madmom and TorchAudio import, `pip check` passes, the final
   freeze confirms the exact versions, and at least 1.5 GiB remains free for
   the checkpoint and inference artifacts. Use `/dev/shm` for temporary wheel
   staging if it has enough free capacity; do not evict existing caches or
   delete the first environment to make room.

The setup timebox is 60 active minutes, starting at 2026-09-23 22:07:06 UTC and
ending at 23:07:06 UTC, or earlier at an infrastructure no-go / first complete
full-song prediction. If any gate fails or the time expires, stop this candidate
and record the exact failure without another version sweep, source patch, shim
or custom attention/kernel build. If gates pass, freeze all versions and the published/default
`harmonix-all` configuration, then obtain and hash the checkpoint before
running the first full song. Start with the already selected Castlecomer
positive case; inspect feasibility and measured resource use. If it completes,
run the same full-song configuration over the four existing songs, including
the two fixed positives and the Feel Good Inc `none` negative, in a new output
directory. Apply only the preregistered criteria above. These setup gates are
predeclared conditions; the separate synthetic Conv2d check stopped the attempt
before NATTEN installation, so no checkpoint or audio run was made.

Current storage snapshot at planning time: 8.15 GiB free on the workspace
filesystem; first environment occupies 5.8 GiB. This is close enough to require
a fresh-environment preflight and use of temporary storage, but not enough to
justify deleting or repurposing existing state. Detailed command logs, final
freeze, test output and any checkpoint identity belong in
`outputs/reviews/all-in-one-feasibility-02/`.

### Attempt 02 outcome — stopped at Torch/cuDNN Conv2d

The authorized 60-active-minute attempt ended before its deadline at the first
failed GPU runtime gate. A fresh Python 3.10.12 environment contains
`torch==2.0.0+cu118`, `torchaudio==2.0.1+cu118`, `triton==2.0.0` and ordinary
Torch dependencies only. `torch.cuda.is_available()` returned true on the RTX
4050 Laptop GPU (capability 8.9), and a synthetic matrix multiplication passed.
A synthetic `Conv2d` call aborted at cuDNN initialization with:

```text
Could not load library libcudnn_cnn_infer.so.8.
Error: libnvrtc.so: cannot open shared object file: No such file or directory
```

The Torch wheel's `METADATA` declares no `nvidia-cuda-nvrtc` runtime dependency.
Its `torch/lib` directory contains the hashed NVRTC library and versioned
builtins, but no plain `libnvrtc.so`. A read-only audit fetched NVIDIA's
`nvidia-cuda-nvrtc-cu11==11.8.89` manylinux x86_64 wheel from its official
PyPI URL and verified SHA-256
`a8d02f3cba345be56b1ffc3e74d8f61f02bb758dd31b0f20e12277a5a244f756` against
PyPI's JSON metadata. The archive contains `libnvrtc.so.11.2` and
`libnvrtc-builtins.so.11.8`, not the required unversioned filename. It was not
installed. Making it visible would additionally require a symlink, loader-path
or system change outside the authorized stop conditions.

Claude Opus 5.5 reviewed the complete runtime facts and recommended stopping at
this first failure; it independently predicted the NVIDIA package would not
provide the required plain filename. Direct archive inspection confirmed that
prediction. The official Torch/NATTEN candidate therefore remains unvalidated;
this is an infrastructure no-go, not a musical-model result. No NATTEN or
All-In-One dependency, checkpoint or song audio entered attempt 02. Preserve
both isolated environments and all logs. Do not begin a third version/runtime
search under the current authorization. Stage-one install/smoke logs and the
wheel audit are in `outputs/reviews/all-in-one-feasibility-02/`.

### Next project step after the no-go

The lead asked Claude Opus 5.5 to challenge the next direction, then updated it
with the exact existing full-song evidence. Its revised recommendation is to
audit the existing `structure-review-03` listener page against a written
contract, not create another separate artifact. The page already contains the
full-song audio, previous/candidate section timelines, recurrence questions and
acoustic diagnostics. The full song is Gorillaz, *Feel Good Inc.*; its exact
source hash matches the annotations. A source-manifest audit verified **64 of
64** source, snapshot and output files across `structure-evaluation-03` and
`structure-review-03`, with zero mismatches.

The section-editor input contains **19 contiguous spans covering all 221.173333
seconds**, with no time gaps or overlaps. Every certainty is `unspecified`, and
the notes are blank. The four listening responses, including the `none`
judgment at 16–26 seconds, are independently hash-bound to the same song. The
coverage means there is no reason to repeat a full-song annotation pass; it
does not establish that every label is certain. The current page has only its
heuristic baseline and candidate (7 and 6 spans). It does not display the 19
human spans, explicit motif groups or the four raw listening notes. Therefore
it fails the current listener-facing contract and needs a focused extension of
that same page before human sign-off.

The page audit contract is:

1. Display all 19 human spans with exact label/times and the saved
   `unspecified` certainty; keep explicit motif IDs visible and empty motifs
   unknown.
2. Keep the acoustic baseline and candidate visibly separate from human marks.
   The candidate has five internal boundaries; **13 of 18** human internal
   boundaries are more than 1.0 second from the nearest candidate boundary.
   Show each timestamp and signed offset. The 1.0-second cutoff selects review
   points; it is not a validated musical timing tolerance. Candidate role names
   do not share the user's section-label taxonomy, so no semantic mismatch is
   inferred.
3. Show the exact four raw listening notes over their source-bound windows:
   16–26s, 57–70s, 74–85s and 119–132s. The `none` note remains a perceptual
   judgment; any measured acoustic movement stays separately described as
   acoustic evidence, not promoted to a musical event.
4. Make the combined page source-traceable and usable for targeted listening at
   the flagged boundaries and feedback windows. No full-song redrawing, new
   model, feature extraction or confidence score is part of this check.

### Completed structure-review overlay — targeted-review objective fails

The existing page was extended in place as a new frozen version:
[`structure-review-04`](http://127.0.0.1:8770/structure-review-04/), package
`outputs/reviews/structure-review-04/`. It contains the source-matched 19-span
human timeline, the four exact raw feedback notes and ranges, and all 13
human/candidate timing records selected by the predeclared 1-second selector.
The user marks remain explicitly uncertain, heuristic timelines remain
separate, and no feature extraction or model inference ran. The builder copied
the verified parent WAV and three linked plot assets into this new version;
the earlier package remains frozen.

Validation: **11 focused tests passed** (with the existing ddtrace warning),
Python compilation and `git diff --check` passed. Chromium rendered all 19 human
details, four raw-feedback cards and 13 timing records. The audio became
seekable; a real browser interaction started the first bounded feedback clip
at 74.82s inside its 74–85s source window. The local server returned HTTP 206
for a byte-range WAV request. All five manifest-listed output hashes, including
the original WAV and the three plots, matched. These checks establish a working
artifact, not that a listener found it useful.

The fixed listening controls reveal why no sign-off was requested. The 13
timing contexts (human boundary ±8s, clipped at the song ends) plus four raw
feedback windows cover **172.97 of 221.17s (78.2%)** in their union. The merged
playback ranges are 0–13.35, 16–41.45, 53.37–70, 74–103.40, 116.30–132.30,
136.18–166.59, and 179.44–221.17s. This is effectively a near-full-song
listening assignment. Under the predeclared contract, **targeted review fails**;
do not ask the human to sign off on all 17 windows or repeat annotation. The
four raw judgments remain useful source-bound references, but listener
validation of this page has not happened.

The table below is descriptive. Signed distance is nearest candidate time
minus human-marked time. The final column assigns each of the five candidate
internal boundaries to its nearest human boundary; the other human boundaries
are unpaired under this one-to-one nearest-time bookkeeping. This is not a
semantic match and does not establish that either timeline is correct.

| Human boundary (s) | Nearest candidate (s) | Candidate − human (s) | Nearest-time pairing |
|---:|---:|---:|---|
| 5.350 | 6.989 | +1.639 | Unpaired |
| 6.157 | 6.989 | +0.832 | Paired |
| 30.467 | 6.989 | −23.478 | Unpaired |
| 33.445 | 6.989 | −26.455 | Unpaired |
| 61.368 | 64.366 | +2.997 | Unpaired |
| 64.855 | 64.366 | −0.490 | Paired |
| 78.890 | 79.087 | +0.197 | Paired |
| 92.816 | 79.087 | −13.729 | Unpaired |
| 95.398 | 79.087 | −16.311 | Unpaired |
| 124.295 | 137.996 | +13.701 | Unpaired |
| 137.999 | 137.996 | −0.003 | Paired |
| 144.178 | 137.996 | −6.181 | Unpaired |
| 158.591 | 165.721 | +7.130 | Unpaired |
| 165.534 | 165.721 | +0.187 | Paired |
| 187.441 | 165.721 | −21.720 | Unpaired |
| 189.926 | 165.721 | −24.205 | Unpaired |
| 203.759 | 165.721 | −38.038 | Unpaired |
| 217.973 | 165.721 | −52.252 | Unpaired |

There are 5 candidate internal boundaries and 18 human-marked internal
boundaries. The count above each descriptive distance selector is **14/18 at
0.5s, 13/18 at 1s, 12/18 at 2s and 11/18 at 4s**. These thresholds are not
acceptance tolerances. The five nearest-time pairs are the human boundaries at
6.157, 64.855, 78.890, 137.999 and 165.534s; the remaining 13 do not receive a
candidate in this one-to-one positional assignment. This points to a possible
granularity mismatch (macro segmentation versus finer human subdivisions), not
to a demonstrated model error or a reason to add another model.

Claude Opus 5.5 independently agreed that the targeted-review condition fails
under the frozen rule: do not narrow windows or alter the 1-second selector
after seeing coverage; preserve this artifact and measurements; distinguish
mechanical completion from listener validation. Its suggested next diagnostic
is to describe all 18 distances and whether candidate boundaries receive unique
nearest-time pairings. The table above completes that read-only calculation.

**Decision recorded below:** the human chose both broad sections and fine
events as separate visual layers. The page remains an inspection surface, while
its 78.2% diagnostic coverage is explicitly not targeted sign-off. The
5-vs-18 count remains a granularity mismatch to keep visible, not a basis for
another model or selector adjustment.

### Product direction: broad sections and fine events in separate layers

The human chose to retain **both** levels, shown as separate visual layers.
This resolves the next representation question; it does not validate an event
detector or authorize another model run.

- The section layer is a contiguous, source-timed partition with optional
  within-layer recurrence identities. Human marks and heuristic candidates stay
  in distinct rows; their labels and certainty do not transfer between sources.
- The fine-event reference layer is independent. A local arrangement change
  may occur inside a section, coincide with a section transition, or span a
  transition. It must not be nested or snapped to section boundaries.
- The four current listening ranges are **review windows**, not event extents:
  their endpoints bound the audio the human was asked to hear. The event
  reference therefore keeps a separate raw label and note, and the 16–26s
  `none` answer remains an explicit reviewed/no-change control. The rest of the
  song is unreviewed by these prompts, not a negative event label.
- The listening export contains no certainty field. Display “not recorded” for
  its certainty; do not copy “unspecified” from the distinct section editor.
- These four examples are too sparse for event taxonomy, onset tolerance,
  detector scoring or hit-rate claims. Keep acoustic candidate support separate
  from listener review windows until a suitable event reference exists.

### Layered reference implementation — structure-review-06

The human's choice is implemented in a new frozen page:
[structure-review-06](http://127.0.0.1:8770/structure-review-06/), built from
`structure-review-05`. It retains the contiguous section timelines and adds a
distinct listener-window lane with the four exact 16–26, 57–70, 74–85 and
119–132s prompt ranges. Three are marked “change heard”; the 16–26s `none`
response is a separately styled reviewed control. Remaining song time is
visibly marked unreviewed. The lane says in place that these are listening
windows, not event boundaries or durations. Its ends fade; clicking a window
plays only its bounded source range.

The raw perceived-change labels and notes remain in separate cards. Certainty
is shown as not recorded because the listening export has no certainty field.
The lane contains no derived points/extents and no candidate/baseline/section
comparison fields. The independent human/baseline/candidate section rows and
the human/candidate section timing diagnostic remain in their own section layer.
No label reconciliation, event scoring, overlap statistic or inference was
added. `structure-review-05` is preserved but superseded because its no-change
card carried a candidate-boundary summary and its window edges were too crisp;
the accepted -06 package removes the comparison and softens the bands. Package
`outputs/reviews/structure-review-04/` and all earlier versions remain intact.

Validation: **13 focused tests passed** across human-reference, page and builder
suites (one existing ddtrace warning); compilation and `git diff --check` passed.
Chromium rendered 19 human details, 4 raw-feedback cards, 13 section diagnostics
and 4 separate listener windows (3 change-heard, 1 none-control). The audio
loaded and the `none` button started playback at 16.58s inside its 16–26s range.
All five manifest-listed output hashes matched; the copied audio and all three
plots are byte-identical to -05. The page returned HTTP 200. This is layout and
source-provenance validation, not listener validation or evidence of an event
detector.

The event reference remains intentionally sparse and imprecise. Do not use the
four windows to set an event taxonomy, onset tolerance, detector score or
hit-rate. The 74–85s prompt has now yielded an approximate drum-entry note, but
that response is potentially exposed as described below. A separate blind
source-only reference is the next useful check. Only after a cue is captured
without supplied predictions should we compare it with existing measurements
and decide whether the event layer needs a new representation; this product
choice alone does not justify another large model.

### Approximate drum-entry onset received — potentially exposed

Asked where the kick/snare first enter within the existing 74–85s source window,
the human replied verbatim: **“79s they enter”**. The contextual interpretation
is an approximate source-song time of 79s; no exact onset, uncertainty interval,
event duration or certainty was supplied. The earlier note that bass might also
enter remains unchanged and uncertain.

The question linked `structure-review-06`, which shows a candidate section
boundary at 79.087s. Whether the human opened the page is unconfirmed. Because
the candidate was available before the response, treat this note as potentially
exposed and do not count its proximity as model/reference agreement. Preserve
the raw reply and provenance in
`outputs/reviews/structure-review-event-intake-01/`. It is an event-reference
format example, not an evaluation label. No inference or score was run.

The next deterministic source-only excerpt has now been prepared below,
without candidate timelines or analysis. Historical human familiarity remains
unknown; this is not a certified holdout.

### Blind source-only listening excerpt prepared — awaiting qualitative note

The metadata-only selection rule chose the central 24 seconds of the first
eligible alphabetical source with no existing `outputs/<stem>/analysis/story.json`
after excluding previously used tracks: `Ella Vos - Eyes v3.flac`. The listener
page deliberately omits title, source offset, candidate timelines, features and
annotations. It asks only whether anything changes, for a rough time in the
clip, what seems to continue, and allows “no clear change” or “unsure.”

The frozen package is
[event-reference-listening-01](http://127.0.0.1:8770/event-reference-listening-01/),
at `outputs/reviews/event-reference-listening-01/`. It contains a native-frame
PCM_24 crop from frames `[5,445,639, 6,504,039)` (44.1kHz stereo, 24.000s), a
blind HTML page, protocol and manifest. Source SHA-256 is
`7b0fd8865086b7716549f101165f65bba25383c0e9241c1d104e4cf4f44ddba5`; excerpt
SHA-256 is `ed543330402ac3ec872487db68e3c2a7439eb4b4cebeacea02887fde6aa32de4`.
Team 1 independently confirmed decoded signed-PCM equality against the exact
source frames, all output hashes/byte counts, HTTP 200/206 delivery, and the
package manifest. Terra's bounded implementation check also confirmed headless
Chrome loaded the blind page and played the muted audio without browser errors
or failed requests. No model, feature, prediction or event inference was read
or created. Historical listener familiarity is unknown, so this is not a
certified holdout or a generalization result.

The page is preserved as an optional future listening reference, not a blocker.
No further short listening response is needed before the next technical test.

### Predeclared existing-section-method transfer screen — 2026-09-24

Opus review identified a layer/method gap: recent local arrangement rules were
tested repeatedly, while the current section method had only the Feel Good Inc
candidate output for direct human-boundary comparison. The next bounded test is
therefore one transfer screen of the existing `songviz.story.compute_story`
section output. It is not a new model, feature, detector threshold or model
runtime attempt.

**Inputs and isolation:** run each full source track at 22,050 Hz mono using
the unchanged current code, `hop_length=512`, `frame_length=2048`, default
automatic beat grid, and `stems=None` / `other_y=None`. This prevents the
section pass from importing the early stem-entry injections used by the
combined analysis path; the user's selected section and event layers stay
separate. The human references are joined only after all four full-song outputs
have been saved. Do not read cached candidate times before these runs. Inputs:
Agnes — MILK, Castlecomer — Move, Arctic Monkeys — Do I Wanna Know_, and
Gorillaz — Feel Good Inc. Use the existing full-track source files only; do not
use separated stems, existing story predictions or review-page data as model
inputs.

**Scoring, fixed before execution:** score final internal section starts from
the complete `sections` sequence, not discarded-boundary diagnostics or
functional labels.

| Track and reference | Full-song scoring window | Decision use |
| --- | ---: | --- |
| Agnes, approximate new-part judgment around elapsed 12s in the 24s clip starting at 93.294263s | 104.294263–106.294263s | Positive; at least one final boundary is required |
| Castlecomer, approximate part change around elapsed 16–17s in the clip starting at 77.633333s | 92.633333–95.633333s | Positive; at least one final boundary is required |
| Feel Good Inc, reviewed `none` passage | 16–26s | Negative; no final boundary is allowed |
| Arctic Monkeys, more elements within the same reported idea around elapsed 17s in the clip starting at 124.197052s | 140.197052–142.197052s | Event-layer scope check only; no section pass/fail score |

The task passes this **small transfer screen** only if both positive windows
contain a final boundary and the Feel Good Inc negative window is empty. Any
miss or negative-window boundary is a screen failure; do not tune, switch beat
grids or rerun. The Arctic event window is reported separately and is not a
positive section reference. The possibly exposed 79s drum-entry reply is not
scored. Report every final boundary, boundary count per song/minute, beat-grid
metadata, fallback/errors, and the Feel Good Inc distances against its 18
human-marked boundaries at descriptive ±0.5s, ±1s and ±3s cutoffs. These are
diagnostics, not calibrated tolerances, significance tests or proof of
generalization. An all-pass result only earns a later review of the frozen
four-song section output; it does not promote a model or label the songs.

**Stop rule:** one canonical execution per song in the project-prescribed
`.songviz/venv` with the fixed code/config. No threshold or timing-window
changes, model shopping, new model runtime setup, or extra listener annotation
follows a valid screen failure. Report infrastructure or unsupported-input
failures separately from musical misses. Another large model is not justified
before this baseline screen; the previous All-In-One infrastructure no-go
remains closed.

The exact runner, inputs and outputs are preserved in
`outputs/reviews/ssm-section-transfer-01/`; bounded implementation is
`experiments/run_ssm_section_transfer.py`, with focused scoring checks in
`tests/test_ssm_section_transfer.py`.

### Execution-environment correction — 2026-09-24

Integration review found that `ssm-section-transfer-01` was executed with the
workspace `.venv` (Python 3.14.3, librosa 0.11.0, NumPy 2.4.3), while the
project's documented validation environment is `.songviz/venv` (Python 3.10.12,
librosa 0.11.0, NumPy 2.2.6). Since beat tracking and feature computation can
depend on the numerical stack, preserve the first output but do not interpret
its failed gate as the project-environment result. The exact same code, full
sources, fixed windows and scoring were then executed once in
`.songviz/venv`, at `outputs/reviews/ssm-section-transfer-02/`. Because run 01's
output was already seen, run 02 is a provenance correction/cross-environment
replication, not blind or independent validation. Run 02 is canonical; no
further rerun or tuning follows. The saved predictions and report are identical
across environments.

### Opus review and strategy decision — 2026-09-24

At the user's request, Claude Opus 5.5 (`claude-opus-5-5`, high effort) reviewed
the current situation. Its central assessment matches the evidence: SongViz has
made real progress in its representation, reference handling and reproducible
experiments, but has not demonstrated a model-quality improvement. Repeated
local contrast and continuity rules have mostly produced useful negative
results. Their reproducibility is a project asset, not evidence that the system
understands a song. Opus recommended testing the existing full-song section
method before collecting more listening notes or starting another model; that
test is now complete below.

The review did not recommend abandoning the multi-layer product goal. It did
recommend changing the work pattern: stop tuning local acoustic heuristics, and
make the next model question one bounded task with fixed positive and negative
references. The stem-free SSM screen below is a no-go for the current section
baseline on these references, not evidence that all section models fail or that
the broader product direction is wrong. Do not add three large models to
compensate for this result. A further CPU-only feasibility route for the already
selected All-In-One section candidate is a possible single next experiment, but
the previously granted extra setup attempt has been used. Any new model/runtime
setup therefore needs fresh, explicit authorization. No such attempt has been
started.

### Canonical transfer-screen result — 2026-09-24

Run 02 in the documented `.songviz/venv` is canonical. It used the predeclared
unchanged stem-free `compute_story` call and fixed windows. The gate **fails**:
neither positive window contains a final internal boundary, while the
Feel Good Inc no-change window is empty. The Arctic event-layer scope window is
also empty and was not scored. All four runs report `section_method=ssm` and no
section error.

| Track | Final internal boundary count | Boundaries in fixed window | Count/minute |
| --- | ---: | --- | ---: |
| Agnes | 7 | none (positive missed) | 1.994411 |
| Castlecomer | 4 | none (positive missed) | 1.338788 |
| Arctic Monkeys | 7 | none (event-only, unscored) | 1.541884 |
| Feel Good Inc | 6 | none (negative passed) | 1.627683 |

Feel Good Inc's in-sample descriptive nearest-boundary counts are 3/18 within
0.5s, 5/18 within 1s and 7/18 within 3s. These were not gate criteria and are
not calibrated accuracy estimates. The saved report contains every boundary
and the full diagnostics:
[`run 02 report`](../outputs/reviews/ssm-section-transfer-02/report.md).

Integration independently checked source and manifest artifact hashes,
re-derived each scored boundary list and window join from the saved story files,
and confirmed identical story/report hashes and predictions across run 01 and
run 02. Thus the Python/NumPy environment difference did not alter these saved
predictions. Focused validation: **3 tests passed**, with one existing ddtrace
warning. `git diff --check -- CONTINUE.md` and the whitespace check for this
document passed. The transfer runner and test are retained; no production code,
threshold, model or runtime changed. This cross-environment replication is not
independent validation because run 01's output was already visible.

The result is actionable but narrow: the existing section baseline misses both
approximate new-part references under the fixed scoring windows and avoids the
negative control. Stop this baseline without tuning. The immediate work queue is
closed pending a decision on whether to authorize one additional, bounded
CPU-only feasibility attempt for this same section candidate. Do not start a
different model, broaden the setup search, or request another listening note
under this result alone.
