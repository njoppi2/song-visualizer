# Continuous all-stem role context

This Team 1 development experiment follows [21](21_change_episodes.md) and the
existing listening judgments in [20](20_listening_feedback.md). It asks whether
the descriptive evidence around a perceived change survives nearby choices of
anchor and scale. It does not estimate musical leadership, vocal function,
perceived importance, or physical transition endpoints.

## Fixed design

Compute the five existing all-stem descriptors at every supported boundary of
the verified beat grid, with 2, 4 and 8 beats on **each** side. The descriptors
are mean RMS, fraction of beats above the stem's audibility floor, fraction of
summed stem RMS power, concentration of averaged log-CQT, and mean adjacent-beat
spectral change. Preserve their existing definitions for comparison with doc 21.
Add signed right-minus-left differences for all five descriptors. Unsupported
windows and unavailable descriptors remain null; semantic fields remain unknown.

The audibility floor is `max(0.02 * whole-track peak stem RMS, 1e-8)`. Local
feature support ends at the right-window endpoint, but calculating the floor
requires the whole track. Support timestamps therefore do not certify streaming
availability. The inherited spectral concentration averages the entire window
if any beat is audible; adjacent spectral change uses only consecutive pairs
whose energies and spectra are usable. These proxies do not identify pitch,
speech, laughter, or leading/background roles.

The numeric extractor receives only features, stem RMS and beat endpoints. It
does not receive human annotations, selected episodes, candidate thresholds or
an event list. The four existing excerpts are used afterward for evaluation.
Choose the nearest grid boundary to each existing focus band's midpoint, with
the earlier boundary winning ties; inspect offsets -2, -1, 0, +1 and +2 beats
at all three scales. This fixed sampling rule is established before the new
extractor runs. Focus bands remain prior listening prompts/reference context,
not newly measured physical intervals. No best anchor or scale is selected.

For each stem, scale and signed descriptor, retain the five values, their
minimum/maximum and positive/negative/zero/unknown counts. Counts use arithmetic
sign, without a tuned perceptual tolerance. Small nonzero changes may be
numerically real and perceptually irrelevant. Full local curves permit further
inspection without changing this fixed evaluation set.

## Acceptance boundary

- Preserve all scales and supported boundaries, with explicit before/after
  windows, local support end and semantic unknowns.
- Check constant/silent inputs, hand-computed shares, signed trends, irregular
  time support, input preservation and parity with existing peak descriptors.
- Check shared gain scaling above the absolute audibility floor: RMS and its
  difference scale; activity, shares and spectral descriptors stay the same.
  Explicitly retain the absolute-floor exception at very low gain. This is a
  descriptor-level test with fixed cached spectra, not a claim that recomputing
  log-CQT after changing source gain would preserve its concentration.
- Check independence from candidate extraction and human evaluation labels.
- Verify cached source/audio/grid and feedback provenance before reuse. Bind
  code, exact raw feedback, outputs and page derivation in a new package.
- Verify native original-audio playback, bounded audition, seeking/retry,
  navigation, selectable context and mobile presentation.

The lead designs and reviews the experiment. Two Terra workers implement the
numeric extractor/tests and the package/page/tests in disjoint paths. Team 2's
direction files and frozen artifacts remain under its separate ownership.

## Development findings

The verified cache has 498 beat intervals / 499 grid boundaries and four stems.
The extractor retains 495, 491 and 483 supported contexts at scales 2, 4 and 8,
respectively: **1,469 all-stem contexts**. These are overlapping measurements,
not independent musical events. Edge windows remain null.

| Existing listening case | Fixed anchor (s) | Five neighboring anchor range (s) |
| --- | ---: | ---: |
| Drum entry (`local`) | 79.103417 | 78.237145–79.969690 |
| Within passage (`none`) | 20.630016 | 19.763743–21.496288 |
| Verse ending (`subtle`, qualified in note) | 124.582729 | 123.716457–125.449002 |
| Breakdown (`broad`) | 63.077374 | 62.211101–63.943647 |

At the verse ending, vocal mean RMS and RMS-power share decrease at all 15
fixed anchor/scale combinations; active fraction is 1 on both sides throughout.
The power-share differences in percentage points are:

| Half-window scale | Minimum difference | Maximum difference | Negative / available |
| --- | ---: | ---: | ---: |
| 2 beats | −18.145 | −4.602 | 5 / 5 |
| 4 beats | −17.247 | −9.635 | 5 / 5 |
| 8 beats | −16.210 | −13.399 | 5 / 5 |

The evidence for continuing voice with reduced acoustic contribution therefore
survives this bounded anchor perturbation; it is no longer contingent on a
selected bass episode. The magnitude remains sensitive to anchor and scale.
Both vocal spectral-concentration differences and adjacent-spectral-change
differences have mixed signs at every scale. This experiment still does not
recognize laughter or loss of a leading role.

At the perceived non-change near 21s, vocal share differences reverse sign
within each scale, and many accompaniment trends also vary. However, the
`other` stem's mean RMS **decreases at all 15 combinations**, as does its
adjacent-spectral-change descriptor. Stable acoustic change coexists with the
human `none` judgment. Stability is insufficient as a perceptual-importance
model; the result must not be used to manufacture a change label here.

At the drum entry, drum mean RMS increases at all 15 combinations. Its power
share increases at 14/15, reversing only at the latest 2-beat anchor, after
the entrance has entered the left window. At the fixed central anchor, vocal
mean RMS rises at all three scales while vocal power share falls. This is a
concrete denominator effect: reduced share alone does not establish that an
instrument became quieter. Bass entry remains uncertain as in the raw note.

Across the breakdown's neighboring anchors, drum power-share differences
reverse sign at every scale; 8-beat drum RMS decreases at all five anchors.
The wider vocal windows begin to include the subsequent vocal return, changing
the sign and magnitude of vocal share. Full before/after support is needed to
interpret this mixture of disappearance and recovery. These descriptors do not
resolve physical transition onset/settling or validate new precise endpoints.

## Decision and next bounded step

Continuous context fixes the dependence of descriptor availability on an
episode threshold. It supports inspecting a small graded-emphasis study around
the verse ending, using the existing human judgment to choose the study and
the descriptor ranges to expose sensitivity. An automatic share-to-emphasis
policy is not supported: the non-change counterexample and denominator effect
remain unresolved. Any visual emphasis curve is an authored experimental
choice and must preserve its provenance and semantic unknowns.

Team 2's independently assigned directing study can continue using its existing
frozen inputs. This package does not silently introduce a new director
interface. The next Team 1 step is to review the saved-plan comparison from
Team 2 against the existing four judgments, checking whether gradual emphasis
preserves continuity and avoids reacting to the `none` example. If that study
does not cover these excerpts, first record its actual coverage and specify
one bounded follow-up passage; do not infer musical acceptance from unrelated
rendering checks. No repeated annotations are required to finish this analysis.

## Validation and package

Final package: [role-context-02](http://127.0.0.1:8770/role-context-02/), stored at
`outputs/reviews/role-context-02/`. `role-context.json` retains the full grid
(7,818,955 bytes); `review.json` contains only the four excerpts, and
`evaluation.json` contains the fixed neighboring-anchor comparisons. The page
is 1,095,640 bytes and references the existing original WAV. Source snapshots,
exact raw feedback and provenance are in `inputs/` and `manifest.json`.

The first new package, `role-context-01`, failed the mobile browser assertion:
the unbroken review fingerprint made a 375px viewport scroll to 623px. It is
retained unchanged as a development artifact. The lead added wrapping to the
footer and generated version 02. Full numeric, evaluation and review JSONs are
byte-identical between the two packages; no analysis parameters changed.

Validation actually performed:

- Seven extractor tests and three package tests passed. The extractor plus
  previous response-episode tests passed together (23 tests, one warning).
  Tests cover silence/constants, exact shares/differences, gain behavior and
  its absolute-floor exception, support, descriptor parity, input preservation,
  candidate independence, feedback/label isolation, immutability and tampering.
- Lead checks passed all 1,469 support/share invariants and exact descriptor
  parity with all 259 frozen episode peaks (maximum absolute error **0**).
- Full shared-worktree suite: **643 passed, 123 warnings**, 17.76s. This included
  concurrent Team 2 work but does not constitute its integration review. Later
  edits only corrected graph labels, the smoke-test caption selector and footer
  wrapping; package tests were rerun after the graph/selector changes, and the
  final page/provenance and browser checks ran after the footer fix.
- All **78** final package source/snapshot/output fingerprints, sizes, HTML
  derivation, original-audio binding and exact raw note contents verified.
  Every bounded curve matches the full-grid subset, all 60 fixed anchor/scale
  contexts match the grid, and all 240 descriptor summaries were independently
  recomputed. The manifest SHA-256 is
  `129eab2556bc43bcef9a351015eed4847ca4b81b3555fe1afbeb615943af6cbe`.
- Browser smoke passed native original audio, bounded stop, seeking/retry,
  four-case navigation, actual scale/anchor updates, real null spectral values,
  mobile width and no console errors. Extra checks found no page overflow at
  320, 375 and 768px. Desktop and mobile screenshots of selected vocal context
  were captured and visually inspected at
  `/tmp/songviz-role-context-02-desktop.png` and
  `/tmp/songviz-role-context-02-mobile.png`. The dense descriptor table scrolls
  within its container; this is a technical review page.

Two Terra workers implemented the extractor/tests and package/page/tests. The
lead reviewed actual source and evidence, requested stronger gain/null checks,
corrected provenance/support validation and display details, and performed final
package/browser verification. Workers have finished. No source audio, raw
feedback, previous control or Team 2 file was changed by Team 1. The ranged
review server was started on port 8770 and left available for the review.

Reproduce into an unused output directory (the final package refuses overwrite):

```bash
.songviz/venv/bin/python experiments/build_role_context_review.py --out <new-directory>
.songviz/venv/bin/python -m pytest tests/test_role_context.py tests/test_role_context_builder.py -q
SONGVIZ_PLAYWRIGHT_MODULE=/home/njoppi2/.npm/_npx/04489aa25f0b609a/node_modules/playwright \
  node experiments/check_role_context_review.cjs http://127.0.0.1:8770/role-context-02/
```

Implementation verification is complete; musical role/importance inference and
production promotion remain unaccepted. No new user feedback is required for
this completed experiment.

## Team 1 integration review of the directed comparison — 2026-09-11

The subsequent review inspected Team 2's completed handoff in
[doc 10](10_directing_prototype.md), the frozen `directed-review-02` and
`directed-replay-02` artifacts, their source snapshots, and the current diffs.
One Terra worker reviewed the direction/render contract read-only; the lead
reviewed package construction and media/browser evidence and independently
reproduced the findings below. Team 2's files were not edited.

### Coverage and visual observation

The comparison covers **130–178s**. None of the four existing focus intervals
falls within it. Only 130–132s, the last two seconds of the verse-ending audition,
overlap an existing excerpt; the focal vocal development around 124s is absent.
The other three excerpts have no overlap. Those four judgments therefore cannot
serve as acceptance labels for this video.

Actual MP4 frames at 150s and 162s show a faint additional vocal ribbon above
the central accompaniment ribbon, stronger at 162s. The central focus remains
the accompaniment. This confirms the modest visual difference reported by
Team 2; it does not establish that the viewer follows musical attention better.
The warm visual vocabulary returns in the saved plan, but no verified musical
identity recognition is inferred from that design choice.

### Verified evidence

- Verified the parent package's 24 recorded hashes, the candidate/replay's
  55 snapshot/output hashes plus 13 nested origin records, and both recorded
  manifest digests. Both HTML pages regenerated exactly from their frozen
  templates/plans/manifest hashes; the page-derivation hashes also matched.
- All eight saved plan/signal/PCM/video files checked matched between candidate
  and replay. `coarse.mp4` and `fixed.mp4` exactly match the original version-01
  control videos. This rechecks recorded replay equality without rerendering.
- All three MP4s have 2,880 frames at 60 FPS and 48s duration. Their decoded
  audio is identical; original PCM equals the FLAC's 130–178s samples exactly.
  AAC/source zero-offset correlation is 0.9972178113557956.
- Frozen plan validation passed. Current focused direction/render/builder tests
  passed **52 tests, 1 warning**; this includes later code and does not certify
  its unhanded visual experiment. No new full-suite run was necessary for this
  read-only review.
- Browser checks passed native video, same-position paused/playing switching,
  simulated load-failure restoration and retry, song timestamp conversion
  (local 32s → 162s), empty-feedback rejection, and 320/375/768px layout including
  expanded details. No page errors. An initial run timed out and the local
  server subsequently refused connections; after restarting the absent server,
  the staged checks passed. No user feedback was created or exported.
- Actual video frames and desktop/mobile pages were visually inspected.
  Temporary evidence: `/tmp/songviz-team1-direction-review.fPGoWl/`, including
  `directed-150.png`, `coarse-150.png`, `directed-162.png`, `coarse-162.png`,
  `desktop.png` and `mobile.png`.

### Contract findings requiring Team 2 follow-up

1. **Medium: v2 focus validation uses a different gain from rendering.**
   The frozen `inputs/songviz/direction.py:84` requires the legacy scalar gain;
   line 107 uses it to decide whether focus is visible. The frozen renderer's
   `_layer_values` at `inputs/songviz/directed_render.py:123` evaluates the
   envelope for v2, and drawing at line 245 uses that value. The lead made an
   in-memory copy of the saved plan, set every envelope gain to zero (preserving
   curve continuity), and `validate_plan` still accepted it because the scalar
   focus gains were positive. The selected plan itself also contains scalar /
   first-envelope differences after continuity adjustment: pulse .08/.12 at
   137.917347s, pulse .12/.08 and vocals 1/.4161526466 at 165.735329s. This does
   not corrupt the existing MP4s, whose renderer consistently uses curves, but
   the saved-plan contract needs one explicit authority and matching validation.
2. **Low: envelope support metadata is recorded but not validated.**
   The planner records `envelope_support` at frozen `direction.py:279–287`.
   The validator checks evidence IDs at line 109 but does not require or check
   those support records. Removing every `envelope_support` from an in-memory
   copy still passes validation. The selected artifact's support appears
   consistent; the gap concerns rejecting future missing/malformed provenance,
   not a demonstrated wrong time in the frozen output.

At that review, the frozen package was accepted as inspectable development
evidence, while general schema-v2/interface acceptance awaited these repairs.
Their subsequent verification is recorded below; artistic acceptance and
production promotion remain separate.

### Ownership and follow-up scope

The four live direction/render/builder/template files now differ from their
version-02 snapshots and contain later visual-vocabulary/layout work absent
from the completed handoff. That work belongs to Team 2; it was neither reverted
nor silently included in this acceptance decision. Team 2 must reconcile the
bounded repair request with its current work and return a separate handoff.
The request was recorded in `CONTINUE.md`, the sole current queue, and has now
been closed following the verification below.

After contract repair, the concrete uncovered passage for a role-context
direction follow-up is **119–132s**: compare an unchanged-focus control with an
explicitly authored gradual reduction of vocal visual emphasis while preserving
an audible/visible vocal trace. Keep identical audio, timing and treatments so
that emphasis is the changed variable; retain the user's qualified `subtle`
note. Choose the visual envelope as a documented experimental decision, not an
automatic mapping from RMS share. New output ownership must be assigned before
that implementation; this review only specifies the experiment. The 21s `none`
case remains a required counterexample before any general automatic policy.

## Repair closure and visual-pass integration review — 2026-09-11

Team 2's combined handoff in [doc 10](10_directing_prototype.md) addresses both
contract findings and the user's separately recorded visual feedback. Team 1
reviewed the actual source, tests, saved plans, media and browser behavior. One
Terra reviewer independently checked visual-only isolation, replay and legacy
rendering; the lead checked both repaired failure cases and final artifacts.
No Team 2 files or existing packages were edited during integration review.

**Both contract findings are closed.** Schema v2 now uses envelope gain for
focus validation; a positive interior value allows intentional endpoint fades.
The scalar gain is explicitly finite/in-range compatibility metadata, not v2
render/focus authority. Original frozen 02 plans still validate unchanged.
The lead reran the former all-zero-envelope and missing-support reproductions
against both frozen 02 plans: both are now rejected. Generated support is
checked against the segment and completed-block signal clock; authored
envelopes require explicit provenance. The regression tests also cover malformed
and off-clock support and prevent presentation metadata from bypassing checks.

The inspected visual artifact is
[directed-visual-02](http://127.0.0.1:8770/directed-visual-02/), with saved replay
at `outputs/reviews/directed-visual-replay-02/`. It preserves all **3,395**
keyframes, scene intervals, visibility, gains, emphasis, focus, palette/motif,
crossfades and evidence from the prior gradual plan. Only layer treatments,
anchors and declared visual authorship change in the plan. Renderer changes
are opt-in and affect appearance, composition and perceived brightness.

Actual MP4 frames at 162s and 168.771s show vertical vocal strands, broad lower
accompaniment contours, and angular snare marks at the right. At 131.094s the
kick is a compact low oval with a halo. The voice/snare silhouettes are now
visually distinct in the inspected overlap frame. This is an observed visual
difference, not acceptance of musical timing, attention or artistic quality.
The passage is still 130–178s; none of the four prior focus intervals is covered.

Final verification:

- Lead focused suite: **71 passed, 1 warning**. Lead full shared-worktree suite:
  **666 passed, 123 warnings**, 17.42s. The Terra reviewer independently ran
  the 71 focused tests and found no blocking regression.
- All **46** candidate/replay snapshot/output hashes and **15** origin records
  verified, as did exact HTML derivation and page-derivation hashes. Current
  implementation files match the new package snapshots. Candidate manifest:
  `95c47b36345d25aa66bb16adfb1a0a7a7289e2998e2851114b05c28d6fa7cc60`;
  replay manifest:
  `ffb3c99452a2d9b92f397a33ffd15478f05e7fefdfa57d14cded91609836f611`.
- Both plans, signals, PCM and both MP4s match the saved replay byte-for-byte.
  `previous.mp4` exactly equals the original `directed-review-02/directed.mp4`.
  The independent reviewer additionally compared current/frozen-renderer pixels
  at eight timestamps each for the original v1 and gradual v2 plans: identical.
- Both videos are 48s, 60 FPS and 2,880 frames; decoded audio is identical.
  PCM equals the original FLAC's 130–178s samples exactly.
- Browser checks passed native playback, two views, playing/paused same-position
  switching, shortcuts, seeking, simulated failure/retry, and feedback
  time/version/exact-manifest linkage. Synthetic export was intercepted in
  memory and not saved as user feedback. There were no page errors or overflow
  at 320/375/768px, including expanded details. Desktop/mobile pages were
  visually inspected; temporary evidence is in
  `/tmp/songviz-team1-visual-review.PXeLxa/`.
- `git diff --check` passed. No packages were regenerated for this review.

**Decision prepared at that review:** review the **Voice + drums · 2:48** shortcut
in the visual comparison and determine whether voice, snare and kick are easier
to distinguish while playing. This addresses the new visual feedback, not the
four earlier musical annotations. Contract repair no longer blocks further
experiments. The proposed 119–132s emphasis study remains a separate, unassigned
follow-up; this integration review does not silently start it or promote a
full-song director.

## Authored-emphasis design review — 2026-09-12

Team 1 reviewed Team 2's prepared proposal at the top of doc 10, the actual
renderer/envelope and package-input contracts, and the original listening note.
The human confirmed this session is **Team 1**, also called Team A. The newer
visual feedback in doc 10 is **"they are better then before."** It establishes
relative improvement, not musical timing or automatic-policy acceptance; it
supersedes the pending visual-improvement question above without altering any
frozen review manifest or raw feedback.

The lead accepts the proposal as a bounded authored experiment and assigned its
implementation to Team 2 in `CONTINUE.md`. That checkpoint remains the only live
assignment. A Terra worker independently reviewed the proposal read-only while
the lead verified its inputs and inspected the actual renderer and builder.
No Team 2 implementation, tests or existing artifact was changed in this review.

### Fixed comparison and acceptance boundary

- One scene covering absolute song time **119–132s**, with `focus: "vocals"`
  in both plans, the refreshed visual identities/anchors, and identical
  visibility, palette, motif, background, transition settings and evidence.
  All accompaniment envelopes stay constant and identical between versions;
  declare their chosen values in the saved plan rather than optimizing them
  separately for either version. Source-driven activity remains unchanged.
- Use the same four vocal knot timestamps in both versions: **119, 123, 125.5,
  132s**. Steady control gain/emphasis is **1/1** at every knot. Candidate is
  **1/1, 1/1, .45/.30, .45/.30**, with linear interpolation. The two parameters
  jointly manipulate visual prominence; this is not a separate causal test of
  gain versus emphasis. Times/values are authored choices around the existing
  development example, not detected transition boundaries or tuned RMS shares.
- Existing primary-focus rendering gives vocals full alpha importance:
  **gain changes opacity; emphasis changes anchored geometry**, not primary
  alpha importance. Holding focus fixed isolates this behavior. Do not change
  renderer semantics to make this study, or claim metadata focus itself moves
  away from vocals. The constant-gain control still responds to source activity;
  it is not constant screen brightness.
- Create a distinct paired-plan validation contract; the existing visual-only
  comparison correctly forbids envelope changes and must retain that behavior.
  Allow only vocal envelope gain/emphasis and explicitly named authorship
  descriptions to differ. Knot times, every other plan field, signals and audio
  must match. Test rejection of changed accompaniment, focus, intervals,
  treatments/anchors, evidence and undeclared provenance changes. Enforce the
  same isolation and fixed-curve boundary on any supported replay override.
- Both plans must be explicitly authored v2 with meaningful
  `envelope_provenance`, not the generated gradual-activity planner label.
  Bind the exact raw feedback file/hash and `verse-ending` answer; preserve its
  qualified `subtle` judgment and complete note. Preserve the distinction
  between signal provenance/clock support and manually chosen envelope times.
  A valid authored envelope is not evidence of inferred laughter or leadership.
- Verify the frozen parent before reuse. Keep its full-track signals unchanged;
  freshly cut the hash-bound original FLAC at 119–132s. The parent 130–178s WAV
  cannot supply this earlier excerpt. Expected PCM is **573,300 stereo frames
  at 44.1kHz**; compare samples against the exact source interval, not duration
  alone. Bind code snapshots, signals, raw note, PCM, both plans and page
  derivation in the new package.
- Replay both saved plans from the new saved signals/PCM, verifying all recorded
  outputs, snapshots and origins. No replanning, source cutting or extraction
  on replay. Both saved plan bytes, signals, WAV and both rendered MP4s must
  match their replay counterparts. Each video is **13s / 60 FPS / 780 frames**.
  Decoded AAC must match between the two versions; compare WAV PCM separately
  against FLAC. Lossy AAC is not expected to equal original PCM sample-for-sample.
- Test evaluated curves before, inside and after the ramp. Require the vocal
  layer enabled, intended focus and positive late gain; inspect actual late
  frames where source vocal signal is present. Nonzero gain does not guarantee
  visible pixels during source silence or after raster rounding. If the trace
  is too faint, report the failed observation rather than silently changing the
  fixed parameters. Inspect shape separation and native same-position playback,
  seeking/failure/retry, paired feedback linkage and 320/375/768px layouts.

No automatic role/share-to-emphasis policy, detector, new role-context interface
or production promotion is included. The 21s `none` counterexample remains
required before generalizing a policy; it is not an additional video demanded
by this bounded two-version study. User preference remains to be collected
after Team 1 verifies the completed artifact, not inferred from technical checks.

### Validation actually performed for this assignment

The lead reran the existing frozen-manifest verifier on `directed-visual-02`:
all **23** recorded snapshots/outputs and its recursively recorded origins
passed. Independently computed SHA-256 values:

- Parent manifest:
  `95c47b36345d25aa66bb16adfb1a0a7a7289e2998e2851114b05c28d6fa7cc60`.
- Parent full-track signals:
  `a0e5d277967f6e2a27735243fa8766dc168dbc24b25691012c7595bea771286f`.
- Raw `benchmark/feedback/listening-examples-01.json`:
  `f69b14f0343d0ebe4af12d37f66e88c76aa5d8e7673f7529345333589d5c47b6`.

Both proposed new package paths were absent at assignment. Team 2 must check
again before creating them and preserve any output that appears meanwhile.
The Terra review identified focus semantics, strict pair isolation, authored
provenance, fresh audio cutting and true two-plan replay as important explicit
constraints; the lead checked these against source and incorporated them above.
No implementation tests, new renders or browser checks were run for this
documentation-only assignment. The **666-test** result above belongs to the
previous integration review, not a new study. No new failures were observed in
this input/design review; the comparison itself remains unimplemented.

## Authored-emphasis integration review — 2026-09-12

Team 1 reviewed Team 2's completed handoff in doc 10 and the actual source,
tests, paired plans, frozen inputs, rendered media and browser behavior. A Terra
worker independently reviewed the direction/builder contracts and ran focused,
full-suite, artifact and browser checks. No blocking correctness, provenance or
isolation finding remains. The lead accepts the implementation as the assigned
authored comparison, **not** as the user's musical preference or a production
policy. The completed assignment is closed in `CONTINUE.md`.

Candidate:
[directed-vocal-emphasis-02](http://127.0.0.1:8770/directed-vocal-emphasis-02/).
Replay: `outputs/reviews/directed-vocal-emphasis-replay-02/`. Both are frozen.
`plan.json` / `directed.mp4` are Reduced; `baseline-plan.json` / `steady.mp4`
are Steady. Manifest hashes:

- Candidate: `7f0e0733cc7e4db25bf75d9d4e35e559c198035de6ce02ff4a4622d99964ab01`.
- Replay: `88a8224337cf4941d3a62eaabc8a07e3829cfa423bd937f3f11a7f298dad0e74`.

### Independent source and isolation review

The lead's recursive comparison found exactly **five differing leaf fields**:
`envelope_provenance.description` plus vocal gain/emphasis at knots 125.5 and
132s. All other fields match. The four timestamps are 119, 123, 125.5 and 132s;
Reduced is 1/1, 1/1, .45/.30, .45/.30 and Steady is 1/1 throughout. The one scene
keeps vocal focus and identical constant accompaniment. All seven treatment/
anchor pairs and the visual policy match the frozen visual parent. The live
renderer is byte-identical to its prior visual-pass snapshot: this study did
not change rendering semantics.

The separate pair contract retains exact curve/interval requirements and a
narrow difference whitelist, including replay overrides. Build inputs bind the
reviewed parent, full-track signals, source FLAC and raw feedback; both plans
preserve the complete qualified `subtle` verse-ending note. Replay loads both
saved plans/signals/PCM without replanning, cutting or separation. Test coverage
includes alternate JSON formatting, source-clock binding, shared-field and
provenance faults. The old visual-only contract still forbids envelope changes.

### Validation actually performed

- Lead full shared-worktree suite:
  `.songviz/venv/bin/python -m pytest -q --disable-warnings` — **695 passed,
  123 warnings**, 20.60s. Independent Terra focused suite — **100 passed,
  1 warning**; its full suite also passed 695 tests. `git diff --check` passed.
- Lead read and reran `tests/test_directed_emphasis_artifacts.py`, changing only
  its diagnostic image destination in memory to a fresh temporary directory.
  All **21 candidate + 24 replay** snapshot/output hashes and recorded origins
  passed. Current five source snapshots, byte-identical raw feedback and full
  note, parent signals, exact HTML reconstruction and page derivation passed.
  An additional independent recursive origin check verified the candidate's
  three and replay's four directly nested path/hash records.
- Both plans, full-track signals, WAV and both MP4s match their saved replay
  counterparts byte-for-byte. Each video is **13s / 60 FPS / 780 frames**.
  The two decoded AAC streams are identical. The separate WAV contains exactly
  **573,300 stereo frames at 44.1kHz**, sample-identical to source FLAC at
  **119–132s**. AAC is not equated to lossless PCM. No package was regenerated.
- Lead read and reran `tests/test_directed_review_browser.cjs` on the actual
  candidate, redirecting only screenshots to the fresh temporary directory.
  Native play/pause, playing and paused same-position switching, seek/restart,
  123/127s shortcuts, initial failure/retry, actual 15-second metadata timeout
  with restoration, abort recovery and post-metadata error handling passed.
  Feedback export preserved exact manifest hash, passage, absolute song time,
  viewed version and preference. Synthetic content was intercepted in memory;
  no download or raw human-feedback file was written. Layout passed at
  320/375/768px with details expanded; no page errors occurred.
- Lead independently exercised the **current** template with in-memory HTML
  interception against all three older modes: `directed-review-01` two-way,
  `directed-review-02` three-way and `directed-visual-02`. Every visible version
  selected its correct native MP4 and preserved paused clip time at 8s; no page
  errors. Frozen pages were not changed. Renderer byte equality plus existing
  tests support legacy behavior; no new old-mode videos were rendered.
- Lead visually inspected desktop/mobile screenshots and ten decoded MP4 frames
  at **121.2, 124.25, 126.5, 129.2 and 131.1s**, paired across both versions.
  Before the ramp the render is identical. Later Reduced vocals are smaller and
  fainter, while accompaniment remains the same. Vocal strands and snare marks
  remain distinct in sampled frames. The late vocal trace is visible in these
  non-silent samples, but especially faint near 129.2s. This is a readability
  caveat for the user, not grounds to retune the fixed study silently. Expanded
  mobile details fit but have tightly wrapped table cells.

Lead diagnostic evidence is in `/tmp/songviz-team1-emphasis-review.Z6Nzmr/`:
`frames.png`, `desktop.png`, `mobile.png`. Screenshots contain synthetic test
form values, not new user feedback. Team 2's browser diagnostic is accepted as
a read-only integration helper; no implementation file, source input or frozen
artifact was edited by Team 1. No new failure was observed in this review.

### Prepared user decision

Watch both versions, using **After the reduction · 2:07** to switch at the same
moment. Does reduced or steady vocal prominence fit this ending better, and is
the reduced voice still readable? Neither/unsure is valid. This new preference
is not answered by the previous visual-improvement feedback. No repeated
annotation of the four earlier musical examples is required. The 21s `none`
counterexample and automatic role/importance inference remain unresolved and
outside this completed implementation. Choose a subsequent bounded task only
after triaging the response; do not silently promote or optimize the candidate.

### Subsequent priority change — 2026-09-13

The user explicitly prioritized song understanding and explanatory text/plots
before artistic visualization, and identified this account as Team 1. The above
prepared preference is now deferred rather than a prerequisite for further
analysis. The new experiment design is in [doc 24](24_music_understanding.md);
the active team assignments and requests are maintained in CONTINUE.md. This
changes the work priority, not the musical acceptance of the frozen comparison.
