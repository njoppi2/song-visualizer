# First directing prototype — 2026-09-10

## Authored vocal-emphasis comparison — 2026-09-12

Team 2 implemented the bounded request assigned in `CONTINUE.md`, after the
human explicitly identified this session as Team 2. The saved comparison is
[directed-vocal-emphasis-02](http://127.0.0.1:8770/directed-vocal-emphasis-02/),
with an independently rendered replay at
`outputs/reviews/directed-vocal-emphasis-replay-02/`. Both target directories were
absent before creation; previous packages, source audio and feedback were not
changed. Team 1 owns integration and the checkpoint; Team 2 has not edited
`CONTINUE.md` or adopted an automatic role/importance policy.

### Fixed experiment actually built

One scene covers **119–132s**, with vocal focus, unchanged refreshed visual
identities/anchors and identical constant accompaniment. The two videos are
**Reduced vocal emphasis** (`directed.mp4`, `plan.json`) and **Steady vocal
emphasis** (`steady.mp4`, `baseline-plan.json`). Both use vocal knots at 119,
123, 125.5 and 132s. Steady remains gain/emphasis 1/1; Reduced is 1/1, 1/1,
.45/.30, .45/.30, with linear interpolation. Nothing else changes between
plans except their explicitly named `envelope_provenance.description`.

Both plans declare authored v2 provenance. These are experimental settings,
not detected boundaries or calibrated musical importance. Vocal focus remains
fixed: gain changes opacity and emphasis changes geometry under the existing
renderer, without moving primary-focus alpha importance to another source.
The scalar vocal gain stays 1 as compatibility metadata; v2 envelopes are
authoritative. The control still follows source activity, not constant brightness.

Constant supporting gain/emphasis values, declared identically in both plans:

| Layer | Gain | Emphasis |
| --- | --- | --- |
| Pulse | .10 | .25 |
| Kick | .38 | .35 |
| Snare | .42 | .45 |
| Hi-hat | .18 | .22 |
| Bass | .26 | .30 |
| Other | .32 | .35 |

The exact raw `verse-ending` answer, including its qualified `subtle` judgment
and complete note, is copied into both plans and bound to the original feedback
file/hash. A byte-identical raw-file snapshot is saved in each package. The
full-track signals are copied unchanged from the verified visual parent.
Fresh PCM was cut from the bound FLAC at 119–132s; the parent's 130–178s WAV was
not used for this earlier excerpt. Replay loads/copies the saved plans, signals
and PCM and re-renders both videos without planning, cutting or extraction.

### Implementation and reliability checks

`make_vocal_emphasis_plans` and `validate_vocal_emphasis_comparison` provide a
separate contract from visual-only comparison. A narrow difference whitelist
and fixed-curve checks reject changed accompaniment, vocal scalar gain, focus,
interval, knot times, treatments/anchors, evidence and undeclared provenance,
including replay overrides. Valid alternative JSON formatting is copied
verbatim on replay. The existing visual-only contract still forbids envelopes
from changing. No renderer semantics were changed for this experiment.

The builder adds `--vocal-emphasis-from`, verifies frozen parent/source/feedback
hashes before reuse, saves both plans and native MP4s, and binds code snapshots,
raw feedback, signals, audio, plans and HTML derivation. Legacy replay now also
rejects duplicate or misbound required output records instead of accepting an
ambiguous first match. Test fixtures are synthetic; the optional real-input
check skips if ignored development media are unavailable in another checkout.

The review page has two native-video views, same-position switching, shortcuts
to 123s/127s and manifest-linked optional feedback with `passage_id: verse-ending`.
The interrupted reliability audit was completed: metadata loading now times out
after 15s, restores the prior clip when possible and re-enables retry; a pending
`play()` cannot indefinitely lock controls. Persistent media errors disable
feedback and offer retry, and buffering is reported honestly. Existing frozen
pages were not rewritten. The current template was separately exercised against
all three old comparison modes using in-memory HTML interception.

A real 320px expanded-details table overflow was found using a temporary browser
fixture, then corrected with narrow-screen cell wrapping before final rendering.
Browser-harness mistakes (initial asset selection and Playwright's fieldset
disabled query) were corrected rather than counted as application failures.
Final package tests pass; the secondary details table still wraps tightly on
narrow phones. Temporary fixture videos were test media only, not review evidence.

### Validation actually performed

- Canonical `.songviz/venv` focused suite: **100 passed, 1 warning**. Final full
  shared-worktree suite: **695 passed, 123 warnings**. `git diff --check` passed.
  Curve evaluation includes before/at/inside/after the ramp; at 124.25s Reduced
  evaluates to .725/.65. Tests cover source-clock equality, positive late vocal
  gain, silence, deterministic seeking, override isolation and provenance faults.
- **21 candidate + 24 replay** snapshot/output hashes and all recorded origins
  verified. Current source snapshots match, raw notes match exactly, and both
  pages derive exactly from their frozen template/plan/manifest with independent
  page-derivation checksums. Signals match the verified parent byte-for-byte.
- Both plans, signals, WAV and **both MP4s replay byte-for-byte**. Each video is
  **13s / 60 FPS / 780 frames**. Decoded AAC is identical between the two views.
  The separate WAV is **573,300 stereo frames at 44.1kHz** and exactly equals the
  source FLAC's 119–132s PCM samples. AAC is not claimed to equal lossless PCM.
- Native browser checks passed on the actual candidate: play, pause, seek,
  restart, paused/playing same-position switches, initial failure/retry, an
  actual 15-second stalled-metadata timeout with restoration, aborted switch
  recovery, simulated post-metadata error, exact song-time/view/preference/hash
  export linkage, and 320/375/768px layouts with details expanded. No JS errors.
  Browser feedback content was synthetic test input, not a new user judgment.
- Lead inspected desktop/mobile pages and ten decoded-video frames at 121.2,
  124.25, 126.5, 129.2 and 131.1s across both views. Voice and snare retain distinct
  silhouettes. Late Reduced vocals remain visible in these non-silent samples,
  but are deliberately faint, especially near 129.2s; parameters were not retuned.
  Nonzero gain does not guarantee visible pixels during source silence.
- Two Terra workers implemented disjoint contracts/builder tests; Luna built
  the browser check. Lead inspected actual changes and reran canonical tests,
  artifact verification and browser checks. A Terra reviewer independently ran
  the artifact verifier and inspected the decoded frame sheet, finding no
  remaining artifact-evidence failure. This is not Team 1's integration review
  or the user's musical/artistic acceptance.

Manifest SHA-256 values:

- Candidate: `7f0e0733cc7e4db25bf75d9d4e35e559c198035de6ce02ff4a4622d99964ab01`.
- Replay: `88a8224337cf4941d3a62eaabc8a07e3829cfa423bd937f3f11a7f298dad0e74`.
- Reduced plan: `0ebd6ed03932f17707a5f3aba314e9b9b9474ada102359f0131b49dd4c8c0cfc`.
- Steady plan: `de3e7dbce64649c2d92078ed4e4d9afc7a17a7a1f338368b2b6fe08ff154c1ef`.
- Original WAV: `69f464bd7d911ce3a9f012e498a4f4271c9fb1031b5ad72fe94c7d055e3bb32e`.

Reproduction/verification commands (the two build commands now correctly refuse
to overwrite their completed destinations):

```bash
.songviz/venv/bin/python experiments/build_directed_review.py --out outputs/reviews/directed-vocal-emphasis-02 --vocal-emphasis-from outputs/reviews/directed-visual-02 --fps 60
.songviz/venv/bin/python experiments/build_directed_review.py --out outputs/reviews/directed-vocal-emphasis-replay-02 --replay outputs/reviews/directed-vocal-emphasis-02 --fps 60
.songviz/venv/bin/python -m pytest -q tests/test_direction.py tests/test_directed_builder.py tests/test_directed_render.py --disable-warnings
.songviz/venv/bin/python -m pytest -q --disable-warnings
.songviz/venv/bin/python tests/test_directed_emphasis_artifacts.py
SONGVIZ_PLAYWRIGHT_MODULE=/home/njoppi2/.npm/_npx/04489aa25f0b609a/node_modules/playwright node tests/test_directed_review_browser.cjs
```

The artifact/browser scripts do not rebuild or mutate packages. Diagnostic
screenshots are under `/tmp/songviz-directed-vocal-emphasis-{desktop,mobile}.png`;
the verifier regenerates `/tmp/songviz-directed-emphasis-verified-frames.png`.

**Handoff to Team 1:** inspect the actual implementation and the two frozen
packages, independently verify isolation/replay/media, then record the result in
`CONTINUE.md`. Only after that review, collect the new steady/reduced listening
preference. No extra annotations are required now. The 21s `none` counterexample
remains necessary before any automatic policy; it is outside this bounded study.

## Historical prepared 119–132s comparison proposal — 2026-09-11

This preparation record is superseded by the assigned implementation above;
its statements about an unassigned study describe the earlier checkpoint only.

After watching the visual pass, the user said: **"they are better then before."**
This records relative visual improvement; it does not establish acceptance of
musical timing, automatic direction, or every individual treatment. The user
then permitted useful continued/parallel work. Team 1's newer checkpoint already
closes integration review, but explicitly leaves the next 119–132s study
unassigned. The following is preparation in Team 2's experiment document;
no new implementation, output directory, assignment or checkpoint was created.

The prepared question is: does gradually reducing vocal visual emphasis during
the previously described verse ending fit the music better than holding that
emphasis steady? Keep the refreshed shapes identical in both versions. Existing
feedback remains `subtle`, qualified as possibly stronger, with voice continuing
as laughter and losing its leading role. This is an authored comparison informed
by that human observation, not automatic laughter/leadership recognition.

Concrete proposed design, for integration-owner assignment:

- One contiguous 119–132s scene; same original audio, cached timing/activity,
  visual identities/anchors, background, palette, layer visibility and focus in
  both versions. Keep vocal focus metadata constant to isolate the envelope.
- Control: vocal gain/emphasis fixed at 1.0 throughout. Candidate: hold both at
  1.0 through 123s, then linearly reduce gain to 0.45 and emphasis to 0.30 by
  125.5s, holding those values through 132s. These endpoints/values are declared
  experimental choices around the existing 124.295s development example, not
  newly measured musical transition boundaries or a policy fitted to RMS share.
- Keep all accompaniment curves identical and constant between versions. Use
  the same source-driven response on both sides. The vocal layer remains enabled
  with nonzero gain; actual silence can still produce no pixels, as in the
  existing renderer. Test before/inside/after the ramp and inspect late vocal
  readability rather than promising constant visible activity.
- Declare authored envelope provenance and link the exact existing human note.
  Compare just **Steady vocal emphasis** and **Reduced vocal emphasis** using
  native same-position playback. No new annotation of the four earlier examples
  is required. The 21s `none` case is still needed before any automatic policy.
- Acceptance checks: plans differ only in vocal envelope and authorship reasons;
  all other fields/signals/audio match; validate effective focus/support contracts;
  exact saved-plan replay; two 13s/780-frame videos at 60 FPS; decoded audio
  equality and source-PCM identity; inspect visible shape separation, late vocal
  trace, seeking/switching and mobile layout. Listening preference remains open.

Read-only preparation verified all visual-parent snapshot/output and origin
fingerprints. Parent manifest SHA-256:
`95c47b36345d25aa66bb16adfb1a0a7a7289e2998e2851114b05c28d6fa7cc60`.
Its `signals.json` SHA-256 is
`a0e5d277967f6e2a27735243fa8766dc168dbc24b25691012c7595bea771286f`;
signal coverage is 0–221.173333s, so the proposed excerpt needs no separation or
new activity extraction. The existing source FLAC is stereo/44.1kHz; a 13s cut
would contain 573,300 frames. The current package's 130–178s WAV cannot supply
119–132s and must not be reused as though it covered that interval.

Descriptive checks of the frozen normalized activity (not absolute loudness):
mean vocals are .5266 over 119–123s, .4567 over 123–125.5s, and .2442 over
125.5–132s; kick/snare evidence continues in all three spans. These checks only
establish available inputs and caution against equating presence with leadership.
They did not choose or validate the authored envelope above.

Proposed human-relayed assignment for Team 1 to record in `CONTINUE.md`:

```text
REQUEST
owner: Team 2
to: Team 1
goal: Assign the prepared 119–132s authored vocal-emphasis comparison.
read: docs/10_directing_prototype.md prepared proposal; docs/23_role_context.md
write_scope: Team 1 updates CONTINUE.md with its decision and exact assignment.
do_not_touch: Frozen packages, raw feedback/audio, Team 2 implementation paths.
validation: Confirm the authored comparison boundary and separate user visual
            improvement feedback from automatic-policy acceptance.
handoff: durable-in-CONTINUE
ask_human_if: The proposed artistic comparison or scope needs a different choice.
```

Suggested Team 2 implementation paths are its existing direction/builder/template
and directing tests/doc 10. Suggested new packages:
`outputs/reviews/directed-vocal-emphasis-02/` and
`outputs/reviews/directed-vocal-emphasis-replay-02/`.
Both were absent at preparation. These are proposed names, not a self-assignment;
the explicit unassigned-study condition in `CONTINUE.md` remains in effect.

## Visual identity pass and v2 contract repairs — 2026-09-11

The user watched only **Gradual attention**, found vocals/snare difficult to
distinguish, described the kick as awkward and the visuals as unfinished, then
explicitly requested the visual pass. This is feedback about that version,
not an A/B preference or new musical annotation. Raw chat wording:

> One thing, um, I saw the gradual tension video, and um, I didn't see the other ones. But, like, one thing I noticed is that I think you're using, like, the same icon or the same type of thing, like, both for the voice but also for the snares. Is that correct? And if so, isn't that bad? Because, like, it seems like they kind of, like, getting mixed up, and I can't really differ them. I don't know. Visuals don't yet look great though, but like, I don't know if we are trying to, like, optimize that right now. But yeah, just telling you, like the, I don't know, the kicks are a little weird. Like, I guess we could improve, like, all of the things, but I don't know if we're trying to improve that or optimize that now.

Inspection confirmed that vocals/snare simultaneously use `ring` after
165.735329s, and vocals/other both use `ribbon` in the preceding sparse span.
The new [visual comparison](http://127.0.0.1:8770/directed-visual-02/) at
`outputs/reviews/directed-visual-02/` compares refreshed identities against the
exact gradual video the user watched. It contains two views and shortcuts to
160s (vocal build) and 168s (voice/drums together). Saved replay is
`outputs/reviews/directed-visual-replay-02/`. Both are new immutable packages
within Team 2's `directed-*-02` scope; neither frozen comparison was overwritten.

### What changed visually

| Sound source | Treatment | Stable placement |
| --- | --- | --- |
| Vocals | Tall flowing, tapered strands (`filament`) | Center, above the accompaniment |
| Snare | Crisp angular broken flashes (`shards`) | Right |
| Kick | Compact filled low oval and halo (`impact`) | Low center |
| Other stem | Broad layered flowing lines (`contour`) | Lower background |
| Bass / hi-hat / pulse | Existing ribbon / ticks / quiet ring | Stable supporting positions |

`make_visual_plan` copies the frozen v2 plan, changing only layer treatment and
anchor plus top-level visual authorship metadata. The three spans, all 3,395
layer keyframes, visibility, gains, emphasis, focus, crossfades, palette/motif
IDs and evidence are preserved. The builder verifies this isolation for new
visual builds and visual replay overrides. New shapes stay recognizable as
focus changes. The renderer adds brighter strokes, opt-in stable composition
and a clean vignette without the static ring/construction guides. These changes
affect perceived intensity through geometry/ink; unchanged plan gain does not
mean identical brightness. Other is drawn behind vocals in the new layout.

The first shape draft was too faint on inspection. Lead refinements increased
vocal scale and contrast, tapered its strands, strengthened percussion marks,
and separated the accompaniment into broad curves. Frame inspection at
131.094s, 150s, 162s and 168.771s, plus the 375px browser view, confirmed that
the visible vocal and snare silhouettes are now distinct. The style remains
an abstract development study; artistic quality/clarity still needs user review.
Kick timing and decay use the same frozen event response, so this pass changes
its appearance without establishing that the underlying kick timing is correct.
The organic shapes are decorative, not a pitch or vocal-function transcription.

### Reconciled Team 1 repair request

During this work, Team 1 posted the bounded v2 contract findings in `CONTINUE.md`
and doc 23. A separate Terra worker implemented those repairs in Team 2's
direction/contract tests, preserving the ongoing visual changes:

- V2 envelopes are the authoritative gain representation. Focus must be visible
  and have positive envelope gain somewhere in its span; intentional zero
  endpoint fades are allowed. Scalar `gain` stays finite/in-range compatibility
  metadata, explicitly ignored for v2 focus/render behavior. The frozen 02
  scalar/curve mismatches remain accepted without rewriting their plans.
  V1 scalar behavior remains unchanged.
- Generated gradual-v2 evidence requires its recorded support. Validation checks
  kind/window, complete finite intervals, required descriptive fields, exact
  completed-block support endpoints, signal-clock bounds and association with
  the segment. Missing, malformed, off-clock or reassigned support is rejected.
- A non-generated v2 envelope must declare
  `envelope_provenance: {"kind": "authored", "description": "…"}`. Adding a
  `visual_policy` cannot waive generated provenance requirements. Visual styling
  leaves generated evidence and support intact.

These repairs are source/contract corrections, not adoption of Team 1's new
role-context interface. This visual pass covers the existing 130–178s passage,
not the four prior focus intervals or the proposed unassigned 119–132s study.

### Verification and reproduction

- Final focused direction/render/builder suite: **71 passed, 1 warning**.
  Full shared-worktree suite: **666 passed, 123 warnings**, 18.10s. Regressions
  include zero-effective-focus, valid fades, scalar compatibility, missing/
  malformed support, preserved visual-only schedules, replay without styling
  or analysis, same-color/same-position silhouette distinction, silent shapes
  and deterministic seeking. Frozen local 02 compatibility tests skip if those
  ignored packages are unavailable in a fresh clone.
- All **23 candidate + 23 replay** snapshot/output hashes and their origin
  fingerprints verified. Both pages derive exactly from their frozen template,
  plan and manifest; separate page-derivation checksums verified. Parent package
  hashes verified before reuse against its documented manifest.
- Both plans, both videos, signals and original PCM replay byte-for-byte.
  `previous.mp4` is byte-identical to `directed-review-02/directed.mp4`.
  An independent check against the parent renderer also matched **36 frames**
  across all three frozen v1/v2 plans, including crossfades.
- Both videos have **2,880 frames at 60 FPS, duration 48s** and identical decoded
  audio. PCM exactly matches the original source FLAC's 130–178s samples.
  Candidate package size: **22,371,914 bytes**.
- Browser checks passed native MP4 playback, both view switches while playing
  and paused, preserved seeking, shortcut positions, failed-load restoration/
  retry, feedback time/version/hash linkage, and no JavaScript errors. No
  overflow at 320/375/768px or with details expanded. The mobile screenshot
  was visually inspected. Test feedback was intercepted in memory and not
  saved as user feedback. Temporary browser script/screenshots and sample sheet:
  `/tmp/songviz-visual-browser.cjs`, `/tmp/songviz-visual-mobile.png`,
  `/tmp/songviz-visual-desktop.png`, `/tmp/songviz-visual-refined.png`.
- `git diff --check` passed. Source changes stayed in Team 2 paths; concurrent
  Team 1 changes to `CONTINUE.md` and its own files were preserved.

Candidate manifest SHA-256:
`95c47b36345d25aa66bb16adfb1a0a7a7289e2998e2851114b05c28d6fa7cc60`.
Replay manifest SHA-256:
`ffb3c99452a2d9b92f397a33ffd15478f05e7fefdfa57d14cded91609836f611`.

Commands used (the completed directories now refuse overwrite):

```bash
.songviz/venv/bin/python experiments/build_directed_review.py \
  --visual-from outputs/reviews/directed-review-02 \
  --out outputs/reviews/directed-visual-02 --fps 60
.songviz/venv/bin/python experiments/build_directed_review.py \
  --replay outputs/reviews/directed-visual-02 \
  --out outputs/reviews/directed-visual-replay-02 --fps 60
.songviz/venv/bin/python -m pytest -q --disable-warnings
```

One Terra worker implemented the initial visual treatments/anchors and tests;
another handled the bounded contract repair. The lead built the two-view
package/UI, reviewed actual changes, refined appearance and support validation,
and inspected/verified the rendered artifacts. Next: Team 1 reviews this combined
handoff and updates its checkpoint; the prepared visual question is whether
voice/snare/kick are now distinguishable in the 168s overlap. Artistic acceptance
and any subsequent role-context direction study remain separate decisions.

## Team 2 comparison design — 2026-09-11

The next bounded direction study uses the same 130–178s passage and frozen
`directed-review-01` signals/audio. Compare three saved plans: gradual emphasis,
the earlier coarse direction, and the fixed always-on view. The earlier
package's 24 recorded source, snapshot and output hashes verified before reuse;
its manifest matches the SHA-256 recorded below. The source package remains a
control. This study does not depend on Team 1's role-context experiment.

The concrete question is whether evolving visual emphasis within the long
137.917–165.735s sparse-percussion span gives a more useful result than holding
the earlier focus constant. Candidate envelopes use each stem's own normalized
activity, keep a stable allocation for other/vocals, and retain the coarse
boundaries and warm/cool visual motif vocabulary. The comparison tests these
direction choices together; it cannot isolate one parameter's artistic effect.
Returning visual motifs mean reuse of a design associated with percussion
activity, not verified recognition of a musical identity.

Required technical acceptance: inspectable and validated saved curves; exact
signal/audio binding; selective omission; continuous curve evaluation and
deterministic seeking; unchanged v1 rendering; a three-view replay without
planning or stem analysis; matching video durations/audio; native video playback,
same-position switching, retry and mobile layout. Artistic acceptance remains
pending even when these checks pass. Existing four-example feedback is retained
as design context and supplies no new acceptance labels for this passage.

The inherited rhythm/percussion evidence predates the later structure-grid
experiments. This experiment holds it fixed for comparability. It cannot establish
correct beat phase, vocal leadership, laughter recognition, salience or full-song
direction. The observed vocal activity rise motivates preparing a reviewable
visual candidate; normalized stem values are not comparable absolute loudness.

Team 2 owns this detailed experiment record. Per the collaboration protocol,
Team 1 must review the completed evidence and update `CONTINUE.md`; Team 2
does not edit that shared checkpoint.

### Completed Team 2 result

Selected artifact: [directed-review-02](http://127.0.0.1:8770/directed-review-02/),
at `outputs/reviews/directed-review-02/` (22,113,425 bytes). Reproduction by
saved-plan replay is at `outputs/reviews/directed-replay-02/` (22,096,909 bytes).
Both directories are new; prior packages and source audio were preserved.

The page offers **Gradual attention**, **Earlier direction**, and **Steady
comparison** at the same playback position. The saved candidate uses schema v2:
every layer has absolute-time gain/emphasis keyframes with exact segment
coverage. There are 3,395 layer keyframes across the three retained spans.
Envelope gain changes intensity; emphasis changes geometry size and supporting
layer position/intensity. Outgoing/incoming scenes retain independent crossfades.
The fixed and earlier controls keep their schema-v1 rendering unchanged.

Within sparse percussion, the allocation keeps other/vocals traces, omits main
percussion and bass in this selected plan, and preserves a quiet pulse. Each
stem's completed normalized activity samples receive a finite 1.5s trailing
mean. Candidate vocal gain is `0.04 + 0.90 × activity`, vocal emphasis is
`0.08 + 0.84 × activity`, and other gain is `0.38 + 0.44 × activity`.
These are explicit development choices. Other remains the coarse central focus;
vocals gradually become a larger, more visible supporting trace. Sampled values:

| Song time | Vocal gain | Vocal emphasis | Other gain |
| --- | ---: | ---: | ---: |
| 142s | .090 | .127 | .589 |
| 150s | .311 | .333 | .696 |
| 158s | .272 | .296 | .783 |
| 162s | .630 | .631 | .760 |
| 165s | .574 | .578 | .758 |

Raw activity still modulates the drawn shape, so envelope values alone do not
measure pixel brightness or perceived attention. Linear interpolation between
saved knots uses the next knot, potentially one 100ms block ahead. This is
offline planning with whole-song p95 normalization. Knot construction uses
completed samples; the saved support metadata includes contributing block
starts, and musical leadership/salience/recurrence remain explicitly unknown.

Lead visual inspection of the evidence chart and candidate/control frames at
150s and 162s found the added violet vocal trace visible but fairly faint;
it becomes stronger at 162s. This demonstrates a modest visual difference,
not musical acceptance. The two ribbons use synthetic wave geometry, not vocal
or accompaniment pitch traces. The study retains the older timing evidence and
the coarse central focus, so it does not resolve the later structural timing
regressions or semantic role recognition. No thresholded change episodes or
Team 1 role-context outputs are consumed by this version.

### Validation performed on the completed package

- `.songviz/venv/bin/python -m pytest -q tests/test_direction.py
  tests/test_directed_render.py tests/test_directed_builder.py --disable-warnings`
  passed **48 tests, 1 warning**. Tests cover valid/invalid curves, evidence-ID
  ordering, hidden layers, geometry continuity, outgoing-scene diagnostics,
  tampered inputs, and a three-way build/replay that forbids replanning.
  The final run followed lead fixes to block support and scene weights.
- All **29** candidate and **26** replay snapshot/output fingerprints verified;
  both HTML pages exactly regenerate from their frozen templates, plans and
  manifest hashes. Their separate `page-derivation.json` records verify the page
  without a manifest self-hash cycle. Live implementation files match the
  candidate snapshots. The parent’s **24** recorded source/snapshot/output
  hashes also verified before reuse.
- All three plans, signals, original PCM and three MP4s reproduce byte-for-byte
  through `--replay`. The new `coarse.mp4` and `fixed.mp4` are byte-identical to
  the frozen `directed-review-01` control videos. An independent comparison
  against its snapshotted renderer also matched **34 frames** across both v1
  plans, including boundaries and crossfades.
- Every video has **2,880 frames, 60 FPS and 48s duration**. Decoded A/B/C audio
  is identical; PCM exactly equals the source FLAC's 130–178s samples. AAC/source
  zero-offset correlation is **0.9972178114**. Frame quantization and activity
  block timing are separate from this audio alignment result.
- Headless Chromium passed native MP4 playback, paused/playing same-position
  switching among all three views, failed-load restoration/retry, seeking,
  original-song timestamp conversion, empty-feedback rejection, and manifest
  linkage. Synthetic feedback was intercepted in memory, not saved as user
  feedback. The 375px mobile layout had no horizontal overflow or JavaScript
  errors and was visually inspected. Diagnostic screenshots are temporary:
  `/tmp/songviz-team2-mobile.png`, `/tmp/songviz-team2-desktop.png`, and
  `/tmp/songviz-team2-contact.png`.
- `git diff --check` passed for Team 2 paths. A whole-repository suite was not
  rerun during the concurrent Team 1 work; the test count above is scoped.
  The environment emitted Datadog/ddtrace startup/telemetry warnings, including
  a denied telemetry temporary file; project checks and rendering exited
  successfully. Local server and Chromium startup needed sandbox approval.

Candidate manifest SHA-256:
`0afb97bdc294827ecde1429f32da4f61f1ea9f300dde4724f43b27d864ef7e90`.
Replay manifest SHA-256:
`5d044824960cf01d03793b2cad59a1296ceaecc2f7e8f036030eaf6bbd41f8d7`.

Commands used (each output path must be new; these two now refuse overwrite):

```bash
.songviz/venv/bin/python experiments/build_directed_review.py \
  --gradual-from outputs/reviews/directed-review-01 \
  --out outputs/reviews/directed-review-02 --fps 60
.songviz/venv/bin/python experiments/build_directed_review.py \
  --replay outputs/reviews/directed-review-02 \
  --out outputs/reviews/directed-replay-02 --fps 60
```

Two Terra workers implemented the bounded direction/render and package/UI
scopes. Lead review inspected actual code and artifacts, corrected three-way
replay, diagnostic presentation, v1 compatibility, evidence support and scene
weights, then ran the checks above. No claim of measured cost savings is made.

### Team 1 integration request

```text
REPLY
status: complete
changed: Team 2 direction/render, builder/template, their tests and doc 10;
         new directed-review-02 and directed-replay-02 artifacts only
result: Inspectable three-way gradual-emphasis passage with exact saved-plan replay.
evidence: 48 focused tests; 55 package snapshot/output hashes; identical replay
          videos/audio; frozen control MP4 equality; browser checks above.
limitations: Modest visual change; development activity proxies and inherited
             timing; artistic acceptance, musical roles and salience unresolved.
next: Team 1 reviews these artifacts/diffs and updates CONTINUE.md. Then review
      whether the vocal trace gaining emphasis around 158–165s helps attention.
```

No new annotations of the earlier four listening examples are required. The
optional page note concerns this prepared visual comparison. Team 2 does not
promote the candidate or modify the integration owner's checkpoint.

This is a local, deterministic rule-based prototype, not an LLM-backed director
or a claim of full-song musical understanding. It implements the first saved-plan
path from the revised roadmap. Artistic acceptance remains pending.

## Run it

```bash
.songviz/venv/bin/python experiments/build_directed_review.py --out outputs/reviews/directed-next
```

The command verifies the selected `rhythm-review-01` package, reads existing
source stems, measures energy, generates and validates a direction plan, then
renders the directed and fixed-comparison videos. Defaults are the local Feel
Good Inc file and 130–178s. `--start` and `--end` can select another span covered
by those inputs; this is not yet a general audio-file CLI. No source separation,
beat fitting or note extraction is rerun. No runtime model calls or uploads occur.

Replay from saved evidence and audio without planning or stem analysis:

```bash
.songviz/venv/bin/python experiments/build_directed_review.py --replay outputs/reviews/directed-review-01 --out outputs/reviews/directed-replay-next
```

To try an edited plan, copy it to a separate file, then pass `--plan path/to/edited-plan.json`
with `--replay`. The edit must retain the excerpt interval and signal hash; invalid
plans fail before rendering. Neither operation overwrites the original package.
Replay uses the current renderer code; source snapshots identify the version used
for a prior artifact. Matching code/configuration is needed for matching pixels.

## What was implemented

- `songviz/direction.py`: JSON-compatible version-1 plans, signal/plan validation,
  automatic activity-gap/energy policy, and a fixed always-on comparator.
- `songviz/directed_render.py`: generic ring, ribbon, ticks and rails treatments
  usable by any layer. Plans control visibility, focus, treatment, gain, palette,
  motif identity and scene crossfades. Primary focus occupies the center; support
  geometry is smaller. Event responses and completed-block energy are causal.
- `experiments/build_directed_review.py`: input verification, energy measurement,
  planning/rendering/replay, source snapshots, provenance, evidence image and page.
- `experiments/templates/directed_review.html`: same-position comparison, an
  expandable plan/evidence view, timestamped feedback, and local/HTTP playback.

The plan contains absolute-time contiguous segments, visible/hidden layers,
bounded gains, supported treatment/palette IDs, focus and motif IDs, transition
durations, reasons and evidence references. It is bound to the exact signals by
a content hash. Validation rejects gaps/overlaps, unavailable layer names,
unsupported treatments, invalid focus/gains, unknown evidence references, altered
signal content and times outside coverage. It checks the contract, not whether a
reason is musically persuasive. Reasons and summaries remain inspectable.

Terra implemented the renderer and renderer tests. Luna implemented the review
page and builder tests. The lead implemented the policy/contract, evidence and
package builder, reviewed worker code, and integrated the result. Review caught
and corrected an overlapping-layer crossfade issue, faint-signal overvisibility,
and a vacuous hidden-layer test. Delegation is recorded, not asserted to be an
optimal or measured billing configuration.

## Selected evidence and resulting plan

Main-percussion gaps are inferred from the song-wide prominent kick/snare event
stream. A gap must last at least 4s; its sparse span begins 350ms after the last
main hit and ends at the next. This tail allowance and threshold are explicit
development policy, not musical section boundaries. Sparse spans need not be
silent: hi-hat and other layers can continue.

Bass, vocals and other use stereo-preserving RMS over completed 100ms blocks,
normalized by each stem's whole-song p95 and capped at one. These are relative
activity measurements, not cross-stem absolute loudness, note events or confidence.
The renderer uses sample-and-hold; energy changes may lag by up to 100ms.

| Song interval | Focus | Policy evidence / choice |
| --- | --- | --- |
| 130–137.917s | Snare | Prominent percussion present; bass is supporting; other/vocals omitted |
| 137.917–165.735s | Other stem | No prominent kick/snare; other has highest mean normalized energy; percussion hidden despite some hi-hat evidence |
| 165.735–178s | Vocals | Percussion returns and mean vocal activity exceeds the 0.65 threshold; vocals lead, percussion supports, bass/other hidden |

Warm `percussion-groove` motif identity is reused before and after the gap; the
sparse texture gets a cool ribbon treatment. This is recurrence of activity and
visual vocabulary, not verified melodic/harmonic motif matching. A fixed comparator
uses the same source audio, timing, signals and treatment vocabulary, but holds
focus/visibility/gains/palette constant. This isolates direction choices as a
bundle; it does not isolate one individual palette or gain parameter.

The policy is implemented from input data; these times are its output, not
hand-authored boundaries. However, the song, excerpt and thresholds are development
choices informed by prior reviews. This is not a holdout result. No cached section
role was assumed correct, and no musical surprise detector is claimed.

## Review and limitations

Selected package: `outputs/reviews/directed-review-01/index.html`. Serve locally:

```bash
python3 -m http.server 8768 --bind 127.0.0.1 --directory outputs/reviews/directed-review-01
```

Open `http://127.0.0.1:8768/`. Compare **Directed sequence** and **Always-on comparison**.
Ask whether attention follows the music and what should be emphasized or omitted.
Feedback exports as `songviz-direction-feedback.json` with original-song timestamps
and a manifest hash. Notes are not saved until downloaded; chat feedback also works.

The evidence plot shows vocal activity rising within the long sparse span before
percussion returns. This coarse planner may therefore foreground vocals later
than desired. It currently summarizes each gap/non-gap span rather than detecting
all phrase-level entrances. Treat that as a concrete open review question, not
proof that the three-state direction tells the whole story. The "other" stem is
not a confidently identified instrument, and separation can contain bleed.

Further limitations: no LLM integration, no automatic full-song style development,
no reliable surprise/semantic-role detection, no independent generalization test,
and a deliberately small geometric vocabulary. An edited human plan is supported
for exploration but cannot count as automatic planner output. Production render
commands and cached analysis remain unchanged. Milestone 2 needs user review;
technical success alone does not complete it.

## Verification of the selected package

- 127 focused tests passed across the new directing modules and prior rhythm,
  percussion, review, benchmark and evaluation paths. Tests cover contract
  rejection, data-derived boundaries, clock translation, hidden-layer behavior,
  treatment changes, independent scene crossfades and source-input isolation.
- Both comparison videos contain 2,880 frames at 60 FPS and last 48 seconds.
  The PCM cut exactly matches the original source interval; decoded A/B audio is
  identical. Zero-offset AAC/source correlation is 0.997218. Visual event sampling
  still has up to 16.67ms quantization, separate from the 100ms energy block lag.
- The full `--replay` path generated `directed-replay-01` without planning/stem
  analysis. Plan, signals, PCM, directed MP4 and fixed MP4 all matched the selected
  package byte-for-byte under the same code/configuration.
- Lead image inspection covered the measured activity plot and rendered samples
  from all three focus spans. Browser checks covered duration, same-position
  paused/playing switching, retry after a simulated fetch failure, timestamp
  conversion (local 15s → song 145s), bounds/empty feedback checks and manifest
  linkage. Test exports were intercepted, not saved as user feedback.
- Input snapshots/output checksums verified. Selected manifest SHA-256:
  `320a058b12a358acf505afc714117169c00f2637802ad5829ae367c50c13f040`.
