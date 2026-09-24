# SongViz — start/resume here

This is the **single entry point for a new agent or compacted conversation**.
Updated 2026-09-24 after the layered-reference and blind-listening handoff.
Read this file, then the linked document for the task at hand. Existing changes
belong to the user; the working tree contains substantial modified/untracked work.

## Current checkpoint

- **Attempt 03 stopped before R9; no song-quality result.** The human authorized
  one bounded CPU-only All-In-One v1.1.0 `harmonix-all` attempt and Opus review.
  R0–R8 passed: the 75-package CPU environment and pinned sources were audited;
  Madmom built once; CPU imports and direct NATTEN arithmetic passed; the
  full-song synthetic model pass took 81s at 1.37 GiB peak RSS; and the official
  eight-fold loader completed a 272.4s synthetic forward in 646.4s at 2.01 GiB,
  with finite expected outputs. These results establish infrastructure only.
  R9 preflight later failed both fixed ext4 limits: at 07:34:15Z free space was
  2,425,147,392 bytes, 151,832,986 below the 2.4 GiB floor, and writes since R0
  were 261,357,568 bytes, 156,499,968 above the 100 MiB cap. R9 never started;
  its official Demucs checkpoint was not downloaded, no study audio was read,
  and R10–R12 were not run. Do not resume this attempt even if disk space is
  restored. There is no All-In-One musical-quality result. Opus judges process
  and reference-product progress real, but detector quality unproven; the older
  SSM baseline's reproducible positive misses reject that baseline only, while
  its empty negative is weak evidence because it emits sparsely. Detailed stop
  evidence and R9 script checks: [R9 preflight](outputs/reviews/all-in-one-feasibility-03/r9-preflight.json).
  Preserve both earlier environments, the frozen protocol, model cache and raw
  references. A new model/runtime attempt requires a fresh preflight and
  explicit authorization; consider prioritizing the within-section event
  references without claiming that this candidate failed musically. See the
  [attempt 03 protocol](outputs/reviews/all-in-one-feasibility-03/attempt-protocol.md),
  [earlier Opus review](outputs/reviews/all-in-one-feasibility-03/opus-review.md),
  and `outputs/reviews/all-in-one-feasibility-03/`.
- **Second All-In-One setup attempt ended at the first GPU convolution gate.**
  CUDA discovery and matrix multiplication passed, but synthetic Conv2d failed:
  cuDNN could not load `libnvrtc.so`. The exact Torch wheel metadata does not
  declare an NVRTC package; the NVIDIA-owned CUDA 11.8 wheel was independently
  hash-checked and contains only versioned NVRTC libraries, so it cannot satisfy
  the gate without a loader-path or symlink change. Opus agrees: infrastructure
  no-go. No NATTEN/model dependencies, checkpoint or song audio were installed
  or run; there is no model-quality result. This used the one extra authorized
  setup attempt; do not start another model/runtime search. Preserve both
  environments. The current `.songviz/allin1-venv-02/` contains only the Torch
  stage and occupies 4.2 GiB. Detailed evidence:
  [doc 33](docs/33_change_decision_screen.md#second-all-in-one-attempt--infrastructure-no-go)
  and `outputs/reviews/all-in-one-feasibility-02/`.
- **Two separate visual levels chosen and prototyped; targeted review still
  fails its frozen coverage rule.** The human chose broad sections and fine
  events as separate layers. Current page:
  [structure-review-06](http://127.0.0.1:8770/structure-review-06/). It retains
  all 19 certainty-unspecified section marks, four source-bound raw notes and
  13 section-timing comparisons, plus a distinct lane for the four exact
  listener review windows (3 change-heard, 1 none-control). The lane states that
  ranges are not event boundaries/durations and the rest is unreviewed. No
  acoustic event prediction or model inference is shown. Version 05 is
  preserved but superseded; -06 removes its candidate/window comparison and
  softens its window ends. **13 focused tests passed**; Chromium rendered all
  19 + 4 + 13 cards and 4 separate windows, loaded the original audio and played
  the 16–26s no-change control. All output hashes passed; audio/plots match the
  parent. The 13 diagnostic contexts plus four feedback windows still cover
  **172.97/221.17s (78.2%)**, so this page is a full-song reference, not a
  targeted sign-off task. No full-track reannotation or sign-off was requested.
  The 18-vs-5 section-boundary comparison and limits are in [doc 33](docs/33_change_decision_screen.md#completed-structure-review-overlay--targeted-review-objective-fails); the layer design and -06 validation follow.
  The rough “79s they enter” drum-entry reply has been preserved, but is treated
  as potentially exposed because the linked section page showed a candidate
  boundary at 79.087s; do not use it as independent timing evidence. A blind,
  source-only excerpt is ready at
  [event-reference-listening-01](http://127.0.0.1:8770/event-reference-listening-01/)
  but no further listener reply is needed now. Opus review: product/process
  progress is real, but no model-quality improvement has been demonstrated; the
  listening loop is not the bottleneck. The existing stem-free section method
  was screened on the two approximate positive references and the Feel Good Inc
  no-change control. Canonical run 02 in `.songviz/venv` misses both positives
  and leaves the control empty; saved predictions and reports match run 01
  exactly across environments. Treat this as a no-go for the current baseline,
  not all section models. One additional CPU attempt was authorized, but attempt
  03 stopped before R9 on the hard ext4 limits; this is an incomplete
  infrastructure screen, not a model-quality failure. No further listening
  note is needed. A new model/runtime attempt requires fresh preflight and
  authorization. [Protocol and results](docs/33_change_decision_screen.md#opus-review-and-strategy-decision--2026-09-24).
- **Distribution-contrast screen complete; new rule rejected.** The fixed rule
  adds persistent bass-pattern evidence near Agnes's new-part reference but also
  at 21.063/21.496s in the original non-change probe. It preserves every one of
  520 frozen level-only records and both known layer additions, but fails the
  predefined non-change requirement. Do not tune its threshold or promote it.
  All six musical acceptance results remain unresolved. **6 tests passed**;
  lead independently checked 4,352 score records, 537 anchors, 77 input and
  14 output hashes; a second reviewer confirmed level parity and interpretation.
  [Doc 33](docs/33_change_decision_screen.md#completed-result-reject-the-new-rule)
  and `outputs/reviews/change-decision-01/` preserve the completed experiment.
- **Agnes cue clarification received; withdrawal diagnostic complete.** The human
  hears it getting more "quiet", thinks there are fewer instruments, and feels
  that something is coming. Preserve perceived quietness, tentative thinning and
  anticipation separately from the earlier tentative section name. Raw text and
  the fixed diagnostic are in `outputs/reviews/change-decision-agnes-cue-01/`.
  Original mix and drum RMS fall in 14/15 overlapping comparisons near the
  reference, while other-stem RMS rises in 13/15. Central mix changes are
  -0.875/-0.653/-0.847 dB at 2/4/8 beats. All separation groups remain active;
  this neither proves nor disproves fewer instruments. Anticipation stays a
  human observation. Lead verified 89 source-window pairs, 356 stem comparisons,
  77 parent input hashes and six package files. No classifier was promoted.
  [Doc 33](docs/33_change_decision_screen.md#completed-signed-level-diagnostic--2026-09-23)
  contains the comparison and limitations.
- **Castlecomer — Move judgment received; fixed-method comparison complete.**
  [Listen to the 24-second excerpt](http://127.0.0.1:8770/change-decision-listening-02/excerpt.wav).
  Package: `outputs/reviews/change-decision-listening-02/`. Alphabetical source
  selection excludes the three comparison tracks; timing uses the native-frame
  midpoint only. The human hears a change around elapsed 16–17s, tentatively
  verse to chorus (song 93.633333–94.633333s). Raw feedback and the completed
  fixed-seconds original-mix diagnostic are in
  `outputs/reviews/change-decision-castle-intake-01/`. Eight of nine comparisons
  rise; the earlier non-change passage rises in all nine. Neither level direction
  nor magnitude establishes a new section. Lead verified 63 source-window pairs,
  four source hashes and four package hashes. Missing stems/beats were then
  prepared with the cached htdemucs model and existing local analysis under a
  separate fixed protocol in `outputs/reviews/change-decision-castle-methods-01/`.
  **All three unchanged methods have zero reported change/dip candidates across
  28 fully supported anchors**, including near all 16–17s calculation guides.
  Acoustic pattern/activity evidence exists but does not pass the fixed policies.
  One unfiltered sustained anchor at elapsed0.502s is excluded because its input
  support extends before the clip. No thresholds were lowered or methods promoted.
  CPU run completed in 120.55s, two threads, networking disabled via `unshare -Urn`.
  Lead verified 24 hashes, 336 per-stem contrast and 224 activity records; Terra
  independently checked provenance/configs/support and the no-candidate result.
  Details and limitations are in [doc 33](docs/33_change_decision_screen.md#completed-input-preparation-and-unchanged-method-comparison).
  The follow-up cue question remains optional; the strategy correction above
  removes it as a dependency. The approximate judgment already supports a
  task-trained boundary comparison. Do not infer the cue from the section name
  or tune near thresholds. Preserve any later clarification without restarting
  the completed local-policy tests.
  Exact native PCM equality, source/artifact hashes, HTTP delivery/ranges and
  actual Chromium playback passed. The localhost server now runs as the
  transient user service `songviz-review-8770` after the detached process stopped.
  Use `systemctl --user status songviz-review-8770` to inspect it and
  `systemctl --user stop songviz-review-8770` to stop it. It is not enabled at boot.
- **Existing transition-component diagnostic complete; selection still unresolved.**
  The unchanged pattern policy proposes one Agnes candidate at elapsed 5.089s,
  not near the human's approximate 12s point. Spectral-change evidence exists
  there with modest level change, but the earlier non-change excerpt attracts
  five candidates and stronger pattern responses. More contrast is not verified
  musical importance. No method/threshold/production change is promoted.
  The frozen development control reproduces exactly; **8 focused tests passed**
  with one existing warning. Lead independently verified 60 per-stem records,
  seven artifact hashes and reserve filtering. [Doc 32](docs/32_transition_components.md)
  records results, initial path-resolution failure and corrected validation.
- **Arctic Monkeys comparison complete; useful arrangement evidence retained.**
  The human hears mostly the same idea, with more elements introduced around
  elapsed 17s (approximately song 141.197052s). Preserve the tentative continuity,
  unspecified instruments and approximate timing. Raw feedback and fixed design
  are in `outputs/reviews/transition-components-arctic-intake-01/`. The unchanged
  sustained-activity policy proposes an `other`-stem increase at elapsed
  **17.329s**, with stable drum/vocal background measurements. Timing/direction
  agree with the approximate note; instrument identity and musical continuity
  are not automatically proven. The control/separate/activity policies produce
  2/3/4 candidates across 18 eligible anchors, so additional proposals remain
  unscored. **5 focused tests passed**; lead verified 14 input/10 output hashes,
  20 per-stem records and reporting support. The run finished before the usage
  interruption and was not repeated. Accepted numeric package:
  `outputs/reviews/transition-components-arctic-arrangement-01/`; corrected plot:
  `outputs/reviews/transition-components-arctic-plot-02/`. [Doc 32](docs/32_transition_components.md#completed-arctic-comparison--reviewed-2026-09-23)
  records validation and limitations. No further Arctic judgment is pending.
- **Completed task: fixed distribution-contrast screen across six seen references.**
  [Doc 33](docs/33_change_decision_screen.md) preserves separate musical acceptance
  criteria and defines one acoustic rule before execution: retain level-only
  additions, compare between-passage spectral differences with within-passage
  variation, require persistence, and keep semantic fields unknown. No threshold
  tuning after the result. The rule is rejected above. No extraction,
  failed-gate rerun or large-model search is assigned.
- **Completed listening intake on another song.** A metadata-selected,
  source-identical 24s Arctic Monkeys excerpt is ready:
  [listen](http://127.0.0.1:8770/transition-components-listening-01/excerpt.wav).
  The human has been asked whether the same musical idea continues or noticeably
  changes, roughly when and how. Approximate prose is enough. Source PCM equality,
  artifact hashes and actual Chromium playback/duration passed. No predictions
  from that passage were inspected before selection/intake; existing methods are frozen. Historical
  exposure is unknown, so do not call it a certified untouched holdout. The
  answer is now preserved verbatim in the new intake above. No follow-up rule
  has been chosen from the answer.
- **Arrangement comparison complete; added continuity gate rejected.** The fixed
  level-only rule finds two overlapping drum-increase anchors in the known focus
  and zero proposals in the non-change focus. Requiring two nearly unchanged
  background layers rejects the drum increase and all 50 level-only proposals
  across 482 eligible development anchors. It adds no benefit in the non-change
  case. The lead independently verified all 4,208 per-stem numerical records,
  520 eligible anchors across both songs, and 13 result fingerprints. **6 focused
  tests passed, 1 existing warning**. No threshold tuning or production promotion.
  [Results and failure diagnosis](docs/31_arrangement_continuity.md#completed-comparison--reviewed-2026-09-21);
  [comparison report](outputs/reviews/arrangement-continuity-01/report.md).
  A usage limit interrupted the handoff; the completed experiment was not rerun
  after the human requested continuation.
- **Reserved-passage judgment received; timing match remains unresolved.** The human
  reports a transition to a new part around elapsed 12s, approximately song
  105.294263s, tentatively verse to pre-chorus. The frozen level-only arm has no
  proposal at the nearest anchor; its only reserve proposal is at elapsed 10.336s.
  That proposal compares source windows spanning elapsed 6.830–14.284s, which
  include the approximate human time. These are feature support, not detected
  event extent. Without a timing tolerance or an exact reference, neither a
  definite miss nor a hit is established. The earlier handoff overstated this.
  The continuity arm has none. Raw feedback is preserved in
  `outputs/reviews/arrangement-continuity-listening-01/human-feedback.json` and
  interpreted in [doc 31](docs/31_arrangement_continuity.md#human-reserve-judgment-received--2026-09-22).
  This is approximate listening evidence, not exact onset or section ground truth.
  Do not tune thresholds on one passage.
- **Completed task: inspect the existing separate pattern and level evidence.**
  Reused the verified caches and unchanged local-structure control and
  separate-channel policy on the four development passages and now-seen Agnes
  passage. This diagnosed whether existing pattern evidence helps before adding
  another detector. No threshold tuning, audio extraction, model run or production
  promotion. Design and results: [doc 32](docs/32_transition_components.md).
  The next listening response has been received above. Do not reuse Agnes as an untouched
  holdout, rerun the failed continuity gate or reopen laughter recognition.
- **Priority corrected: reusable musical developments across passages.** The
  human challenged laughter recognition as too specific to this song. Team 1
  agrees: that sound category is an optional diagnostic, not a prerequisite for
  general song analysis or progress. The bounded question is whether existing
  stem and recurrence evidence can distinguish an arrangement change within
  continuing material from ordinary variation, evaluated across contrasting
  passages. The first fixed comparison is now complete above. Keep the benchmark
  and prospective-reserve distinctions; do not replace one song-specific target
  with another isolated showcase.
  Musical role/importance remains a separate unresolved capability. No new
  annotation UI or model search is assigned by this correction.
- **Vocal-behavior reference page complete and parked; no human action required.**
  The human requested continuation with Terra after the usage limit reset. Terra
  implemented the page; Team 1 accepted
  [vocal-behavior-reference-03](http://127.0.0.1:8770/vocal-behavior-reference-03/).
  It starts blank and plays the exact original 119–132s excerpt, with overlapping
  labels, timing uncertainty, local drafts and JSON export. **3 focused tests**
  passed; lead verified native PCM/source parity and browser playback, exact
  interval stop, export/restore, failure recovery and mobile layout. Drafts 01/02
  are preserved and superseded. Evidence:
  [doc 30](docs/30_vocal_event_probe.md#temporal-reference-page-accepted-human-labels-pending).
  No human labels or model runs were created. The later priority correction parks
  this task: do not ask the human to complete it before continuing the project.
  Preserve the page and experiment as optional diagnostics; if feedback is later
  volunteered, preserve it without substituting model scores or old focus bands.
- **Local vocal-event screen complete; negative result.** Team 1 ran the fixed
  eight-condition YAMNet experiment once: 196 windows, 2.04s, 150.4 MiB peak RSS.
  Original and half-gain mixes failed the behavior criterion. Both vocal stems
  detected early speech, but their maximum laughter scores stayed below 0.5
  throughout the excerpt (Demucs 0.1484; RoFormer 0.1992). Passing the instrumental
  control does not establish specificity when positive inputs fail. No detector
  or directing policy is promoted. [Result and validation](docs/30_vocal_event_probe.md#eight-condition-screen-complete--2026-09-16);
  [saved-score report and plots](outputs/reviews/vocal-event-analysis-01/report.md).
  Focused tests: **9 passed, 1 existing warning**; lead verified all 20 probe and
  seven analysis fingerprints and independently reproduced all eight input arrays
  and per-window RMS. Source audio stayed local; no paid service was used.
- **Vocal-event branch deferred; existing evidence preserved.** Existing focus
  bands are not event annotations; retain the four musical judgments unchanged.
  PANNs remains deferred after
  read-only source review: nominal 10ms output rows repeat broader segment
  scores, and its recognition advantage is untested. This is a task-fit decision,
  not a finding that PANNs fails. No second model, full-song extraction or repeated
  YAMNet run is queued. Concrete follow-up boundary in
  [doc 30](docs/30_vocal_event_probe.md#follow-up-decision-defer-a-second-model).
- **Current assignment: Team 1, one lead with bounded Terra/Luna workers.**
  The human authorized consolidating the proposed three-lead split and explicitly
  selected Team 1. The combined evidence inventory/development benchmark is now
  accepted; reserve another strong model for a focused experiment review when
  a concrete proposal exists. Team 2/3 are idle. Design and completed evidence:
  [doc 29](docs/29_development_benchmark.md). The historical results below are
  evidence, not a queue of runtime setup or review tasks to repeat.
- **Development benchmark complete; subsequent method screen complete in doc 30.**
  `outputs/reviews/development-benchmark-02/` retains four exact raw judgments,
  60 acoustic observations, eight intersecting section-span records, separate
  manual pass/fail/unresolved criteria and a blank output schema with neutral IDs.
  Terra's focused tests: **4 passed, 1 existing warning**. Lead verified source/
  output hashes and source parity and replayed all five files byte-for-byte.
  Manifest: `684b38314825c61c743459337631085abee87e12bd8157aab2b6fef952ff4f97`.
  Draft 01 is preserved and rejected after rubric/provenance review. This is an
  unscored development reference, not improved automatic understanding. The
  subsequent two-method comparison selected the now-completed local screen above.
  Temporal vocal labels were a proposed follow-up, now parked rather than required.
  No detector promotion or paid/remote run is assigned.
- **Team 1 CPU synthetic screen complete; subsequent loader/forward closed.** The
  user authorized an isolated CPU-only environment and no-input synthetic tests.
  Binary Python-3.10 wheels installed cleanly; an offline harness verified the
  exact Music Flamingo imports and expected decoder-Linear replacement/exclusions.
  A separate largest-layer NF4 case (`18944 × 3584`) completed in 4.679s at
  1.687 GiB peak RSS, below but close to its 2-GiB guard. This establishes only
  synthetic compatibility, not full checkpoint loading, song latency, or semantic
  validity. No model tensor payload, song/stem input, driver/system change,
  remote service or paid resource was used. Evidence and exact hashes:
  [doc 24](docs/24_music_understanding.md#team-1-cpu-synthetic-compatibility-screen--2026-09-14)
  and `outputs/reviews/music-representation-cpu-smoke-01/`.
- **CPU full-loader smoke complete, with a residency limit.** The pinned public
  16,534,531,504-byte checkpoint is cached and SHA-256 verified in Team 1's
  dedicated cache. Once the host met the 12-GiB-available/1-GiB-free-swap gate,
  an offline CPU-only subprocess loaded all 830 BF16 CPU parameter tensors (zero
  meta, 8,267,215,360 parameters) in 3.823s wall at 417,705,984-byte peak RSS.
  No audio/prompt/model forward ran. Safetensors lazy residency explains why this
  does not establish a resident full model or feasible audio forward. Evidence:
  `outputs/reviews/music-representation-full-loader-01/` and
  [doc 24](docs/24_music_understanding.md#team-1-full-loader-smoke--2026-09-14).
- **Team 1 synthetic first-forward closes the BF16 CPU route for now.** Terra's
  bounded all-synthetic probe completed the audio tower and projector but hit its
  external 8.5-GiB RSS guard during language-model execution: SIGKILL at
  136.314111s / 9,128,382,464 bytes, with no outer-forward output, logits,
  prompt, song audio or semantic claim. The lead independently checked the
  package/result invariants. This is a negative feasibility result for this
  pinned CPU BF16 condition, not a model-understanding result or a license to
  raise the guard/retry. Evidence:
  `outputs/reviews/music-representation-first-forward-01/` and
  [doc 24](docs/24_music_understanding.md#team-1-synthetic-first-forward-feasibility--2026-09-14).
- **Team 1's private Modal endpoint feasibility gate is complete.** Private
  health passes authenticated HTTP 200 and rejects no-credential access. Earlier
  L4 probes found then corrected the missing `librosa` dependency, then exposed
  Modal's documented 303 long-request result handoff. A locally tested client
  now allows a captured handoff to be fetched only with same-origin GET, never a
  second POST. The final explicitly authorized synthetic run had private health
  HTTP 200 followed by exactly one 1-second-silence L4 POST, which returned a
  terminal HTTP 200 (350 bytes; response hash retained, no text). Logs record
  all 830 shards loaded and 117.6 seconds of execution. No source/user audio
  was uploaded. The app, containers and proxy tokens are all stopped/empty.
  Focused acceptance is **19 passed, 1 existing ddtrace warning** plus compile
  and diff checks. This run cost `$0.03538923`; observed diagnostics total
  `$0.12277228`. Evidence: `outputs/reviews/music-flamingo-modal-result-05/`;
  prior inconclusive handoff remains in `music-flamingo-modal-l4-librosa-04/`.
  This proves only endpoint feasibility, not useful music understanding.
- **Returned handoffs reviewed.** Team 1 accepts `evidence-timeline-10` as the
  frozen baseline interface, with two minor wording/status caveats, and accepts
  Team 2's corrected runtime audit. Team 2's repair assignment is complete.
  **Team 3's design revision 3 is accepted; R1–R3 are closed.** All pending
  counterpart reviews/repairs are complete. Historical closure and execution gates:
  [doc 24](docs/24_music_understanding.md#team-1-design-closure--2026-09-14).
  Do not repeat completed assignments or launch another design-repair round.
- **Historical semantic-design acceptance; execution subsequently completed.**
  Doc 26's local no-run conclusion
  is accepted with a qualification: the local GPU is an RTX 4050 Laptop GPU;
  driver repair alone does not establish capacity for the 15.40 GiB BF16 weights.
  Accepted doc 27 revision 3 has a fixed eight-clip minimum, Stage-A-only full
  credit, assertion polarity/uncertainty rules and validated candidate times.
  SHA-256: `a0d7bb953cdeccad95457dd2a5af480ec25d2d98f6150e232be8a9a7b31453c9`.
  The original manipulation check failed; the replacement passed, preregistration
  completed, and corrected execution refuted this model/protocol (records below).
  These are no longer pending gates. No further execution is assigned.
  Lead review plus a Terra trace covered all 11 new worked cases. Nonblocking:
  doc 27's historical "22 scenarios" claim differs from its 27 listed rows;
  Team 1 does not adopt that coverage claim. Details in doc 24.
- **Original semantic preregistration retained; invalid control superseded.**
  `outputs/reviews/semantic-probe-01/` contains the exact eight required local
  WAV clips, their frame bounds and SHA-256 hashes, fingerprints for the
  original mix/four stems, the fixed Stage-A/B prompts, seeded order mapping,
  scoring template and blank manipulation-check template. Lead verification
  checked all eight clips byte-hash/format and sample equality against their
  declared mix or stem-sum recipes. Focused builder tests are **6 passed, 1
  existing ddtrace warning**; compilation and `git diff --check` passed. No
  audio upload, network/model call, model text, new label or source mutation
  occurred. The recorded blind listener check is now complete: the listener
  heard laughter in both the resum and no-vocals clips (roughly 3s and 7s
  respectively). That makes the required vocal-ablation control **invalid**
  under doc 27, so this protocol cannot reach SUPPORT. Do not upload clips or
  run a paid semantic probe under this control. The replacement and amended
  execution are complete below. See [doc 24](docs/24_music_understanding.md).
- **Team 1's local replacement vocal-ablation control has passed its blind
  listener check.**
  `outputs/reviews/stem-separation-control-01/` binds the same 119–132s
  source excerpt to Kimberley Jensen's Mel-Band RoFormer two-stem model
  (checkpoint SHA-256 `87201f…7559e`, author-published MIT record). Its CPU
  13-second run completed in 22s (2.48 GiB peak RSS) and produced a new
  `other`/instrumental candidate plus the paired vocals output, both format and
  frame-count verified. The diagnostic comparison page includes the original,
  known-invalid Demucs no-vocals result, and both new outputs. A separately
  randomized, neutral-name check recorded `candidate-other=no` and
  `resum-positive=yes`, satisfying the declared polarity rule. This validates
  only the replacement listener control: it is not a semantic result and does
  not authorize an upload or a full-song rerun. Detailed evidence:
  [doc 24](docs/24_music_understanding.md#independent-vocal-ablation-candidate--2026-09-15).
- **Team 1 has built and verified the local `semantic-probe-02` package.**
  `outputs/reviews/semantic-probe-02/` is a fresh, eight-clip PCM-24 package
  governed by [the frozen amendment](docs/28_semantic_probe_02_amendment.md).
  It preserves the seven unaffected clip samples exactly from semantic-probe-01
  and substitutes only `verse-ending.novocals` with the passed Mel-Band
  RoFormer candidate; its resulting package WAV is sample-identical to that
  candidate and separately hash-bound. The package fingerprints the exact
  completed listener-control record, replacement audio, original/stem inputs,
  prompts, seed/order and scoring template. Lead validation checked all eight
  media hashes/formats, seven antecedent sample equalities and replacement
  sample equality; focused builder tests are **8 passed, 1 existing ddtrace
  warning**, plus compile/diff checks. At creation this was a local preregistration;
  the subsequent authorized executions are recorded below. See [doc
  24](docs/24_music_understanding.md#semantic-probe-02-local-preregistration--2026-09-15).
- **Music Flamingo semantic-probe-02 is an observed, protocol-qualified
  execution—not a clean preregistered refutation.** The authorized eight-clip
  run made 32 analysis POSTs; both selected positive-control Stage-B replies
  say “No change,” and the generic/hallucinated output is unsuitable for
  SongViz decisions. But it used 256 rather than the frozen 400 max tokens and
  repeated the unchanged Stage-A prompt on retry; every initial/retry pair is
  byte-identical. Its exact-protocol conclusion is therefore
  **UNRESOLVED(protocol-fidelity)**. The app is stopped (zero tasks), its
  ephemeral Proxy Token was deleted, and immediate observed cost was
  **$0.08827776**, below the $1 cap. Raw historical evidence remains in
  `outputs/reviews/music-flamingo-semantic-01/`.
- **The local no-model execution-contract audit is complete.** Production
  source now requires the frozen 400-token budget; retry Stage A adds the
  frozen timestamp suffix exactly once; Stage B binds the exact Stage-A prompt
  and response; and future `run.json` records the frozen package plus local
  runner/app/schema hashes and effective decoding settings. Offline focused
  checks passed **32 tests, 1 existing ddtrace warning**, plus compilation and
  diff checks. The record explicitly cannot recover or prove the historical
  remote deployment bytes, so it does not change the above qualification.
  Evidence: `outputs/reviews/music-flamingo-execution-contract-01/` and
  [doc 24](docs/24_music_understanding.md#execution-contract-audit-complete--2026-09-15).
- **One corrected semantic-probe-02 execution is complete and stops this
  model/protocol on this song.** The human explicitly authorized a fresh
  eight-clip run under a $1 cap. It used the recorded 400-token contract and
  exact one-suffix retries: all eight initial pairs were retry-eligible, all
  eight permitted retry pairs were distinct and retained, for 32 POSTs. The
  selected Stage-B answers explicitly say “No change” for both
  `drum-entry.core` and `transition-extent.core`; under frozen G3 that is
  **REFUTE(positive-control)**. Immediate cost for the corrected app was
  `$0.15133208`; the same-day report across all SongViz Modal apps was
  `$0.40735652`, below the cap. The app is stopped and proxy-token list empty.
  Evidence: `outputs/reviews/music-flamingo-semantic-02/` and
  [doc 24](docs/24_music_understanding.md#corrected-semantic-probe-02-execution--refuted-2026-09-15).
- **Coordination decision:** Team 1 is the active lead and may directly use
  Terra/Luna workers for implementation, evidence inventory and checks. Team 2
  is not a delegation gateway. A second Astra or the user's Opus account may
  later challenge a concrete experiment; neither has a standing task. See the
  [multi-team protocol](docs/22_collaboration_protocol.md#delegation-is-available-to-every-team)
  and current assignment below. Model preferences do not launch other accounts.
- **Current priority: understand the song before improving artistic visualization.**
  The user explicitly assigned this account Team 1 and requested a work split.
  Text explanations, simple plots and synchronized metrics are the immediate
  review outputs. The steady/reduced preference is deferred and no longer blocks
  analysis. Detailed experiment design: [doc 24](docs/24_music_understanding.md).
- **Team 1 MuQ deliverable complete.** Real full-song base/shifted CPU inference,
  source-bound numerical comparisons and saved-feature replay are verified.
  [Report](outputs/reviews/music-representation-02/report.md),
  [static curves](outputs/reviews/music-representation-02/local-curves.svg),
  detailed results/failures/reproduction: [doc 24](docs/24_music_understanding.md#completed-team-1-experiment--2026-09-14).
  Do not rerun extraction merely to resume. Music Flamingo probing is complete
  and stopped; no MERT or other model run is currently assigned.
- **Runtime and findings:** the pinned 333,401,472-parameter MuQ checkpoint
  produced 5,530x1024 finite frames per pass (45 chunks), with about 51/54s summed
  CPU model inference and 3.35 GiB peak RSS. GPU access failed with driver/library
  mismatch and PyTorch error 804; no driver changes or remote compute were used.
  All 498 beats, 1,469 local contrasts and 12,898 recurrence pairs per pass are
  retained, plus 60 fixed listening joins and 616 explicit positive-return pairs.
  Drum entry gives a strong response already explained by the baseline, but the
  `none` example also responds and chunk placement materially changes scores.
  **No demonstrated salience, vocal-role or semantic-identity advantage; no policy
  promoted.** Full-mix versus separated-stem support confounds model comparisons.
- **MuQ validation:** lead focused tests **33 passed, 1 warning**; fresh Terra
  shared-worktree suite **731 passed, 123 warnings**. Lead verified all **161
  dependencies + 140 snapshots + 10 outputs per final/replay package**; all ten
  outputs replay byte-for-byte from saved features. Final plot was inspected;
  `git diff --check` passed. Frame-phase rounding and tail-ownership gaps were
  fixed before accepted extraction; `music-representation-runtime-01` remains
  rejected. Final runtime packages are `music-representation-runtime-02` and
  `music-representation-runtime-shifted-02`; final comparison/replay are
  `music-representation-02` and `music-representation-replay-02`. Pins in doc 24.
- **Team 2 evidence page accepted:**
  [evidence-timeline-10](http://127.0.0.1:8770/evidence-timeline-10/), manifest
  `a2b2b55fbb00b0bdd8859c4bd88be25336fa433590ef72010bed611a04c91fa1`.
  All 113 bound records and source/adapter parity verified; focused tests
  **3 passed, 1 warning** and expanded browser checker passed. Lead inspected
  actual changes and desktop/mobile captures; actual paired buttons stop at
  their saved endpoints. P1–P4 defects are resolved. The timeline JSON is
  byte-identical to version 05; that version and development 06–09 stay frozen.
  Minor caveats: opening-anchor wording after scale changes, and completion
  text replaced by a seek message. No rebuild assigned for those now.
  Evidence: `outputs/reviews/music-representation-integration-review-02/`.
  Acceptance is diagnostic-interface acceptance, not musical-role or user
  comprehension validation. No full-suite or model rerun was performed.
- **Latest completed Team 2 integration:** Team 1 verified Team 2's **119–132s authored
  vocal-emphasis comparison**. The implementation request is closed; do not
  rebuild it. Team 1 / Team A remains integration owner. Independent evidence:
  [doc 23](docs/23_role_context.md#authored-emphasis-integration-review--2026-09-12);
  implementation and reproduction commands: [doc 10](docs/10_directing_prototype.md).
- **Deferred artistic listening comparison:**
  [directed-vocal-emphasis-02](http://127.0.0.1:8770/directed-vocal-emphasis-02/),
  replay `outputs/reviews/directed-vocal-emphasis-replay-02/`. Compare **Reduced
  vocal emphasis** with **Steady vocal emphasis**. They start alike; only vocal
  gain/geometry decrease from 123 to 125.5s in Reduced. Shapes, focus, activity,
  accompaniment and original audio are otherwise identical. These are authored
  settings, not automatically detected musical roles or transition boundaries.
- **Authored-emphasis integration validation:** lead full suite **695 passed, 123 warnings**;
  Terra focused suite **100 passed, 1 warning**, independent full suite and review
  also passed. All **21 candidate + 24 replay** snapshot/output hashes and recorded
  origins verified, as did exact raw notes, source snapshots and HTML derivation.
  Both plans, signals, WAV and both MP4s replay byte-for-byte. Each video is
  **13s / 60FPS / 780 frames**; decoded audio matches between versions, and the
  573,300-frame stereo WAV exactly matches source FLAC samples at 119–132s.
  Native browser playback/switching, failure/timeout/retry, exact feedback binding
  and 320/375/768px layouts passed. Lead inspected ten decoded frames and pages;
  late Reduced vocals are visible in sampled non-silent frames but faint near
  129.2s. No blocking integration finding; musical preference remains open.
- **New visual feedback received:** after the visual pass, the user said
  **"they are better then before."** Doc 10 preserves this feedback and the
  prepared follow-up. This establishes relative visual improvement only, not
  acceptance of timing, automatic direction or every treatment. Do not ask for
  the same visual-improvement decision again before starting this bounded study.
- **Previous integration review complete:** Team 1 verified Team 2's visual pass
  and both v2 contract repairs. Detailed implementation is in
  [doc 10](docs/10_directing_prototype.md); independent integration evidence and
  closure are in [doc 23](docs/23_role_context.md#repair-closure-and-visual-pass-integration-review--2026-09-11).
  V2 focus now checks authoritative envelope gain; missing/malformed generated
  support is rejected. Both former failure reproductions now fail as intended,
  while the original frozen 02 plans validate unchanged.
- **Previous visual-identity comparison:** [directed-visual-02](http://127.0.0.1:8770/directed-visual-02/),
  replay at `outputs/reviews/directed-visual-replay-02/`. Voice uses vertical
  strands, snare angular marks, kick a low oval/halo, accompaniment broad lower
  contours. The plan preserves all 3,395 keyframes and the prior direction
  schedule. `previous.mp4` exactly matches the gradual video the user watched.
  The user feedback recorded in doc 10 concerned difficulty distinguishing
  voice/snare, awkward kick appearance and unfinished visuals; it was not an
  A/B preference or a new musical annotation.
- **Previous visual-pass validation:** lead full suite **666 passed, 123 warnings**;
  focused suite **71 passed, 1 warning**. All 46 new-package snapshot/output
  hashes plus 15 origin records verified; current sources and exact HTML/page
  derivation match. Both plans, audio and videos replay byte-for-byte; both
  videos are 48s/60FPS/2,880 frames with identical decoded audio and exact source
  PCM. Browser playback, switching, shortcuts, failure/retry, feedback linkage
  and 320/375/768px layouts passed; desktop/mobile pages and actual frames were
  inspected. A Terra reviewer found no blocking regression and checked old v1/v2
  pixel compatibility. Artistic acceptance and production promotion remain open.
- **Coverage limit:** the visual pass still covers 130–178s and misses all four
  prior focus intervals. Only the last 2s of the 119–132s audition overlap.
  The completed 119–132s authored-emphasis study now covers that excerpt, but
  does not establish performance on the other three prior listening examples.

- **Previous completed analysis experiment:** [continuous all-stem context](docs/23_role_context.md),
  implemented by two Terra workers and reviewed by the lead. It retains
  **1,469** supported all-stem contexts across the 499 grid boundaries:
  495 / 491 / 483 at 2 / 4 / 8 beats per side. Five before/after descriptors,
  signed differences, support and semantic unknowns are preserved independently
  of episode thresholds. No production or director policy was promoted.
- **New findings:** near the verse ending, vocal RMS and power share decrease at
  all 15 fixed anchor/scale combinations while activity stays 1 on both sides.
  Share differences range from −18.145 to −4.602 percentage points; spectral
  trends have mixed signs. Continued voice with reduced acoustic contribution
  is robust to this bounded anchor perturbation, but vocal function is unknown.
  In the user's `none` excerpt, the `other` stem also has consistent decreases
  across all 15 combinations. Stability cannot establish importance.
  At drum entry, vocal RMS can rise while its relative share falls: denominator
  changes matter. Detailed cases and limitations are in doc 23.
- **Current analysis artifact:** [role-context-02](http://127.0.0.1:8770/role-context-02/)
  (`outputs/reviews/role-context-02/`). Four readonly cases, selectable anchors,
  scales and descriptors, all-stem tables and original audio. Full grid:
  `role-context.json`; sensitivity: `evaluation.json`, `report.md`; provenance:
  `manifest.json`, `inputs/`. Page ~1.10MB, full context ~7.82MB.
  `role-context-01` remains a failed-mobile-check development artifact; version
  02 fixes footer wrapping, with byte-identical numeric/evaluation/review JSON.
- **Role-context verification:** then-current shared-worktree suite **643 passed, 123 warnings**;
  seven extractor and three package tests passed. Lead verified all 1,469
  support/share invariants and exact parity at all 259 frozen episode peaks.
  All **78** final-package fingerprints, HTML derivation, audio binding, raw
  notes, 60 fixed anchor/scale contexts and 240 summaries verified.
  Final browser smoke passed audio, bounded stop, seek/retry, four cases,
  actual scale/anchor updates, unknown spectra and mobile layout; no page
  overflow at 320/375/768px. Desktop/mobile screenshots visually inspected.
  Full-suite timing relative to final presentation fixes is recorded in doc 23.
  These earlier checks were not musical acceptance; Team 2's later integration
  review is recorded above.
- **Feedback used:** [guided listening intake](docs/20_listening_feedback.md).
  Raw `benchmark/feedback/listening-examples-01.json` remains hash-bound.
  Judgments: drum entry `local`, within-passage `none`, verse ending `subtle`
  (possibly stronger per note), breakdown `broad`. Voice may remain audible
  while losing its leading role; bass entry remains uncertain. Preserve the
  differing verse/chorus wording and every raw note.
- **Frozen response experiment:** [change-episodes-01](http://127.0.0.1:8770/change-episodes-01/),
  documented in [21](docs/21_change_episodes.md): 259 overlapping responses,
  3/17/11/26 in the four excerpts; four of five interpreted transition spans
  overlap a response, first short transition missed. Window blur and missing
  importance remain unresolved. Prior verification: 614 tests, 123 warnings;
  97 fingerprints and browser smoke passed.
- **Frozen policy comparison:** control / separate channels / sustained activity /
  combined yield **16 / 39 / 69 / 79 changes**, with the same two dip intervals.
  No variant promoted. Technical artifact:
  `outputs/reviews/local-structure-comparison-02/`; details in doc 18.
- **Prior listening artifact:** [listening-examples-01](http://127.0.0.1:8770/listening-examples-01/).
  Excerpts 74–85s, 16–26s, 119–132s, 57–70s. Focus bands remain prior prompts or
  an existing human interval; they are not newly inferred physical endpoints.
- **Delegation:** MuQ workers/reviewers finished. All teams may now delegate
  bounded work directly: OpenAI runners to available Sol/Terra/Luna workers;
  the user's Claude Opus 5 runner to available Sonnet/Haiku workers. Leads own difficult
  decisions and consequential review; workers own normal execution/fix cycles.
- **Baseline-explanation cards remain an unaccepted comprehension aid.** The
  user found the three layers “a little clear” but did not understand the purpose
  of another report. Preserve `baseline-explanation-01`; do not repeat its review
  or build an interactive version. The next artifact has a different purpose:
  explicit evaluation requirements for future methods, with source-bound evidence.
- **Strategy decision accepted.** After a direct current-versus-target explanation,
  the human authorized the proposed consolidation and starting the highest-priority
  task. Stable acoustic change does not establish meaningful development. Keep
  arrangement, continuation/identity, vocal behavior and transition extent as
  separate requirements; do not collapse them into a scalar importance target.
- **Three-lead proposal superseded by one active workstream.** Team 1 combines
  the small source-linked inventory and development benchmark. Terra implements
  the reproducible package; Luna audits evidence/uncertainty; the lead defines
  acceptance and reviews actual outputs. Broad research into four capabilities
  is deferred. A later second-model review gets one bounded experiment and a
  specific challenge, rather than another general planning assignment.
- **No additional annotations are required for the completed analysis.**
  Do not request the same four examples again. Use them for development evaluation.

## Active workstreams and cross-account coordination

Read [the collaboration protocol](docs/22_collaboration_protocol.md) after this
checkpoint when working across accounts. This table is the authoritative live
assignment; the protocol explains how to relay a request through the human when
the other team is offline. The integration owner alone updates this table and
the rest of this checkpoint after reviewing a completed handoff.

| Team | Owner | Bounded outcome | Owned paths | May start now? |
| --- | --- | --- | --- | --- |
| Team 1 / Team A — analysis and integration | This account / integration owner; one lead with Terra/Luna workers | Fixed arrangement comparison complete; hard continuity gate rejected. Human Agnes judgment received; timing match unresolved. Arctic Monkeys fixed-method comparison complete; doc-33 rule rejected; Agnes cue diagnostic complete; Castlecomer unchanged-method comparison complete. Attempt 02 ended at synthetic GPU Conv2d because the CUDA wheel lacked NVRTC; no checkpoint/audio/model result. `structure-review-06` represents the human's choice of separate section and event-reference layers; its separate event lane contains prompt windows, not extents or detector outputs. Targeted sign-off fails because diagnostic windows cover 78.2% of the song. The “79s they enter” note is potentially exposed to the candidate section boundary and is not independent evidence. Opus review finds real product/process progress but no demonstrated model-quality improvement; its suggested first screen of the existing section method is complete. Canonical run 02 matches run 01 across environments: both positive windows are empty and the negative control is empty. This rejects the current baseline for these references, not all section models. Attempt 03 stopped before R9 after R0–R8 passed. The ext4 floor and cumulative-write cap were both exceeded; see `outputs/reviews/all-in-one-feasibility-03/r9-preflight.json`. No study audio or Demucs checkpoint was touched. There is no All-In-One quality result. Do not resume this attempt; any new model/runtime experiment requires stable disk headroom, fresh preflight and explicit authorization. | `CONTINUE.md`, `docs/22_collaboration_protocol.md`, `docs/24_music_understanding.md`, `docs/29_development_benchmark.md`, `experiments/build_development_benchmark.py`, `tests/test_development_benchmark.py`, `docs/30_vocal_event_probe.md`, `experiments/probe_vocal_events.py`, `tests/test_vocal_event_probe.py`, `experiments/build_vocal_behavior_reference.py`, `experiments/templates/vocal_behavior_reference.html`, `tests/test_vocal_behavior_reference.py`, `docs/31_arrangement_continuity.md`, `experiments/arrangement_continuity.py`, `experiments/run_arrangement_continuity.py`, `experiments/extract_arrangement_reserve.py`, `tests/test_arrangement_continuity.py`, `docs/32_transition_components.md`, `experiments/diagnose_transition_components.py`, `tests/test_transition_components.py`, `experiments/evaluate_arctic_arrangement.py`, `tests/test_arctic_arrangement.py`, `docs/33_change_decision_screen.md`, `experiments/change_decision.py`, `experiments/run_change_decision.py`, `experiments/run_ssm_section_transfer.py`, `tests/test_change_decision.py`, `tests/test_ssm_section_transfer.py`, `.songviz/vocal-event-venv/`, `.songviz/vocal-event-models/`, and new `outputs/reviews/development-benchmark-*/`, `outputs/reviews/vocal-event-*/`, `outputs/reviews/vocal-behavior-reference-*/`, `outputs/reviews/arrangement-continuity-*/`, `outputs/reviews/transition-components-*/`, `outputs/reviews/change-decision-*/`, `outputs/reviews/event-reference-listening-*/`, `outputs/reviews/ssm-section-transfer-*/`, and `outputs/reviews/all-in-one-feasibility-*/` directories. Prior Team 1 representation, role-context, semantic runner/Modal paths and isolated environments retained for maintenance only. May inspect/test counterpart paths read-only for integration. | No active model/runtime execution. Attempt 03 stopped before R9; preserve its inputs and outputs. Another experiment requires stable disk headroom, a fresh preflight and explicit authorization; do not infer model quality from this resource stop. |
| Team 2 — bounded engineering | Independent account; Terra lead with direct available workers | Page 10 and corrected runtime audit accepted; no active task. | `docs/26_semantic_runtime_feasibility.md`, `docs/25_evidence_timeline.md`, `experiments/build_evidence_timeline.py`, `experiments/templates/evidence_timeline.html`, `experiments/check_evidence_timeline.cjs`, `tests/test_evidence_timeline*.py` and new `outputs/reviews/evidence-timeline-*/` directories, for explicitly assigned maintenance only. Prior direction/doc 10 ownership retained; artistic work deferred. | Completed; no rebuild, integration or runtime execution assigned. |
| Team 3 — difficult research/design | Independent Claude account; user's Claude Opus 5 lead with direct available Sonnet/Haiku workers | Revision 3 design accepted; R1–R3 closed; no active task. | `docs/27_semantic_experiment_design.md` for explicitly requested maintenance only. Other sources/artifacts read-only. No production code, checkpoint or counterpart-document writes. | Complete; no further design round or model execution assigned. |

All teams may read all inputs and frozen packages. Only the integration owner
updates `CONTINUE.md`. No team may alter another team's owned paths, shared
entry files (`AGENTS.md`, `README.md`),
raw feedback, source audio, controls, or existing review artifacts. A requested
interface change goes through the protocol and requires integration-owner review.
The requested shared delegation clarification is recorded in doc 22; no shared
`AGENTS.md` or `README.md` change was needed. Preferred lead names are workflow
choices, not verified provider IDs or automatic account settings. Availability
must be checked in each runner; do not silently substitute for Claude Opus 5.

**Current shared interface:** Team 2's page consumes the frozen schemas of
`role-context-02`, `structure-evaluation-03` and the existing feedback/audio
packages. Team 2 owns its display adapter. The page's first version does not
depend on Team 1's model schema. Model results will be integrated through a
separate reviewed request after both deliverables exist. Source support,
unknowns and provenance must remain visible. No director interface is adopted.

## Analysis-first workstream status

**Team 1: fixed arrangement comparison completed; reserve judgment received.**
The added continuity gate in doc 31 failed to retain the known drum increase.
Keep level changes and continuity uncertainty separate; do not tune this gate
merely to make a familiar example pass. The reserved Agnes excerpt now has frozen
predictions. The human reports a transition near elapsed 12s / song 105.29s,
tentatively verse to pre-chorus. The only level proposal is 1.664s earlier, with
feature support spanning the human time; hit/miss matching remains unresolved.
The existing separate pattern/level diagnostic in doc 32 is complete and supplies
no validated musical-importance rule. The next design question is selection and temporal support, not
threshold tuning on one passage. It is not a generalization success. Musical role/importance
remains distinct from acoustic activity. Existing judgments and frozen packages
remain intact. Do not rebuild the benchmark, repeat the comparison, restart YAMNet
or reopen Music Flamingo merely to resume. The vocal-labeling branch stays parked.

The four raw notes and uncertainty remain authoritative. Evaluation references
must stay out of future candidate inputs; the output template is not an inference
runner or proof of isolation. These are seen development cases, not held-out
validation. The metadata-selected 24s Arctic Monkeys response is recorded.
Doc 33's completed rule fails its non-change check. The Agnes clarification
and signed-level diagnostic are complete. Castlecomer's new-part judgment and
fixed-seconds and unchanged-method comparisons are now preserved.
Its missing stems/beat inputs were prepared and the unchanged-method comparison
completed as above. Following the user's strategic challenge, the audible-cue
question is optional. The one-shot stem-free section-method transfer screen is
complete and fails both positive windows while passing the no-change control.
Do not tune this baseline or collect more listener notes. Attempt 03 passed R0–R8 and stopped before R9 when the ext4 floor and cumulative-write cap were exceeded. No study-audio or model-quality result was produced. Do not resume attempt 03 or start another runtime search without stable disk headroom, a fresh preflight and explicit authorization. If work resumes without another model attempt, prioritize existing within-section event references and stems while keeping that evaluation separate from section-boundary work.

The former semantic execution gates are closed: the replacement listener control
passed, semantic-probe-02 was frozen, and corrected Music Flamingo execution
failed its positive controls. The stopping rule still applies. Team 2's page 10
and runtime audit and Team 3's revision-3 design are accepted, with no repairs
pending. Historical validation counts in the checkpoint were not rerun now.

## Completed analysis-first implementation requests

The implementation requests below are acceptance records, not instructions to
repeat completed work. Team 1's MuQ request is closed; Team 2's version-05
implementation was returned, and the version-10 repair candidate is now accepted.
All existing review directories remain immutable. Integration evidence is in doc 24;
Team 1 does not edit Team 2's owned implementation paths.

```text
REQUEST
owner: Team 1 (integration)
to: Team 1
goal: Measure MuQ inference feasibility and produce a source-bound comparison
      of learned local change, recurrence and historical novelty with the baseline.
read: CONTINUE.md; docs/24_music_understanding.md; docs/16_structural_evaluation.md;
      docs/20_listening_feedback.md; docs/23_role_context.md
write_scope: Team 1 paths in the active table; only unused output directories.
do_not_touch: Team 2 files; source audio; raw feedback; frozen packages; existing
              runtime/analysis caches; system GPU drivers.
validation: Small real inference before full extraction; pin model/config/source;
            frame/support and chunk-seam checks; numeric invariants and replay;
            inspect comparisons across existing listening and positive-return cases.
handoff: Detailed results/commands/failures in doc 24; lead updates CONTINUE.
ask_human_if: Paid compute, external audio upload, or system changes are needed;
              first exhaust a bounded local feasibility check and report its result.

REQUEST
owner: Team 1 (integration); Team 2 (implementation)
to: Team 2
goal: Build an inspectable full-song evidence page with synchronized original
      audio, selectable metrics, factual text and paired recurrence listening.
read: CONTINUE.md; docs/22_collaboration_protocol.md;
      docs/24_music_understanding.md (Team 2 design/acceptance);
      docs/16_structural_evaluation.md; docs/20_listening_feedback.md;
      docs/23_role_context.md; frozen baseline manifests and schemas.
write_scope: Team 2 evidence-timeline paths in the active table; start with
             outputs/reviews/evidence-timeline-01/ only if it does not exist.
do_not_touch: CONTINUE.md; Team 1 files; shared entry files; original audio;
              raw feedback; frozen packages; detector/director/render code.
validation: Consumed hashes and displayed-value/source parity; focused mapping
            tests; native audio, seeking, paired listening, unavailable support,
            retry/error, desktop/mobile checks and actual page inspection.
handoff: Record commands, artifacts, hashes, findings and limits in doc 25;
         human returns to Team 1 for integration and CONTINUE update.
ask_human_if: Required inputs fail provenance or changes exceed the assigned scope.
```

## Closed Team 2 repair request

Both requested v2 contract repairs are implemented and independently verified.
Envelope gain is authoritative for v2 focus/render behavior; scalar gain remains
bounded compatibility metadata. Generated envelope support must match its
segment and signal clock; authored envelopes require explicit provenance.
Frozen 02 plans remain compatible, and the former zero-focus/missing-support
reproductions are rejected. See doc 23 for closure evidence.

Do not run the old repair request again. Team 2's completed visual pass and both
rounds of raw visual feedback are documented in doc 10. No role-context director
interface has been adopted.

## Completed Team 2 implementation request

**Closed after Team 1 integration review, 2026-09-12.** The request below is
retained as the acceptance record, not runnable work. Both output destinations
now exist and are immutable. Team 2 also supplied the reviewed read-only browser
diagnostic `tests/test_directed_review_browser.cjs`; it does not write packages
or save synthetic feedback as human input. No follow-up artistic change is assigned.

```text
REQUEST
owner: Team 1 (integration); Team 2 (implementation)
to: Team 2
goal: Build an inspectable 119–132s steady/reduced vocal-emphasis comparison.
read: CONTINUE.md; docs/22_collaboration_protocol.md; docs/10_directing_prototype.md
      prepared proposal; docs/23_role_context.md authored-emphasis design review.
write_scope: songviz/direction.py; songviz/directed_render.py;
             experiments/build_directed_review.py;
             experiments/templates/directed_review.html; tests/test_direction.py;
             tests/test_directed_*.py; docs/10_directing_prototype.md;
             outputs/reviews/directed-vocal-emphasis-02/;
             outputs/reviews/directed-vocal-emphasis-replay-02/.
do_not_touch: Team 1 code/docs; CONTINUE.md; AGENTS.md; README.md; raw feedback;
              source audio; caches; all existing packages and controls.
validation: Focused directing/builder/render tests and full pytest suite; strict
            paired-plan isolation and authored provenance; frozen input hashes;
            saved-plan replay; 13s/60FPS/780-frame videos, identical decoded
            audio and source PCM; actual frame and native browser/mobile checks.
handoff: Record actual changes, commands, hashes, failures and limitations in
         doc 10; return to Team 1 for durable integration review in CONTINUE.md.
ask_human_if: A target output already exists, source provenance fails, scope
              expansion is needed, or the fixed experiment requires redesign.
```

Use one contiguous scene with fixed vocal focus, refreshed shapes/anchors and
identical constant accompaniment envelopes. Both vocal envelopes use timestamps
119, 123, 125.5 and 132s: control gain/emphasis stay 1/1; candidate stays 1/1
through 123s, linearly reaches .45/.30 at 125.5s and holds to 132s. These are
authored experimental choices, not detected boundaries or calibrated importance.
Keep all remaining plan fields identical except explicitly identified authorship
descriptions; validate this with a narrow whitelist, including replay overrides.
Reuse verified full-track signals, but cut fresh 119–132s PCM from the bound
original FLAC: the parent WAV only covers 130–178s. Replay must use the newly
saved plans/signals/PCM without replanning or re-extracting inputs. Preserve
old-mode behavior. See doc 23 for renderer semantics and detailed acceptance.

## Product intent

A single-command **automatic visual director for a song**: analyze audio, save a
direction plan, render video synchronized to the original audio. An LLM may
assist planning; no particular provider is required and it need not act per
frame. The final full-song command/director does not exist yet.

The visuals should tell the song's story: selective focus and omission, multiple
treatments per instrument, recognizable motifs on returns, changes that follow
musical developments. This is not an always-on instrument dashboard or one fixed
animation per stem. The user's 2026-09-13 priority is analysis and explanatory
listening tools first; artistic visual development is deferred. Perfect
transcription is not required to inspect and improve song understanding.

Keep distinct: **musical identity**, **arrangement variation**, **transition
intervals**, **local change**, and **unfamiliar material relative to history**.
A familiar chorus can return with a strong local change and different energy.

User clarification during the comparison: **a single discrete section level is
not the intended musical model**. Verse/chorus/bridge can be useful identity
anchors, while smaller and larger transitions and degrees of novelty coexist.
Preserve overlapping, multiscale and graded evidence; a thresholded candidate
list is a diagnostic view, not the complete representation or a mandatory visual
cut list. Continuous visual evolution should not require crossing a section
boundary. Scale, magnitude, familiarity and artistic importance are distinct.

## Next implementation / integration review

The continuous-context experiment is complete; read [23](docs/23_role_context.md)
for its fixed design, findings, verified artifact and limits. [21](docs/21_change_episodes.md)
documents the preceding response experiment; [18](docs/18_local_structure_comparison.md)
and [17](docs/17_local_structure.md) retain the frozen comparison/control.

Team 1's review of the saved-plan comparison, contract repairs and later visual
pass is complete, as is the subsequent 119–132s authored-emphasis integration.
The MuQ comparison and Team 2's evidence-timeline-10 integration are complete.
Music Flamingo's corrected probe failed and is stopped. The benchmark is accepted;
the local vocal-event screen is complete with a negative result in doc 30.
The temporal vocal-reference branch is parked after the human's generalizability
correction; it does not block arrangement/continuity work. Do not repeat page-05 integration
or semantic setup. Artistic preference remains deferred. Existing audio, timing
and completed packages remain controls; page/model integration requires its own
concrete experiment.

The subsequent fixed arrangement comparison (doc 31) is complete and rejects the
added hard continuity gate. Its numeric output and reserved-passage predictions
remain frozen. The human judgment on the 24-second passage is recorded. The
completed doc-32 diagnostic reused existing algorithms and verified features.
The stem-free transfer screen for the existing section method is complete and
misses both positive windows while leaving the negative window empty. No
production detector or visualization promotion is queued; the next model setup
requires fresh, explicit authorization as recorded at the checkpoint.

A small authored graded-emphasis study is supported as an experiment; an
automatic role/importance or share-to-emphasis policy is not. Preserve raw notes
and uncertainty from [20](docs/20_listening_feedback.md), including perceived
non-change at 21s and continuing voice near 124s. No new detector or full-song
director is authorized merely by finishing these descriptors.

Immediate regression examples: chorus low/high-energy marks **78.889763s** and
**165.533910s**, verse-2 outro **124.295069s**, chorus return **158.590911s**.
Preserve useful candidates near 33.445s and 203.759s as well. Timing tolerance
and formal hierarchy are still open. Feel Good Inc is development data; reserve
new listening examples before general-quality claims. Unlabeled events are not
automatically false positives.

## Completed work and limitations

| Work | What exists / what not to assume | Details |
| --- | --- | --- |
| Baseline and rhythm review | Original audio/caches preserved; user clarified the hard-to-follow pulse was **cached**, not the reviewed regular pulse | [07](docs/07_restart_review.md), [08](docs/08_rhythm_feedback.md) |
| Visual vocabulary and directing | Fixed opening study plus a rule-based saved plan/render/replay for a 48s passage (130–178s). Not an accepted full-song director; musical emphasis still needs review | [09](docs/09_visual_passage.md), [10](docs/10_directing_prototype.md) |
| Structural beat grid | Explicit grid/provenance; reviewed pulse about 138.5 BPM versus cached 92.3 BPM. Bar/downbeat phase is unverified | [11](docs/11_beat_grid_audit.md), [12](docs/12_structure_grid.md) |
| Boundary/novelty fixes | Fuse agreeing timestamps once; missing history is unknown; no per-lag/per-song rescaling. Ordered 16/32-beat recurrence is acoustic evidence, not certified identity | [13](docs/13_structure_review.md) |
| Audio repair | Native WAV URL and ranged localhost server. Previous Blob playback failed; don't reintroduce fetch-to-Blob audio | [13](docs/13_structure_review.md) |
| Section editor/feedback | Blank independent layers; received 19 spans, four explicit motif groups and short transitions. No formal hierarchy imposed | [14](docs/14_section_annotation.md), [15](docs/15_section_feedback.md) |
| Structural evaluation | Identity groups separate from analyst variation/transition tags; pattern/level and local/history evidence. Identity windows may cross adjacent same-motif variations without merging spans | [16](docs/16_structural_evaluation.md) |
| Local candidates | 2/4/8-beat contrasts; bounded 2–12-beat dip/recovery intervals. No new partition, semantic identity assignment or production/render change | [17](docs/17_local_structure.md) |
| Fixed policy comparison | Separate thresholds plus sustained per-stem activity. 16/39/69/79 changes; more local evidence but no validated salience hierarchy or promoted winner | [18](docs/18_local_structure_comparison.md) |
| Guided listening | Four audio excerpts with plain-language hypotheses, optional perceived-change/visual-emphasis feedback, isolated drafts and export | [19](docs/19_listening_examples.md) |
| Continuous role context | All supported grid anchors at 2/4/8 beats per side; fixed neighboring-anchor sensitivity and raw notes. Stable acoustic change is not importance or vocal function | [23](docs/23_role_context.md) |
| MuQ representation comparison | Real CPU extraction, full base/shifted local/ordered-history evidence, 60 listening joins and 616 positive-return pairs; exact saved-feature replay. Feasible diagnostic, no demonstrated semantic or salience advantage | [24](docs/24_music_understanding.md#completed-team-1-experiment--2026-09-14) |
| Change responses | Independent multiscale intervals and all-stem context; physical duration, hierarchy and perceived importance remain unknown | [21](docs/21_change_episodes.md) |

Known failures, not unfinished installation steps:

- Legacy six-section candidate calls the last ~55s an outro; user instead marks
  chorus/verse returns, with only the last ~3.2s the song's outro.
- The original local policy misses both chorus energy splits despite old cuts near them.
  Contrast exists but falls below current automatic thresholds.
- The activity variant supplies nearby evidence for those changes, at a cost of
  many more proposals. Combining channels can worsen a timestamp via suppression;
  the full source support matters more than treating a selected peak as truth.
- Dip proposals 63.944–65.243s and 93.830–95.563s cover only parts of two
  interpreted transitions (IoU ~0.235 / ~0.571). Three others have no overlap.
  The deepest energy trough is not necessarily the whole musical transition.
- Long fixed windows lack support for many short variations. Crossing adjacent
  same-motif subdivisions supports identity, not a single variation.
- Scores are uncalibrated. Offline thresholds use whole-song statistics and local
  comparisons need right-side context. Full-phrase novelty is not instantaneous
  surprise. Keep support/availability times explicit.
- Episode windows blur sharp changes; 259 responses are not a musical event list.
  Context at a selected peak can miss the relevant vocal development. Continuous
  context now removes that selection dependency, but role/importance remains
  unknown; stable changes also occur in the human non-change example.

## Authoritative inputs and artifacts

Paths are relative to the repo. Existing packages are immutable controls; older
numbered drafts remain for provenance, not the preferred entry point.

- Raw feedback: `benchmark/feedback/section-editor-02.json`, copied byte-for-byte
  from the user's `Downloads/songviz-sections (1).json`. SHA-256:
  `dd100b34d43ece3dbe500b297f0fb4320db71e3eebb9dc705959145e54aa8b2f`.
- Guided listening export: `benchmark/feedback/listening-examples-01.json`, SHA-256
  `f69b14f0343d0ebe4af12d37f66e88c76aa5d8e7673f7529345333589d5c47b6`;
  all four responses verified, interpretation and qualifications in doc 20.
- Analyst mapping: `benchmark/feedback/section-editor-02.interpretation.json`.
  Hash-bound to the export; **not** extra user annotation. All user certainty is
  `unspecified`. Empty motif means unknown; distinct names are not automatically
  negative identity examples. Do not force verse 1 and verse 2 into one group.
- Source: `songs/Gorillaz - Feel Good Inc (featuring De La Soul).flac`, duration
  221.173333s; SHA-256:
  `657af9333edafbb6cf8cc573446650773d1169b0117c784106d1895cf802cb44`.
- Cached stems/story: `outputs/Gorillaz - Feel Good Inc (featuring De La Soul)/`.
- Current proposal control: `outputs/reviews/local-structure-02/`:
  `predictions.json`, `evaluation.json`, `report.md`, `index.html`, `manifest.json`,
  and `inputs/` code snapshots. Check these before redoing work.
- Reusable evidence: `outputs/reviews/structure-evaluation-03/`:
  **`features.npz`** (numeric beat log-CQT/RMS; load with `allow_pickle=False`),
  `timing.json`, `reference.json`, `recurrence.json`, `legacy-sections.json`, manifest.
- Accepted development benchmark: `outputs/reviews/development-benchmark-02/`;
  source-bound reference, manual rubric and blank output schema. Draft 01 rejected
  and frozen. Full validation and next experiment boundary in doc 29.
- Completed vocal-event screen: `outputs/reviews/vocal-event-probe-01/`, manifest
  `85dc9734ec5a8a34625879a0b8ca5be50d3c9b31e2fbf5502ee56fa74e510434`.
  Frozen protocol/runner/guard in `vocal-event-execution-01/`; saved-score report
  and PNG/SVG curves in `vocal-event-analysis-01/`; runtime in
  `vocal-event-runtime-01/`, all under `outputs/reviews/`. Negative result; do not
  rerun or promote. Exact pins and checks in doc 30.
- Accepted blank temporal-reference page: `outputs/reviews/vocal-behavior-reference-03/`,
  manifest `66471289b7bde9745970c0870e8ac0f79bf17f42732fd7e869094abbb2077504`.
  Lead verification/scripts/screenshots: `vocal-behavior-reference-validation-01/`
  under the same review root. Versions 01/02 are superseded development artifacts.
  No human feedback is present yet; schema defaults are not annotations.
- Completed arrangement comparison: `outputs/reviews/arrangement-continuity-01/`,
  manifest `f077d3025280a6aab5eb3330ba9ed226d718e26b396442ac6dac00c5f43a4a70`.
  Frozen rules in `arrangement-continuity-registration-01/`; fresh, source-bound
  Agnes features in `arrangement-continuity-reserve-features-01/`; independent
  arithmetic in `arrangement-continuity-audit-01/`; neutral original audio in
  `arrangement-continuity-listening-01/`, all under `outputs/reviews/`. Added gate
  rejected; the reserve judgment is now recorded separately below. Full validation/commands in doc 31.
- Human reserve feedback: `outputs/reviews/arrangement-continuity-listening-01/human-feedback.json`.
  Approximate transition near elapsed 12s / song 105.294263s; tentative verse to
  pre-chorus. This is raw new feedback, not exact section truth.
- Verified continuous context: `outputs/reviews/role-context-02/`; full
  `role-context.json`, fixed sensitivity `evaluation.json`, bounded `review.json`.
  Version 01 is preserved after its failed footer/mobile-width check. See doc 23.
- Verified MuQ comparison/replay: `outputs/reviews/music-representation-02/`
  and `music-representation-replay-02/`; base/shifted extraction in
  `music-representation-runtime-02/` and `music-representation-runtime-shifted-02/`.
  Do not reuse rejected runtime-01. Full pins, measurements and commands in doc 24.
- Accepted evidence page: `outputs/reviews/evidence-timeline-10/`; page 05 and
  development versions 06–09 remain frozen. Implementation evidence is in doc 25;
  accepted integration is in `music-representation-integration-review-02/`.
- New frozen response experiment: `outputs/reviews/change-episodes-01/`.
  All 259 responses and 24 full curves are retained; no production detector changed.
- Legacy full-song review/native original WAV: `outputs/reviews/structure-review-03/`.
  WAV hash: `7d800828527fb8812dc6433361d1d72ad1f7cdba368b9c51dce4abc954f7a5c1`.
- Verified visual comparison: `outputs/reviews/directed-visual-02/` and
  `directed-visual-replay-02/`; same saved direction schedule, refreshed shapes,
  exact replay and original gradual-video control. Technical integration passed;
  user visual acceptance remains pending.
- Verified directed comparison: `outputs/reviews/directed-review-02/` and
  `directed-replay-02/`; exact three-way replay and source audio verified.
  V2 contract follow-up is closed; see doc 23 integration review and doc 10.
- Blank editor: `outputs/reviews/section-editor-02/`; directed preview:
  `outputs/reviews/directed-review-01/`; saved-plan replay: `directed-replay-01/`
  under the same reviews root.

Verify manifests before reuse. Avoid dumping huge story/recurrence JSONs into
context; inspect targeted fields or the small numeric cache. Audio and `outputs/`
are gitignored: a fresh clone alone cannot reproduce these experiments. Check
missing files and disk space first; do not silently rerun expensive separation
or erase artifacts to make room.

## Code and commands

- Development benchmark: `experiments/build_development_benchmark.py`; focused
  checks: `tests/test_development_benchmark.py`; design/reproduction in doc 29.
- Completed vocal-event probe: `experiments/probe_vocal_events.py`; focused
  checks: `tests/test_vocal_event_probe.py`; frozen execution/result in doc 30.
- Temporal-reference page: `experiments/build_vocal_behavior_reference.py`,
  `experiments/templates/vocal_behavior_reference.html`; focused checks:
  `tests/test_vocal_behavior_reference.py`. Accepted package and next intake in doc 30.
- Fixed arrangement comparison: `experiments/arrangement_continuity.py`,
  `experiments/run_arrangement_continuity.py`, `experiments/extract_arrangement_reserve.py`;
  focused checks: `tests/test_arrangement_continuity.py`; result/reproduction in doc 31.
- MuQ numerical comparisons: `songviz/music_representation.py`; CPU extraction:
  `experiments/probe_music_representation.py`; saved-feature review builder:
  `experiments/build_music_representation_review.py`; focused tests:
  `tests/test_music_representation*.py`. Final reproduction commands in doc 24.
- Accepted evidence-page implementation (maintenance only): `experiments/build_evidence_timeline.py`,
  `experiments/templates/evidence_timeline.html`,
  `experiments/check_evidence_timeline.cjs`, `tests/test_evidence_timeline_builder.py`.
  Read doc 25 before checks; these remain Team 2's owned implementation paths.

- Continuous context: `songviz/role_context.py`; builder:
  `experiments/build_role_context_review.py`; page:
  `experiments/templates/role_context_review.html`; browser check:
  `experiments/check_role_context_review.cjs`; focused tests:
  `tests/test_role_context.py`, `tests/test_role_context_builder.py`.

- Change responses: `songviz/change_episodes.py`; builder:
  `experiments/build_change_episode_review.py`; tests:
  `tests/test_change_episodes.py`, `tests/test_change_episode_builder.py`.
- Detector: `songviz/local_structure.py`; evaluation: `songviz/local_structure_evaluation.py`.
- Comparison detector: `songviz/local_structure_variants.py`; runner:
  `experiments/compare_local_structure.py`; tests:
  `tests/test_local_structure_variants.py`, `tests/test_local_structure_comparison.py`.
- Guided review builder: `experiments/build_listening_examples.py`; page:
  `experiments/templates/listening_examples.html`; browser check:
  `experiments/check_listening_examples.cjs`; evidence tests:
  `tests/test_listening_examples.py`.
- Builder: `experiments/build_local_structure_review.py`; page:
  `experiments/templates/local_structure_review.html`; smoke:
  `experiments/check_local_structure_review.cjs`.
- Tests: `tests/test_local_structure.py`, `tests/test_local_structure_evaluation.py`,
  `tests/test_local_structure_builder.py`.
- Feedback contracts: `songviz/structure_annotations.py`, `songviz/structure_evaluation.py`;
  recurrence: `songviz/recurrence.py`; broader detector/grid: `songviz/story.py`,
  `songviz/structure_grid.py`.
- Later director integration: `songviz/direction.py`, `songviz/directed_render.py`.

Run from the repo root (current checkout:
`/home/njoppi2/projetos/pessoal/song-visualizer`):

```bash
.songviz/venv/bin/python -m pytest -q --disable-warnings

# Current continuous-context review:
SONGVIZ_PLAYWRIGHT_MODULE=/home/njoppi2/.npm/_npx/04489aa25f0b609a/node_modules/playwright \
  node experiments/check_role_context_review.cjs \
  http://127.0.0.1:8770/role-context-02/

# Start only if port 8770 is not already serving the reviews:
.songviz/venv/bin/python -m songviz.review_server

# Optional frozen-control browser smoke; resolve this installation if unavailable:
SONGVIZ_PLAYWRIGHT_MODULE=/home/njoppi2/.npm/_npx/04489aa25f0b609a/node_modules/playwright \
  node experiments/check_local_structure_review.cjs \
  http://127.0.0.1:8770/local-structure-02/
```

Frozen-control page: `http://127.0.0.1:8770/local-structure-02/`. The server was available
at handoff, but processes may not survive a new session. Use the existing Python
3.10 venv; system Python may differ. Builders refuse existing output directories.
Follow doc 17 to reproduce under a new directory; rerunning the same defaults is
not the proposed next algorithm. For the completed comparison package, run:

```bash
SONGVIZ_PLAYWRIGHT_MODULE=/home/njoppi2/.npm/_npx/04489aa25f0b609a/node_modules/playwright \
  node experiments/check_local_structure_comparison.cjs \
  http://127.0.0.1:8770/local-structure-comparison-02/
```

## Delegation and quality policy agreed with the user

All teams follow the shared [delegation policy](docs/22_collaboration_protocol.md#delegation-is-available-to-every-team).
It explicitly covers OpenAI and Claude leads delegating directly, disjoint scope,
worker-owned execution/fix cycles, lead review, proportionate validation and
actual cost/rework evidence. The active table above alone assigns current tasks
and preferred lead models. No team is the exclusive implementation gateway.

Ask the user about meaningful musical ambiguity/artistic preference on prepared
short examples, not to debug every iteration or repeat existing annotations.

No custom orchestration platform is needed. Preserve dirty/untracked work; no
commits, resets, cleanup or broad rewrites were requested.

## Later direction and documentation ownership

After the first analysis deliverables: evaluate semantic model claims, improve
transition extents and recurring-material evidence, and validate on held-out
music. Resume artistic direction when an inspectable analysis review supports
the next experiment. Full-song saved-plan/render remains the end goal.

- Product intent: [README.md](README.md).
- Milestones/working loop: [docs/01_roadmap.md](docs/01_roadmap.md).
- Architecture/open decisions: [docs/02_architecture.md](docs/02_architecture.md).
- Detailed inventory/history: [docs/03_working_state.md](docs/03_working_state.md).
- Current comparison evidence: [docs/18_local_structure_comparison.md](docs/18_local_structure_comparison.md);
  frozen-control details: [docs/17_local_structure.md](docs/17_local_structure.md).
- Guided listening and optional-feedback contract: [docs/19_listening_examples.md](docs/19_listening_examples.md).
- Received listening judgments and next-step implications: [docs/20_listening_feedback.md](docs/20_listening_feedback.md).
- Response-episode design, findings and next context experiment: [docs/21_change_episodes.md](docs/21_change_episodes.md).
- Continuous role-context evidence and next integration review: [23](docs/23_role_context.md).
- Analysis-first experiment history: [24](docs/24_music_understanding.md).
- Current development benchmark: [29](docs/29_development_benchmark.md).
- Local vocal-event screen: [30](docs/30_vocal_event_probe.md).
- Arrangement/continuity comparison: [31](docs/31_arrangement_continuity.md).
- Feedback semantics: [docs/15_section_feedback.md](docs/15_section_feedback.md),
  [docs/16_structural_evaluation.md](docs/16_structural_evaluation.md).

Historical inventories and phase numbers do not override this checkpoint. If
files or fresh checks disagree, investigate and update the handoff rather than
assuming an old result holds. After the next task, update **this checkpoint,
next step, failures and artifact pointers** and the relevant experiment document.
Keep prior results attributable; do not create another competing status index.
