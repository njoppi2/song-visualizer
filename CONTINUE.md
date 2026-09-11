# SongViz — start/resume here

This is the **single entry point for a new agent or compacted conversation**.
Updated 2026-09-11 after verifying the multiscale change-response experiment.
Read this file, then the linked document for the task at hand. Existing changes
belong to the user; the working tree contains substantial modified/untracked work.

## Current checkpoint

- **Latest completed experiment:** [multiscale change responses](docs/21_change_episodes.md).
  Detector and review package implemented by Terra and verified by the lead.
  It preserves independent overlapping
  2/4/8-beat stem/channel responses, graded curves, release-confirmation support,
  censoring and all-stem before/after descriptors. Physical onset/settling,
  perceived importance and vocal function remain unknown. No production change.
- **Feedback driving this experiment:** [guided listening intake](docs/20_listening_feedback.md).
  All four examples have selections and notes; export, review and source/audio
  fingerprints matched. Raw: `benchmark/feedback/listening-examples-01.json`.
  Judgments: drum entry `local`, within-passage `none`, verse ending `subtle`
  (qualified as possibly stronger in the note), breakdown `broad`.
  Voice may stay audible while losing its leading role. Bass entry remains
  uncertain; preserve the user's differing verse/chorus wording without
  rewriting earlier identities. No detector changed during intake.
- **New result:** 259 overlapping response episodes, with 3/17/11/26 in the four
  listening excerpts. Four of five interpreted transition spans overlap at
  least one response; the first short transition is missed. Window blur broadens
  even an instantaneous step. Best overlap is retrospective and is not accuracy.
  Vocal share near 125.449s falls 17.38%→3.69% while remaining active, but this
  context is anchored to a bass episode; vocal-role recognition remains unsolved.
  Seventeen responses in the user's `none` excerpt expose missing importance
  modeling. Detailed signed endpoint errors and limitations belong in doc 21.
- **Frozen comparison result:** control / separate channels / sustained activity / combined yield
  **16 / 39 / 69 / 79 changes**, respectively, with the same two dip intervals.
  Activity exposes nearby candidates at all four earlier regression examples,
  but adds many within-phrase changes. Combining policies worsens one chorus
  timestamp. **No variant is promoted to production.**
- **Latest verification:** full suite **614 passed, 123 warnings**; detector's
  16 focused tests and builder's three tests passed. All **97** package
  source/snapshot/output fingerprints, HTML derivation and exact raw notes
  matched. Browser smoke passed native original audio, bounded playback,
  four-case navigation, readonly notes and mobile layout. Additional checks
  passed seeking/retry, selected bass context and visible frozen dip; desktop
  and mobile screenshots captured, mobile visually inspected. These are
  implementation checks, not musical acceptance or production promotion.
- **Current artifact:** [change-episodes-01](http://127.0.0.1:8770/change-episodes-01/)
  (`outputs/reviews/change-episodes-01/`). Four readonly listening cases, local
  response bands and all-stem context; no new feedback is requested. Complete
  curves: `episodes.json`; diagnostics: `evaluation.json`, `report.md`; provenance:
  `manifest.json`, `inputs/`. The page is ~147KB and references existing audio.
- **Previous verification:** nine focused explanation/evidence tests passed;
  all 15 guided-package source/snapshot/output fingerprints and page derivation
  matched. Browser checks passed bounded original-audio playback, navigation,
  optional feedback export, draft persistence/isolation, and mobile layout;
  the mobile page was visually inspected. No full-suite rerun for this review-only
  change. The previous comparison's full suite passed **586 tests, 123 warnings** after the final
  package/payload work; its focused comparison suite passed **35 tests, 1 warning**.
  All 78 final-package fingerprints, page derivation and browser
  smoke (native audio, seeking, retry, bounded audition, mobile) passed. These
  recorded results are not a guarantee after later edits.
- **Prior user-facing artifact:** [listening-examples-01](http://127.0.0.1:8770/listening-examples-01/)
  (`outputs/reviews/listening-examples-01/`). Four curated excerpts: 74–85s,
  16–26s, 119–132s, 57–70s. Focus bands are listening prompts; the last reuses
  the existing human transition interval. None is a newly inferred extent.
  Optional export: `songviz-listening-examples-feedback.json`; unanswered items
  remain null. Do not ask the user to choose an algorithm or repeat all sections.
- **Technical comparison artifact:** `outputs/reviews/local-structure-comparison-02/` at
  `http://127.0.0.1:8770/local-structure-comparison-02/`. Version 01 is retained
  as a development artifact; 02 is the bounded-page review package.
- **Delegation:** two Terra workers implemented detector/tests and review
  builder/page/tests, respectively; lead designed the experiment and reviewed
  actual code, numerical findings and limitations. Both workers are now closed.
- **Next analysis step after this package is verified:** compute all-stem role
  context over the entire beat grid, independently of thresholded episodes.
  Compare nearby anchors/scales on these same four examples before interpreting
  reduced relative contribution as a change of musical role. See doc 21.
- **No additional user annotations are required to start the next experiment.**
  The guided feedback is now received and hash-bound. Do not request the same
  four examples again; use the notes and their qualifications in doc 20.

## Active workstreams and cross-account coordination

Read [the collaboration protocol](docs/22_collaboration_protocol.md) after this
checkpoint when working across accounts. This table is the authoritative live
assignment; the protocol explains how to relay a request through the human when
the other team is offline. The integration owner alone updates this table and
the rest of this checkpoint after reviewing a completed handoff.

| Team | Owner | Bounded outcome | Owned paths | May start now? |
| --- | --- | --- | --- | --- |
| Team 1 — analysis | This account / integration owner | A continuous all-stem role-context experiment, evaluated independently of thresholded episodes. It must retain scales, support/unknown fields and raw feedback without treating RMS share as vocal leadership. | New `songviz/role_context.py`, `experiments/build_role_context_review.py`, `experiments/templates/role_context_review.html`, `experiments/check_role_context_review.cjs`, `tests/test_role_context*.py`, `docs/23_role_context.md`, a new ignored `outputs/reviews/role-context-01/` only. | Yes. It does not wait for Team 2. |
| Team 2 — direction | Other Astra account | A bounded directed-passage comparison using gradual emphasis, selective omission and motif continuity. It must be an inspectable saved-plan experiment, not a claim that all analysis is solved or an LLM/full-song director. | `songviz/direction.py`, `songviz/directed_render.py`, `experiments/build_directed_review.py`, `experiments/templates/directed_review.html`, `tests/test_direction.py`, `tests/test_directed_*.py`, `docs/10_directing_prototype.md`, and a new ignored `outputs/reviews/directed-*-02/` only. | Yes. It may use frozen current evidence and must not wait for Team 1's future role-context output. |

Both teams may read all inputs and frozen packages. Neither team may alter the
other's owned paths, shared entry files (`AGENTS.md`, `CONTINUE.md`, `README.md`),
raw feedback, source audio, controls, or existing review artifacts. A requested
interface change goes through the protocol and requires integration-owner review.

**Current shared interface:** Team 2 may treat `change-episodes-01` as a readonly
development input. A candidate plan may consume its existing timestamps/stem
labels/unknown fields, but cannot convert a response count, RMS share or missing
episode into a claim of importance or vocal function. Team 1 will not promise a
new interface before its experiment has been reviewed. The two streams therefore
have no blocking dependency.

## Product intent

A single-command **automatic visual director for a song**: analyze audio, save a
direction plan, render video synchronized to the original audio. An LLM may
assist planning; no particular provider is required and it need not act per
frame. The final full-song command/director does not exist yet.

The visuals should tell the song's story: selective focus and omission, multiple
treatments per instrument, recognizable motifs on returns, changes that follow
musical developments. This is not an always-on instrument dashboard or one fixed
animation per stem. Develop visual experience alongside analysis; perfect
transcription is not a prerequisite for another directed passage.

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

## Next implementation

Read [docs/18_local_structure_comparison.md](docs/18_local_structure_comparison.md)
for the fixed comparison, limits, timing regressions and next decision. The
original detector/control is documented in [17](docs/17_local_structure.md).
Read [20](docs/20_listening_feedback.md) before designing evaluation: preserve
the perceived non-change at 0:21, continuing-but-less-prominent voice near 2:04,
local instrumentation change near 1:19 and broad breakdown near 1:01–1:05.
Selections are subjective scope judgments; written qualifications matter.

Read [21](docs/21_change_episodes.md) for the implemented response hypothesis and
its failures. The concrete next experiment is **continuous role context**:

1. Extract the existing all-stem before/after descriptors at every supported
   grid boundary and scale, independently of which episodes exceed thresholds.
2. Review signed trends and anchor/scale sensitivity on the same four excerpts,
   especially continued voice near 124s and perceived non-change near 21s.
   Preserve raw notes and uncertainty; do not equate RMS share with leadership.
3. Reuse verified cached features/grid/audio in a new immutable package. Human
   annotations enter evaluation only. Retain missing evidence and support times;
   thresholds and audibility floors still require whole-track offline context.
4. Test silent/constant inputs, proportional stem scaling and candidate-selection
   independence. Distinguish descriptor availability from semantic interpretation.
5. Decide whether stable context supports a small graded-emphasis directing study.
   Physical transition extents and perceptual salience remain separate unsolved
   tasks; do not promote on density, retrospective best overlap or tests alone.

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
  context and role/importance interpretation still need separate evaluation.

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
- New frozen response experiment: `outputs/reviews/change-episodes-01/`.
  All 259 responses and 24 full curves are retained; no production detector changed.
- Legacy full-song review/native original WAV: `outputs/reviews/structure-review-03/`.
  WAV hash: `7d800828527fb8812dc6433361d1d72ad1f7cdba368b9c51dce4abc954f7a5c1`.
- Blank editor: `outputs/reviews/section-editor-02/`; directed preview:
  `outputs/reviews/directed-review-01/`; saved-plan replay: `directed-replay-01/`
  under the same reviews root.

Verify manifests before reuse. Avoid dumping huge story/recurrence JSONs into
context; inspect targeted fields or the small numeric cache. Audio and `outputs/`
are gitignored: a fresh clone alone cannot reproduce these experiments. Check
missing files and disk space first; do not silently rerun expensive separation
or erase artifacts to make room.

## Code and commands

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

- Lead (Astra when available): ambiguous diagnosis, musical assumptions,
  experiment design, acceptance criteria, integration and final review.
- Terra: most bounded implementation, tests, reports and UI under an explicit
  contract. User explicitly requests this delegation for implementation work.
- Luna: optional mechanical/documentation tasks with easily checked outputs.
  Sol or the lead can take deeper implementation reasoning if needed.
- Start with one worker; add another for independent work. Give disjoint files,
  focused context, required inputs, outputs and checks. Do not have the lead
  rewrite the delegated task in parallel or send the whole conversation.
- Inspect actual code and evidence, including failures. Model choice, passing
  tests and agent agreement do not establish musical correctness. Delegation
  alone does not guarantee savings; retries and coordination count too.
- Ask the user about meaningful musical ambiguity/artistic preference on prepared
  short examples, not to debug every iteration or repeat existing annotations.

No custom orchestration platform is needed. Preserve dirty/untracked work; no
commits, resets, cleanup or broad rewrites were requested.

## Later direction and documentation ownership

After the immediate comparison: improve transition extents/non-dip hypotheses,
recognize recurring identities with variations, validate on held-out music, and
feed reliable evidence into the visual director. Small visual passage experiments
can proceed alongside analysis. Full-song saved-plan/render remains the end goal,
not permission to implement every milestone at once.

- Product intent: [README.md](README.md).
- Milestones/working loop: [docs/01_roadmap.md](docs/01_roadmap.md).
- Architecture/open decisions: [docs/02_architecture.md](docs/02_architecture.md).
- Detailed inventory/history: [docs/03_working_state.md](docs/03_working_state.md).
- Current comparison evidence: [docs/18_local_structure_comparison.md](docs/18_local_structure_comparison.md);
  frozen-control details: [docs/17_local_structure.md](docs/17_local_structure.md).
- Guided listening and optional-feedback contract: [docs/19_listening_examples.md](docs/19_listening_examples.md).
- Received listening judgments and next-step implications: [docs/20_listening_feedback.md](docs/20_listening_feedback.md).
- Response-episode design, findings and next context experiment: [docs/21_change_episodes.md](docs/21_change_episodes.md).
- Feedback semantics: [docs/15_section_feedback.md](docs/15_section_feedback.md),
  [docs/16_structural_evaluation.md](docs/16_structural_evaluation.md).

Historical inventories and phase numbers do not override this checkpoint. If
files or fresh checks disagree, investigate and update the handoff rather than
assuming an old result holds. After the next task, update **this checkpoint,
next step, failures and artifact pointers** and the relevant experiment document.
Keep prior results attributable; do not create another competing status index.
