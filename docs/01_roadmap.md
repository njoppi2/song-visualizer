# Roadmap

For the current checkpoint and next concrete task, start at
[CONTINUE.md](../CONTINUE.md). This document owns milestones and the working loop;
historical progress notes below are not a separate execution queue.

Revised 2026-09-10 after the product-direction clarification.

## Direction

Make listening more compelling through visuals that follow meaningful musical
events. Develop reliable evaluation alongside short, reviewable visual examples.
Preserve the existing pipeline and improve it incrementally.

Target: a command-driven automatic visual director that chooses what to emphasize
or omit, how to depict it, and when to change or reuse a visual motif based on the
song's structure and local events. Fixed instrument effects are building blocks,
not the final experience. The README owns product intent; `02_architecture.md`
owns the proposed analysis/director/renderer split and unresolved design choices.

Implementation status is separate from musical validation. Passing unit tests,
matching algorithm-derived references, and producing plausible audio do not by
themselves establish fidelity or a better listening experience.

Reduced representation remains a useful analysis path and listening diagnostic.
Compare audio features, discrete events, and their combination where relevant;
accurate transcription of every note is not a prerequisite for useful visuals.

## Existing foundation

The former phases 0–3 delivered ingest/render, stems, lyrics, and heuristic story
analysis. Former phase 4 already includes drums, vocals, bass, sonification, and
MIDI export. The repo also has a dashboard, evaluations, five benchmark songs,
and exploratory structure/phrase diagnostics. These capabilities exist; their
musical quality and generalization need validation.

`03_working_state.md` contains the implementation inventory and dated results.
`06_reduced_representation.md` preserves the original reduction design, not the
current task queue. The milestones below supersede the old phase ordering.

## Milestone 1 — Reproducible baseline and first review (complete)

Technical review package: `outputs/reviews/restart-02/index.html`. Reference audit
and reproduction notes: `07_restart_review.md`. First user review received and
preserved in `benchmark/feedback/restart-02.json`. Its concrete acceptance criteria
are a clap-along pulse without unexplained timing changes and recognizable drum
indicators that respect quiet intervals. Triage and the first controlled candidate
are in `08_rhythm_feedback.md`; reference uncertainties remain explicitly open.

Deliver a small review package using existing local songs and cached artifacts.

- Inventory inputs, dependencies, cached stems, analyses, and rendered outputs.
- Select two or three short passages with contrasting behavior: repetition,
  instrument entry/exit, and a build/release or quiet transition. Initial choices
  can come from cached signals but remain unverified until reviewed.
- Capture current behavior before changing extraction. Provide synchronized
  original audio, baseline video, relevant signal overlays, and reduced audio
  only where it helps explain the behavior.
- Record source audio hash, excerpt times, code revision and local changes,
  configuration, dependency/model versions where available, and cache provenance.
  Mark unknown provenance explicitly; old artifacts are not current-code proof.
- Audit reference provenance and consistency. Separate independent annotations
  from algorithm-derived diagnostics. Check MIDI alignment before trusting timing
  scores. Do not silently revise references to fit predictions.
- Connect section evaluation to the batch benchmark and label activity metrics
  as activity metrics, distinct from event timing and transcription fidelity.
- Save timestamped user observations with expected behavior, uncertainty, and
  links to the exact comparison. Turn objective corrections into regression cases;
  keep artistic preferences distinct from musical facts.

Exit: the baseline can be reproduced, omissions are visible, and the user has
reviewed a small package that establishes the first concrete acceptance criteria.
No recollection of earlier problems is needed. This milestone does not require a
complete annotation dataset or perfect extraction.

## Milestone 2 — Direct one passage across a meaningful musical change

Prototype implemented: `outputs/reviews/directed-review-01/index.html` now compares
an automatically generated rule-based plan with a fixed mapping over 130–178s.
The saved contract, validator, generic renderer and replay path are implemented;
127 focused tests pass and replayed videos match byte-for-byte with current code.
User review of musical direction is pending. See `10_directing_prototype.md`.
Do not restart the plan contract from scratch; use the prototype and its explicit
limitations to decide the next policy or treatment change.

Earlier timing experiment: `outputs/reviews/rhythm-review-01/index.html`. It isolates pulse
timing and separates percussion indicators; it is a diagnostic comparison, not an
accepted artistic design or a production beat-tracker replacement. The second
review's written notes support regular-pulse alignment and percussion legibility.
The user clarified that the hard-to-follow pulse was cached, not regular. Preserve
the raw export in `benchmark/feedback/rhythm-review-01.json` and its qualifications
in `08_rhythm_feedback.md`. Timing approval is not approval of a fixed visual
mapping. Targeted Milestone 3 reliability work supports the directing experiment.

First artistic artifact: `outputs/reviews/visual-passage-01/index.html`, a 22-second
opening using the exact reviewed events. It is a fixed-effects vocabulary study,
not proof of automated direction. The user's response clarified the larger goal;
it is not an acceptance or rejection of every animation. Retain the artifact and
verification in `09_visual_passage.md`, but do not make further animation polish
the gate for starting the director.

- Select a passage with meaningful before/change/after context. Feel Good Inc
  around 130–178s is a candidate from existing reviews, not a certified structural
  reference; confirm useful events against evidence before locking the excerpt.
- Specify a minimal saved plan for focus, omissions, treatment choices, motif
  continuity, and evidence-linked transitions. Preserve source timing separately.
- Implement a small planner and plan-driven renderer path using existing signals
  and reusable visual treatments. A deterministic baseline can exercise the
  contract; compare an LLM-assisted proposal when justified, without committing
  to a provider or building a general agent orchestration platform.
- Render the directed sequence and an always-on fixed-mapping comparison from
  identical evidence/audio. A hand-authored intent example must be labeled and
  cannot count as the automatic planner's output.
- Review whether focus follows the music, omissions make room for key moments,
  and changes/returns preserve a coherent visual story. Ask about musical
  emphasis and continuity, not only whether an individual effect looks pleasant.

Exit: an inspectable, validated plan produces a user-reviewed sequence showing
selective focus and context-sensitive treatment across a meaningful change, with
documented benefits/remaining problems and automated timing/visibility checks.
Replay the saved plan without another model call. A rendered file, a plausible
explanation, or another agent's approval alone does not complete this milestone.

## Milestone 3 — Targeted musical reliability

Current structural prerequisite completed: explicit beat-grid input, persisted
timing provenance and isolated full-song regeneration with both cached and
reviewed grids under the same code (`12_structure_grid.md`). This does not certify
sections or novelty. The boundary/recurrence listening review and two isolated
logic fixes are now implemented (`13_structure_review.md`): fuse agreeing
detector timestamps once, and treat missing novelty history as unknown.
Role-independent ordered phrase comparisons supply review candidates, not
confirmed section identities. Next, collect targeted listening feedback before
further tuning or using these predictions as directing ground truth.

The user now prefers defining their own sections. The blank annotation editor
(`14_section_annotation.md`) supports free labels and independent layers without
requiring one flat segmentation or a predetermined hierarchy. Use those examples
to clarify meaningful structural levels before locking detector targets. Earlier
algorithm-proposed questions are optional, not the only allowed reference format.

First free-form annotations received (`15_section_feedback.md`): 19 spans,
including short transitions and low/high-energy variants sharing a chorus
identity. The separate structural/evaluation representation is now implemented
(`16_structural_evaluation.md`), including pattern-versus-arrangement evidence,
local-versus-historical phrase context and explicit unsupported cases. No new
automatic section/identity/transition classifier is claimed. Next, develop
shorter-scale change/transition proposals alongside longer recurrence evidence;
16/32-beat windows alone miss the support needed for several user-marked spans.
Formal hierarchy and timing tolerance remain open; no additional annotation is
a prerequisite for that next development experiment.

The first short-scale proposal experiment is now implemented
(`17_local_structure.md`): separate local-change points and bounded energy-dip
intervals, with original-audio comparison and full development diagnostics.
It adds useful proposals inside long legacy sections but misses the two
low/high-energy chorus changes and three of five interpreted transition
intervals. Do not promote it over the current detector. Next, compare separate
channel thresholds and sustained activity evidence against this frozen control;
retain missed chorus variations as regressions and inspect proposal burden.

- Prioritize failures from the previews. Initial candidates are beat/bar timing,
  instrument activity, repetition, and meaningful transitions.
- Test evidence needed for directing choices: entrances/exits, returns, changes
  relative to established patterns, and uncertainty. Neither a loud onset nor a
  section boundary alone establishes a musical surprise.
- Preserve measured attack times separately from quantized or templated rhythms.
  Compare drum template and onset paths; a plausible imposed groove must not be
  reported as verified transcription.
- Separate acoustic repetition from inferred functional roles so a role error
  does not automatically determine which sections count as repetitions.
- Attach evidence, source, and uncertainty to interpretations. Heuristic scores
  are not calibrated probabilities; permit unknown or ambiguous outcomes.
- Evaluate one hypothesis at a time on tuning examples and independent holdout
  songs/passages. Reserve holdouts before tuning; once used to guide a change,
  treat them as development data and reserve new examples for the next check.

Exit per change: repeatable improvement on its intended behavior, reported
holdout performance and regressions, and a relevant visual/listening comparison.
Do not claim generality from repeated tuning on the same five songs.

## Milestone 4 — Expand demonstrated value

Extend the proven directing loop toward the full-song, single-command product:
whole-song motif continuity, plan validation/replay, cache reuse, and explicit
failure behavior. Resolve CLI, optional planner configuration, cost/data-sharing
controls and necessary user overrides using the open questions in architecture.
Full-song support is a product requirement; its quality is not established by
the first local passage.

Add harmony, timbral descriptors, richer note visuals, or detailed vocal animation
when an accepted visual use case needs them. Compare raw-feature, event-based,
and combined approaches before introducing a mandatory dependency. Revisit
separation models only when evidence identifies separation as the limiting step.

Exit per feature: a concrete user-facing improvement with reproducible evidence
and documented runtime/resource costs.

## Working loop and delegation

Work in small increments within the active milestone. Each task specifies its
hypothesis, allowed files, observable acceptance criteria, validation, and artifact.
The lead agent owns architecture, task selection, integration, and final review.

Suggested routing, subject to models actually available in the session:

| Owner | Scope |
| --- | --- |
| Lead agent | Ambiguous diagnosis, musical assumptions, experiment design, integration, final quality judgment |
| Luna | Bounded inventories, documentation, mechanical edits with clear checks |
| Terra | Contained implementation work with explicit behavior and relevant tests |
| Sol | More involved implementation or review when a smaller model struggles |
| User | Musical ambiguity and artistic preference on short, prepared examples |

This is a starting routing policy, not a measured cost optimum. Current model
positioning is documented in the [official model catalog](https://developers.openai.com/api/docs/models).
Verify session availability before delegation; do not infer billing or savings
from a model name. Track usage where exposed and account for retries/review work.

Use one worker initially; add a second only for independent work with disjoint
file ownership. Give workers focused context. Keep tightly coupled or blocking
work with the lead. Review results against evidence, not another agent's approval.
Escalate a task when its assumptions or failed checks require deeper reasoning.

Reuse cached computation with verified provenance and test short passages before
expensive full-song runs. Do not build a separate orchestration framework for
this restart. Record decisions and feedback in the repo so future sessions can
resume without asking the user to repeat them.

Use waveform/spectrogram close-ups with predicted events overlaid when checking
timing. A vision-capable worker can flag visible mismatches; numerical checks own
precise residuals, and human listening owns musical ambiguity and preference.
Whole-song images establish structure, while short close-ups expose transients.
Neither agreement between agents nor matching separated-source peaks establishes
perceptual quality or independent transcription accuracy.

## First execution batch

1. Inventory reusable artifacts and record the baseline provenance.
2. Audit the selected references and fix the batch section-evaluation omission.
3. Prepare two or three baseline excerpts and their relevant diagnostics.
4. Ask the user to watch/listen and identify the first mismatch or preference.
5. Choose one improvement, implement it, compare it, and retain the result as a
   regression case or an explicitly subjective design preference.

Ask for user input when an artifact makes the question concrete or a missing
choice blocks progress. Continue independent technical work while feedback is
pending; do not mark a perceptual milestone validated without that feedback.
