# Four-case development benchmark

## Decision and purpose — 2026-09-15

The human authorized consolidating the proposed three-lead workflow and selected
Team 1 as integration owner. One lead owns the evidence inventory, evaluation
design and final review. Terra implements the bounded package; Luna checks source
evidence. A second strong model is reserved for challenging a concrete experiment.
Current assignment/status remains in `CONTINUE.md`.

This package answers: **what must the next method demonstrate beyond the acoustic
measurements we already have?** It combines existing evidence with explicit future
evaluation criteria. It does not create another song interpretation, a new detector,
a full-song knowledge graph, or a request to repeat listening annotations.

## Four requirements

| Existing case | Intended understanding from existing feedback | Current evidence | Missing capability |
| --- | --- | --- | --- |
| Within passage, 16–26s | No recognizable musical development in the reviewed context. | Accompaniment-level/spectral differences remain measurable across neighboring windows. | Avoid converting ordinary acoustic variation into a meaningful event. |
| Drum entry, 74–85s | Added kick/snare make the continuing idea fuller; bass is uncertain. | Drum RMS rises at all 15 fixed observations; share rises at 14. Existing section spans retain chorus identity across the variation. | Attribute an arrangement development while preserving continuity. Preserve the raw note's verse/chorus naming disagreement. |
| Verse ending, 119–132s | Reported lead spoken/sung vocal gives way to laughter; vocal sound continues. | Vocal RMS/share decrease while activity remains; the section reference retains verse-2 identity. | Identify changing vocal behavior without equating presence, loudness or acoustic share with musical function. |
| Breakdown, 57–70s | A broad drop/stop and recovery into another passage. | Multiscale summaries mix disappearance and recovery; wider windows change sign. | Explain the temporal sequence and its extent, without inventing precise endpoints. |

The rows are separate requirements, not ordered importance classes. The listener's
`subtle` selection is explicitly qualified in the raw note. The 15 correlated
anchor/scale observations per case are robustness context, not 15 independent
human judgments. Success at a narrow capability is not complete understanding.

## Reference and evaluation contract

- Retain exact raw notes/selections, excerpt and prompted focus bounds, fixed
  anchor, all 2/4/8-beat support windows and source fingerprints.
- Preserve intersecting section spans, original labels, identity IDs, unspecified
  certainty and analyst interpretation. Different/blank motifs do not create
  negative identity examples. The 57–70s audition contains the separately annotated
  61.368411–64.855320s transition; it is not an endpoint annotation. No numerical
  timing tolerance has been validated.
- Expected distinctions and rubrics are authored requirements derived from
  existing feedback, not automatic predictions or new human labels. Future
  outcomes start unscored. Missing/ambiguous output is unresolved; contradiction
  is failure. An all-abstaining method cannot pass by avoiding wrong assertions.
- Keep notes, labels, target descriptions and scoring criteria out of candidate
  inputs. Future experiments must declare exactly which audio/numeric features
  they see and freeze outputs before reference comparison. Feature paths or
  timestamps alone do not establish this separation.
- These are already-seen development cases. Guided explanations may have
  influenced judgments; disclose any tuning on them. They are not held-out data.
- Inspect actual candidate assertions and cited evidence. Keyword matches or an
  LLM's agreement with the notes are insufficient. Report each requirement's
  outcome and uncertainty; no aggregate quality score or automatic promotion.

### Worked review cases

These are authored checks on the evaluation design, not new model observations.

| Future output condition | Review consequence |
| --- | --- |
| All candidate fields are null or explicitly unknown. | All four requirements unresolved; abstention earns no positive capability claim. |
| At 21s, a method reports measured acoustic differences and leaves meaningful development unknown. | Unresolved on perceived development; the acoustic observation may still be correct. |
| At 79s, a method supports added drums and continuing material but does not name a verse/chorus or assert bass entry. | Eligible for the arrangement/continuation requirement; it need not know the reference's naming disagreement. |
| At 124s, a method reports only falling vocal RMS with continuing activity. | Correct narrow acoustic evidence, unresolved vocal behavior; no laughter or leadership credit. |
| At 124s, a suitable independent method supports laughter replacing spoken/sung behavior while voice continues. | Eligible for vocal-behavior credit; musical leadership and verse-ending interpretation remain separate. |
| A method explicitly denies any breakdown in the broad-transition example. | Contradicts the development reference; fail that requirement. |
| A method proposes supported transition timing but does not call it ground truth. | Judge the broad temporal account; exact endpoint accuracy remains unscored until a valid tolerance/reference is defined. |
| A fluent answer copies the evaluation note or sees semantic case IDs/target text as input. | Cannot establish independent candidate capability; report contamination and do not award validation credit. |

The semantic example IDs are useful for evaluator joins but reveal the target.
The output template therefore uses neutral IDs; the benchmark retains their
mapping on the reference side. A blank template is not itself an inference runner
or proof of blind evaluation.

## Implementation scope

`experiments/build_development_benchmark.py` builds deterministic JSON/text from
frozen role-context and structural-reference outputs and original feedback. It
verifies consumed hashes, retains a builder snapshot and refuses existing output
directories. No audio loading, model execution or production detector changes.
Focused tests: `tests/test_development_benchmark.py`.

This is an evaluation artifact, not a product page. The earlier
`baseline-explanation-01` cards remain frozen and unaccepted as a comprehension
aid. No browser or visual work is needed for this task.

## Next bounded experiment decision

This was the 2026-09-15 selection. Its experiment is complete in doc 30. On
2026-09-20 the human challenged the resulting laughter-specific priority, and
Team 1 parked that branch. The four requirements above remain useful; current
work emphasizes reusable arrangement/continuity evidence across passages rather
than requiring vocal-event labels. Follow `CONTINUE.md` for the live queue.

Prioritize **changing vocal behavior despite continuing voice**; the other rows
remain development controls. This is a specific representational gap: RMS/activity
cannot distinguish the reported behavior even when it detects a level change.
Another magnitude or neighboring-anchor stability sweep would mostly repeat the
completed role-context/MuQ work.

Before implementing a new extractor, compare at most two concrete ways to obtain
temporal vocal-behavior evidence against the existing RMS/activity baseline.
Require declared inputs, feasible resource use, label-free extraction and the
ability to abstain. A method that only restates loudness or consumes the reference
note does not qualify. Recognizing laughter would still not prove reduced
leadership or an approaching verse ending; preserve the familiarity caveat.
Do not broaden this into a survey of all music-understanding models.

The review must produce one testable proposal or a concrete no-go reason, stating
the narrow capability, output time support, failure modes, comparison conditions
and stopping rule. A second Astra or Opus can review that proposal if needed;
no standing counterpart assignment is created. New model setup, paid compute and
remote audio transfer are not part of this benchmark task.

## Validation and handoff

**Complete — accepted by Team 1, 2026-09-15.** Final package:
`outputs/reviews/development-benchmark-02/`. It contains `benchmark.json`, the
manual evaluation report, a blank candidate output schema, a builder snapshot
and a manifest. `development-benchmark-01` remains a rejected frozen draft.

Lead review of draft 01 found overlapping pass/unresolved rules, criteria that
expected hidden reference wording, semantic IDs in the purportedly isolated
template, and unpinned source manifests. Terra corrected these before building
02. The final output schema has neutral IDs and explicitly makes no inference
isolation claim. The final transition criterion permits evidence-backed timing
proposals while leaving exact endpoint accuracy unvalidated.

Terra's focused command:

```bash
.songviz/venv/bin/python -m pytest -q tests/test_development_benchmark.py
```

Result: **4 passed, 1 existing ddtrace warning**. Checks include exact raw notes
and blank outputs, raw-source tampering, simultaneous source/manifest tampering,
overwrite refusal and deterministic output. Compilation and diff checks passed.

The lead independently inspected final code, report, rubric and JSON. Direct
comparison verified all four raw note/selection pairs, excerpt/focus/anchor
fields, **60 verbatim observation records**, and **8 intersecting span records**
against their frozen sources. Raw listener notes also match role-context's
embedded notes exactly. The lead verified both accepted source-manifest hashes,
five consumed-source hashes, three output hashes and the saved/current builder
hash; independently replayed all **five package files byte-for-byte** from the
snapshot in a temporary directory. Drum RMS/share counts and continuing-vocal
activity claims in the table above were checked against all 15 observations.
Luna supplied the supporting evidence audit; its summary's erroneous wording
that the audition “exactly matches” the transition interval was corrected by
direct source inspection. The interval distinction above is authoritative.

Reproduce under a fresh directory (the builder refuses an existing destination):

```bash
.songviz/venv/bin/python experiments/build_development_benchmark.py \
  --repo . --output outputs/reviews/development-benchmark-replay-01

# To replay the exact accepted builder instead of current source:
.songviz/venv/bin/python \
  outputs/reviews/development-benchmark-02/build_development_benchmark.py \
  --repo . --output outputs/reviews/development-benchmark-snapshot-replay-01
```

The example replay destinations above were not created during acceptance;
acceptance replays used temporary directories. Existing source packages are
required; this fixed-source builder does not regenerate missing experiments.

| Accepted file | SHA-256 |
| --- | --- |
| `manifest.json` | `684b38314825c61c743459337631085abee87e12bd8157aab2b6fef952ff4f97` |
| `benchmark.json` | `223e19591f21dcb009a801447094f8754a3e1e4cd947647afe4334839b0eac79` |
| `future_candidate_output_template.json` | `dc804079f58d691eb4fb728657893fd7563296f455c13312babbaf4901a0cb10` |
| `report.md` | `f7e7183eeba7cc438e0f48835787c23b2a7fb895741995e74a5589f64c0385bf` |
| `build_development_benchmark.py` | `b3ad74b5d98f501287fab5c5b1a8cb9ed0b77bb5ebae3411756fdaa83cc47f26` |

No full-suite/browser/model rerun was needed for this isolated builder. No audio
was loaded/uploaded and no new semantic result or production policy is claimed.
The accepted result is a reproducible development reference and manual rubric;
future capability evaluation still needs a declared candidate, isolated inputs
and saved outputs. The subsequent bounded vocal-behavior method comparison and
negative YAMNet screen are complete in [doc 30](30_vocal_event_probe.md).
Follow `CONTINUE.md` for the current next step; this benchmark remains frozen.
