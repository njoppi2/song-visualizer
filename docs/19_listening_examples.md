# Guided listening examples

The user could not interpret the internal policy names in comparison 18. The
next requested work is a small, understandable listening experience using that
existing evidence. It does not implement the queued multiscale episode detector.

The page presents one short example at a time with original audio, a local
timeline, replay and optional feedback. Algorithm names are kept out of the
listening task. Explanations are revealed separately as hypotheses, and all
questions can be skipped. No answer is preselected or inferred from navigation.

The four lead-curated development examples are:

| Listening excerpt | Focus | Why it is included |
| --- | --- | --- |
| 74–85s | Drum activity around 79s | Explain evidence near the already annotated chorus intensity change |
| 16–26s | Accompaniment decrease around 21s | Show a within-passage proposal whose perceptual significance is unknown |
| 119–132s | Accompaniment decrease around 124–125s | A nearby candidate does not yet explain the full annotated verse subdivision |
| 57–70s | User's 61.368–64.855s transition interval | Demonstrate that the measured energy dip covers only part of a larger transition |

Focus bands are curated listening prompts, not inferred change extents. The last
band uses the user's existing annotation; no relabeling is required. The `other`
stem is described as a mixture of accompaniment sounds, not an identified
instrument. Text is bound to inspected event IDs/timestamps and stem evidence;
the builder refuses mismatched evidence rather than attaching text to a nearby
event in a different run.

Optional feedback records perceived change size and free text about visual
emphasis/timing. It is subjective guidance, not correctness labels or new
section boundaries. Downloads carry the example set ID, review payload hash,
source/audio hashes and per-example IDs. Missing answers remain null. Drafts
are isolated by review hash; existing annotation/review drafts are untouched.

Builder: `experiments/build_listening_examples.py`; page:
`experiments/templates/listening_examples.html`; browser check:
`experiments/check_listening_examples.cjs`. Parent:
`outputs/reviews/local-structure-comparison-02/`. All analysis and audio are reused.

The builder verifies parent outputs/snapshots and source/audio provenance, saves
source snapshots and hashes all generated payload/page outputs. Every build
requires a new output directory. This is a curated explanation package, with
no new detector training, separation, section inference or runtime model calls.

Ready: [listening-examples-01](http://127.0.0.1:8770/listening-examples-01/), under
`outputs/reviews/listening-examples-01/`. The native audio server on 8770 was
available at handoff. If absent, start `python -m songviz.review_server` using
the repository venv. Reproduction always uses a fresh output directory:

```bash
.songviz/venv/bin/python experiments/build_listening_examples.py \
  --out outputs/reviews/listening-examples-NEW
SONGVIZ_PLAYWRIGHT_MODULE=/home/njoppi2/.npm/_npx/04489aa25f0b609a/node_modules/playwright \
  node experiments/check_listening_examples.cjs \
  http://127.0.0.1:8770/listening-examples-01/
```

Verified nine tests covering explanation/evidence binding, rejected altered
inputs/ranges and safe embedded text. All 15 source/snapshot/output hashes and
the template-to-page derivation matched. Page size is about 38 KB; original
audio is referenced rather than duplicated. Browser smoke verified playback,
bounded stop, navigation, optional answers and clear choice, JSON export, draft
persistence/isolation, mobile width, and no page/console errors. Mobile layout
was also visually inspected. Existing analysis tests were not rerun for this
presentation-only follow-up.

The next analysis step remains the multiscale episode hypothesis in CONTINUE.md
and doc 18. Optional feedback can guide perceived importance and visual emphasis;
the lack of a new export does not block development from existing evidence.

Update: all four responses were received and verified on 2026-09-11. Raw export:
`benchmark/feedback/listening-examples-01.json`; interpretation, qualifications
and implications: [20_listening_feedback.md](20_listening_feedback.md).
