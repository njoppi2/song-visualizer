# Guided listening feedback — 2026-09-11

All four examples have an explicit selection and nonempty written feedback.
Raw export is preserved byte-for-byte at
`benchmark/feedback/listening-examples-01.json`, from the user's
`/home/njoppi2/Downloads/songviz-listening-examples-feedback.json`.

- Export SHA-256: `f69b14f0343d0ebe4af12d37f66e88c76aa5d8e7673f7529345333589d5c47b6`.
- Review payload SHA-256: `2cdd54d4ca38c3d1af58f0d69e3b8767944a67a5f1d22afadd676e1754093c98`.
- Example set: `listening-examples-01`; exported at `2026-09-11T14:12:17.276Z`.
- Set ID, all example IDs, schema, selection values, source/audio hashes and
  review hash were checked. All 15 package source/snapshot/output records and
  the referenced original audio verified. No missing or duplicate answers.

## What the user reported

| Example / excerpt | Selection | Written meaning and qualifications |
| --- | --- | --- |
| Drum entry / 74–85s | `local` | The same musical part continues in a fuller arrangement as kick/snare enter. Bass may also enter, but the user explicitly is unsure. |
| Within passage / 16–26s | `none` | No recognizable change in this excerpt. The user describes a preparatory/early verse passage, with tentative terminology. |
| Verse ending / 119–132s | `subtle` | Leading spoken/sung vocals give way to laughter; voice remains audible but loses much of its leading musical role. The note qualifies the selection as possibly more than subtle and acknowledges prior familiarity may help identify an outro. |
| Transition extent / 57–70s | `broad` | A clearly noticeable breakdown removes energy and feels like a stop before a transition into the chorus. No new precise interval endpoints are supplied. |

These descriptions are user listening judgments, not independently verified
transcription or stem-source assignments. In the drum-entry note the user says
"verse", whereas the earlier free-form annotation places this region in the
chorus. Preserve both records; do not silently rename existing identity groups.
The useful consistent observation is continuation with added instrumentation.

The voice/laughter description conveys the user's perceived change in function.
Do not convert wording such as "no melody" into a measured zero-pitch claim or
an automatic laughter label. The within-passage response is a perceptual
non-change example in this review context, not proof that acoustic features are
constant. The broad breakdown judgment does not validate the narrow dip's exact
timing or turn the entire 57–70s audition into a transition annotation.

## Implications for the next experiment

The interpretation below is the lead's reasoning from the feedback, kept
separate from the raw export:

1. Retain these four distinct development cases when evaluating proposed change
   episodes: fuller instrumentation within a continuing idea, an acoustic proposal
   with no perceived change, reduced prominence/function of continuing vocals,
   and a broad breakdown with preparation/recovery.
2. Preserve continuous and overlapping evidence per stem and scale. A stronger
   feature contrast cannot by itself determine perceived importance or the size
   of a visual response. In particular, avoid automatically emphasizing the
   0:21 proposal just because its accompaniment-level contrast is large.
3. The 2:04 example needs evidence about changing vocal behavior, beyond presence
   or RMS alone. Use this as a known missing capability in the next episode
   representation/evaluation. Acoustic proxies may be examined, but do not claim
   to identify speech, singing or laughter from these four responses alone.
4. For the broad transition, compare extent hypotheses against the existing
   human span and narrow dip while retaining uncertainty in endpoints. The new
   broad selection provides perceived scope, not a new timing tolerance.

The follow-up experiment is documented in [21_change_episodes.md](21_change_episodes.md).
At intake, the queued next step was an explicit multiscale change-episode hypothesis.
These responses refine its evaluation and explain why it must keep event cause,
duration and perceived importance distinct. This intake implements no detector,
salience classifier, hierarchy or visual policy, and promotes no variant.

No further user clarification is needed to proceed. Do not ask for the same four
examples again. New listening requests should address a concrete remaining
ambiguity after an inspectable experiment. These are four guided examples from
development music; the shown explanations may have influenced judgments, and
there is no export field showing whether they were read before answering.
