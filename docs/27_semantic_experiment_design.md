# Semantic experiment design — Team 3

**Status: complete — awaiting integration (revision 3, 2026-09-14).**
Revision 1 (SHA-256 `88939f3c473c35ad359afc2c2114a5686cc2551cf63c17bb496d7d5a8592c363`)
was returned by Team 1 with findings S1–S5
([doc 24](24_music_understanding.md#team-1-integration-review--2026-09-14)).
Revision 2 (SHA-256 `f13be1df36a3dbc74ce870528eab205f0e93a6c346e71a678813324ce1236119`)
was returned with scoring findings R1–R3
([doc 24](24_music_understanding.md#team-1-revision-review--2026-09-14)); the
eight-clip vocal-manner question and all other protocol are unchanged.
Lead: Claude Opus 5 (user reassigned Team 3 from Fable 5.1 on 2026-09-14).
Worked outcomes were traced by Sonnet workers (§8). Integration
owner: Team 1. Design only: no audio model run, no audio cut, no annotation
requested, no code or artifact written.

**Question.** Can a semantic audio model on the original mix produce a claim about
this song that (1) matches the user's note, (2) the stem baseline and MuQ cannot
express, and (3) is demonstrably tied to the audio rather than to familiarity,
clip position or prompting?

**Decision this test can make:** proceed to a separately designed held-out
evaluation (SUPPORT), stop semantic probing with this model/protocol on this
song (REFUTE), or neither (UNRESOLVED, with a reason code). No accuracy, rate or
generalization claim follows from any outcome.

## Changes in revision 3

| Finding | Resolution |
| --- | --- |
| R1 prompted completion earned unprompted credit | New grade **T5-full-A** (§4.2): every predicate of the credited transition (affirmed voice-attributed laughter, affirmed lead delivery, valid ordering) must come from Stage A claims alone. G6 now requires T5-full-A. Stage B may still make a T5-full match for G5, but it cannot upgrade Stage A. Worked case W13. |
| R2 polarity, uncertainty, ablation exemption | Every claim records `polarity` ∈ affirmed / negated / uncertain, with frozen cue lists (§4.3). Only affirmed claims earn target credit, count as contradictions or cause ablation conflict. Explicit absence ("no laughter is audible") is never a conflict. Uncertainty has its own gates: G4u → UNRESOLVED(uncertain-candidate), G10c → UNRESOLVED(ablation-uncertain). G10a/G10b are now source-agnostic, matching the manipulation check's "Can you hear laughter?"; an affirmed laughter sample in the no-vocals clip is a conflict. Worked cases W15–W18. |
| R3 invalid timestamps | Times are validated against the actual clip duration before mapping (§4.3 step 1). Invalid or untimed claims keep their semantic mapping with `time_status` ≠ valid, cannot establish ordering by time, cannot satisfy G7 or a padded match, and cannot create a mismatch. Each core candidate is evaluated separately with its own valid time; there is no inheritance of the earliest time (§4.4, G8). Worked cases W19–W21. |

G7's reason is renamed UNRESOLVED(core-time-unverified), covering untimed and
invalid times. G8/G8b now evaluate each valid core candidate. No clip, prompt,
decoding, retry, manipulation-check or rubric-target change was made.

## Changes from revision 1

| Finding | Resolution in revision 2 |
| --- | --- |
| S1 contradictory precedence | One ordered gate list (§5): protocol → completeness → scorer consistency → positive control → decisive target → its controls. Each gate either ends with exactly one outcome or passes. Absent benefit is only read after scoreability gates; missing times and scorer disagreement cannot produce REFUTE. |
| S2 minimum run vs padding | The decision uses only the verse-ending controls. Required set **R** (8 clips) is fixed; the other six padded clips form descriptive set **D**, which never changes the outcome and may be omitted. Padding is scored only on the shared core interval with explicit matching, tolerance and missing-time rules (§4.4); omission in a longer clip is not treated as invented audio. |
| S3 manipulation validity | Drum ablations removed. The vocal ablation is valid only with an independent blind human audibility check of the two ablation clips (§3.4), collected before any model output is viewed; otherwise the control is **unverified** and SUPPORT is unreachable. Existing evidence cannot establish it, so **control validity is currently UNRESOLVED.** Stem-activity disagreements are logged as proxy conflicts and never affect the outcome. |
| S4 continuation loophole | The positive control requires an explicit entry/drop proposition (T1 or T7), never continuity. T2 is credited only with T1 in the same answer; T3 only after the positive control. Constant "same part, no change" fails at G3 (worked case W7). T2/T3 are descriptive only. |
| S5 underspecified targets and execution | T5 has frozen full/partial grades with lexicons (§4.2); only unprompted T5-full can support. Loss of leading delivery (note-supported, T6a) is separated from "supporting role" (analyst inference, T6b, never credited). Seed/order, token budget, retry eligibility and prompt, stage aggregation, and segmentation/mapping are frozen in §3–4. Preregistration completes only when G0 is satisfied. |

Also changed: the second-AVU requirement is removed. The decision rests on the
one documented missing capability — vocal-manner change at the verse ending
(doc 20, implication 3); other targets are reported, not decisive.

## 1. Evidence boundaries

The song is Gorillaz, *Feel Good Inc.* (source SHA-256 `657af933…`). It is very
widely released and its laughter is a recognizable signature. Hiding the title
does **not** remove pretrained familiarity; only anchoring and ablation controls
bear on whether a claim is audio-driven.

| Kind | Content |
| --- | --- |
| Human notes (`benchmark/feedback/listening-examples-01.json`) | 74–85s `local`: "continued the verse after this transition with the drums, like the kick and the snare at least, and maybe bass as well (not sure though)". 16–26s `none`: "couldn't really recognize any difference"; "pre-verse where there's no, like, voice". 119–132s `subtle`: "more of a spoken verse before … There is still some voice, but it's not really the guy singing. It's just, like, a laugh, which occurs on the final part of the verse"; "Maybe it's a little bit more than just a subtle change"; familiarity acknowledged. 57–70s `broad`: breakdown that "comes to a stop" before "a transition to, like, the chorus phase" (doc 20 summary). |
| Analyst hypotheses (not labels) | Laughter lies in the later part of 119–132s; the 0:21 `other` decrease is imperceptible in context; the voice "moves to a supporting role". |
| Missing labels | Any timing inside excerpts; vocal behavior elsewhere; second rater; counterfactual labels for manipulated audio; unfamiliar-listener judgments. |
| Baseline/MuQ-reachable (**B**) | Stem enters/leaves/louder/quieter/fuller/thinner; spectral pattern change; "something changes near t" (MuQ distance); "resembles an earlier passage" (MuQ recurrence). At the verse ending vocal RMS and share fall at all 15 anchor/scale combinations while activity stays 1 (doc 23). |

Operational terms: *acoustic contribution* = signed stem descriptor change at
declared anchors/scales; *audible vocal behavior* = manner of voice production a
listener hears and whether it changes; *musical role/importance* = listener's
organization (leading delivery, continuing part, breakdown). Current labels cannot
establish timing, rater agreement, laughter as transcription, or section names
(the user says "verse" at 74–85s where the earlier annotation says chorus).

## 2. Stimuli

All clips are sample-exact cuts, exported without metadata.

| Set | Clip id | Source | Interval (s) | Role in decision |
| --- | --- | --- | --- | --- |
| R | `drum-entry.core` | original mix | 74–85 | G3 positive control (T1) |
| R | `within-passage.core` | original mix | 16–26 | G11 false alarm |
| R | `verse-ending.core` | original mix | 119–132 | G4–G7 target |
| R | `transition-extent.core` | original mix | 57–70 | G3 positive control (T7) |
| R | `verse-ending.pre` | original mix | 115–132 | G8 anchoring |
| R | `verse-ending.post` | original mix | 119–136 | G8 anchoring |
| R | `verse-ending.resum` | bass+drums+other+vocals stems summed | 119–132 | G10a ablation reference |
| R | `verse-ending.novocals` | bass+drums+other stems summed | 119–132 | G10b–c ablation |
| D | `{drum-entry,within-passage,transition-extent}.{pre,post}` | original mix | core start −4 / core end +4 | descriptive only |

Stems: `outputs/Gorillaz - Feel Good Inc (featuring De La Soul)/stems/`. Omitting
any R clip gives UNRESOLVED(incomplete); omitting D changes nothing.

## 3. Frozen execution rules

### 3.1 Order, naming, isolation

Sort the ids to be run in ASCII order; `order = random.Random(20260914).sample(ids,
len(ids))` (Python 3); name files `clip_01.wav…` in that order. The id↔file mapping
is hashed and withheld from scorers until §4.3 step 2. Each clip gets a fresh model
context; Stage B is sent in the same context as Stage A.

### 3.2 Prompts and decoding

Stage A (saved before Stage B is sent):

> Listen to this audio clip. Describe what you hear and how it develops over
> time. Give clip-relative times in seconds for anything that starts, stops or
> changes. If you are unsure, say so instead of guessing.

Stage B:

> 1. Is there a moment where the arrangement changes (parts entering, leaving,
>    becoming fuller or thinner)? If yes: when, and which parts. If no, say "no change".
> 2. Do any voices appear? For each: how is it produced (for example singing,
>    rapping, speaking, or another vocal sound), and does that manner change? When?
> 3. Does any part move between leading and supporting roles? When, and which?
> 4. Does the clip feel like one continuing musical part, or a move from one
>    part to another? If a move, when?
> Answer "unsure" where appropriate.

Decoding: greedy (`do_sample=False`), `max_new_tokens=400` per stage, framework
seed 0; record model revision and runtime. Stage B never mentions laughter and is
asked of every clip, so prompted claims are identifiable as Stage-B-only.

### 3.3 Retry (mechanical, per clip, at most once)

Applied by script before any person or scorer reads replies. A clip is eligible
if, in either stage, the reply is empty or stopped at `max_new_tokens` without
end-of-sequence, or if neither stage matches
`\b\d{1,3}(?:\.\d+)?\s*(?:s|sec|secs|second|seconds)\b|\b\d{1,2}:\d{2}\b`.
The retry reruns both stages in a fresh context with this sentence appended to
Stage A: "Write each observation as `[start–end s] description` or
`[time s] description`." Retry replies replace the originals for scoring; both
are archived. No other retry, prompt change or re-run is permitted.

### 3.4 Manipulation check (precondition for SUPPORT; not collected by Team 3)

One listener, having seen no model output, hears `verse-ending.resum` and
`verse-ending.novocals` in a pre-recorded coin-flip order and answers per clip:
"Can you hear laughter? yes / no / unsure." **Valid** only if resum = yes and
novocals = no. This is a new label about manipulated audio, not a repeat of the
four listening examples. Collecting it is Team 1's choice; without it G9 caps
the outcome at UNRESOLVED.

### 3.5 Recognition probe

After both scorers' records are hashed, send each core clip to a fresh context:
"Do you recognize this recording? If so, name it." A positive answer sets
`familiar=true`. It changes no gate; a SUPPORT is then reported as
"audio-anchored on a familiar song".

## 4. Scoring

### 4.1 Rubric (T = target, C = contradiction, U = unscoreable; **B** = baseline-reachable)

| Case | Targets | Contradictions | Unscoreable |
| --- | --- | --- | --- |
| drum-entry | **T1 (B)** drums/kick/snare/percussion enter, come in or start. T2 the same part continues through that entry (credited only with T1 in the same answer). | New section/part begins; drums leave. | Bass; section names. |
| within-passage | T3 one continuing part / no change (credited only if G3 passed). | Claim that a section/part changes, or that an instrument fully enters or leaves (the core clip is the excerpt, so untimed claims count). | Thinner/quieter/timbral description; "no vocals". |
| verse-ending | **T4 (B)** voice continues but quieter/less prominent. **T5** vocal-manner change (§4.2). T6a leading sung/rapped/spoken delivery ends or recedes while some voice continues. T6b voice becomes a supporting/background role (analyst inference; recorded, never credited). | Voice absent after the change; no change in the clip; manner changes toward singing/lead delivery. | Lyrics; "outro"; section names. |
| transition-extent | **T7 (B)** energy drops / parts drop out / breakdown / stop. T8 that drop leads into or prepares a following part (credited only with T7 in the same answer). | Build-up without a drop; no change. | "Chorus"; endpoints. |

### 4.2 T5 grades (case-insensitive word-stem lexicons)

- **Laughter terms:** laugh, laughs, laughing, laughter, chuckle, giggle, cackle, "ha ha".
- **Lead-delivery terms:** sing, sung, singing, rap, rapped, rapping, speak, spoken, speaking, talk, narrate, lead vocal, vocal line, verse vocal.
- **Other-vocalization terms:** shout, yell, ad-lib, exclamation, scream, grunt, breath, vocal sound(s), vocalization, non-melodic.

**T5-full:** a laughter term explicitly attributed to a voice or person (voice,
vocalist, singer, rapper, man, woman, he, she, someone), preceded by a
lead-delivery term — earlier in time, or linked by an ordering phrase (then,
after, before, turns into, gives way, replaced by, followed by, stops and).
**T5-partial:** an other-vocalization term with such ordering, or a
voice-attributed laughter term without preceding lead delivery. A laughter term
attributed to a sample, effect or other non-voice source is U. Any other claim
is not T5.

T5 is graded on the aggregated answer (§4.3 step 3), not on one atomic claim,
using **affirmed** claims only (§4.3 step 1). The graded claim is the laughter or
other-vocalization claim. Its preceding lead delivery may be a separate affirmed
claim in the same answer if either (a) both claims have `time_status = valid`
and the lead claim's start is earlier than the graded claim's reference time, or
(b) the reply text links the two with an ordering phrase. Invalid or untimed
times never establish order under (a).

- **T5-full / T5-partial** use claims from either stage.
- **T5-full-A** is a T5-full match whose graded claim, lead-delivery claim and
  ordering evidence all come from Stage A. Stage B claims can never supply a
  missing predicate for T5-full-A. `unprompted` means T5-full-A.
- Each T5 match keeps its graded claim's own time and `time_status`. When several
  matches exist, each is a separate candidate (§4.4); none inherits another's time.

### 4.3 Segmentation, mapping, stage aggregation

1. **Segmentation (blind).** Each scorer receives only `clip_NN` and raw replies,
   splits each into atomic claims (one proposition, one source, one time;
   conjunctions split), and records verbatim span, stage, question number and
   parsed time: "at 5s" → point 5; "5–8s" → [5,8]; "around 5s" → point 5, hedged;
   "after 5s" → [5, clip end]; "towards the end" or no number → untimed; m:ss
   converts to seconds.
   - **Time validation.** Let D be the clip's exact duration from its manifest
     sample count (R clips: 11, 10, 13, 13, 17, 17, 13, 13s). A point t is valid if
     finite and 0 ≤ t ≤ D. An interval [a,b] is valid if finite and
     0 ≤ a ≤ b ≤ D, and a = b is treated as a point. Anything else (reversed,
     negative, beyond D, non-numeric) gets `time_status = invalid`, and its
     source times are left null. Otherwise `valid` or `untimed`. Source time =
     clip start + clip time, computed only for valid times.
   - **Polarity.** The proposition is recorded in affirmative form ("laughter is
     audible"), with `polarity`:
     *negated* if the claim denies it (no, not, without, absent, none, never,
     stops entirely);
     *uncertain* if it hedges its existence (unsure, not sure, uncertain, unclear,
     may, might, maybe, possibly, perhaps, could be, seems to be, I think);
     otherwise *affirmed*. Descriptive qualifiers ("faint", "in the background",
     "sounds like laughter") and time hedges ("around 6 s") stay affirmed.
     Negation and uncertainty together → uncertain.
   Segmentation files are hashed before step 2.
2. **Mapping.** Scorers receive the id mapping and §4.1–4.2, and assign each claim
   a target/contradiction/U id, T5 grade (full, full-A, partial), and proxy
   conflict (§4.5). No other judgment is added. Only affirmed claims can match a
   target or a contradiction. **Negation boundary:** a negated claim whose
   proposition is the positive counterpart of a §4.1 contradiction is re-recorded
   as that affirmed contradiction, only in cases whose §4.1 row lists it (in
   within-passage, "no change" is T3, not a contradiction). Two such pairs exist: "voice is present after
   the change" negated = "voice absent", and "the clip changes" negated = "no
   change". Any other negated claim earns nothing and only cancels an equal
   affirmed target proposition (step 3).
3. **Aggregation per clip.** Stage A and B form one answer. A target is matched
   if some claim matches it and the answer holds no contradiction for that case.
   A negated claim in either stage whose proposition equals an affirmed
   target claim's proposition cancels that target. A matched target's time is the
   earliest *valid* time among its affirmed claims (for descriptive targets).
   T5 follows §4.2 and keeps per-candidate times. Targets that need other claims
   (T2, T5, T8) are graded across the answer.
4. **Two scorers** (humans or model workers) perform steps 1–3 independently.
   Their agreement measures scoring consistency, not auditory truth.

### 4.4 Padding match on the shared core

**Core candidates** are the T5-full-A matches in `verse-ending.core` whose graded
claim has `time_status = valid` (source time therefore in [119,132]). A
candidate's time c is its graded claim's source interval start or point. Each
candidate is evaluated separately.

For `verse-ending.pre` and `.post`, consider the affirmed T5-full matches (either
stage) whose graded claim has a valid time with source time or interval
intersecting the shared core [119,132]. These are *valid padded matches*.
Material in the padding, and invalid, untimed, negated or uncertain claims, are
ignored for matching. Reference time = interval start or point. For candidate c,
each padded clip is:

- **matched** — some valid padded match has its reference time within **±2.0s**
  (≈4.5 grid beats) of c, or has an interval of ≤8s containing c;
- **mismatched** — at least one valid padded match exists and none is matched;
- **unverified** — no valid padded match exists (omission, scope change, or only
  invalid/untimed/uncertain claims).

Candidate c is **anchored** if both padded clips are matched, **refuted** if
either is mismatched and it is not anchored, and **unverified** otherwise.

The tolerance tests anchoring, not timing accuracy; no human timing exists.

### 4.5 Records and proxy conflicts

Claim record fields: `clip, stage, question, span, proposition, polarity,
claim_type, clip_duration_s, clip_start_s, clip_end_s, time_status,
source_start_s, source_end_s, hedged, target_id, grade, is_contradiction,
proxy_conflict`. A **proxy conflict** is a presence
claim disagreeing with frozen stem activity. It is diagnostic only: thresholded
activity cannot prove auditory absence, so it is never a contradiction and never
affects a gate.

## 5. Decision procedure (ordered; stop at the first gate that assigns an outcome)

| Gate | Passes when | Otherwise |
| --- | --- | --- |
| G0 protocol | This document's hash, clip manifest (hashes, intervals), order mapping and manipulation-check order are recorded before any output is generated; no deviation beyond §3.3. | UNRESOLVED(protocol) |
| G1 complete | Every R clip has Stage A and B replies after §3.3. | UNRESOLVED(incomplete) |
| G2 scorers | Run G3–G11 on each scorer's records. Identical outcomes → adopt it. Different → Team 1 re-maps only the differing records by rubric text alone and reruns G3–G11 on the adjudicated records; if rubric text cannot decide a differing record whose alternatives yield different outcomes → stop. | UNRESOLVED(scoring) |
| G3 positive control | T1 matched in `drum-entry.core` or T7 matched in `transition-extent.core`. | REFUTE(positive-control) |
| G4 candidate | `verse-ending.core` has an affirmed T5 match at any grade. | go to G4u |
| G4u uncertainty | (reached only if G4 fails) `verse-ending.core` has an uncertain claim that would be T5 at any grade if affirmed. | UNRESOLVED(uncertain-candidate) if yes; REFUTE(no-candidate) if no |
| G5 grade | Some T5 match is T5-full. | UNRESOLVED(partial-target) |
| G6 unprompted | Some T5 match is T5-full-A. | UNRESOLVED(prompted-only) |
| G7 core time | At least one core candidate exists (T5-full-A with valid time, §4.4). | UNRESOLVED(core-time-unverified) |
| G8 anchoring | Some core candidate is anchored. | go to G8b |
| G8b | (reached only if G8 fails) Every core candidate is refuted. | REFUTE(unanchored) if yes; UNRESOLVED(anchoring-unverified) if no |
| G9 control valid | Manipulation check exists and is valid (§3.4). | UNRESOLVED(control-unverified) if missing; UNRESOLVED(control-invalid) if not valid |
| G10a reference | `verse-ending.resum` has an affirmed laughter-term claim, whatever source it is attributed to and whatever its time status. | UNRESOLVED(control-inconclusive) |
| G10b ablation | `verse-ending.novocals` has no affirmed laughter-term claim, whatever source it is attributed to (voice, sample, effect) and whatever its time status. Negated claims pass. | REFUTE(ablation-conflict) |
| G10c ablation uncertainty | `verse-ending.novocals` has no uncertain laughter-term claim. | UNRESOLVED(ablation-uncertain) |
| G11 false alarm | `within-passage.core` has no contradiction. | UNRESOLVED(false-alarm) |
| — | All gates passed. | **SUPPORT** |

Ordering rationale: controls that can only remove credit (padding, ablation) are
required for SUPPORT but never needed to reach REFUTE. Missing, invalid or
uncertain evidence (G4u, G7, G8b unverified, G10c) and scorer disagreement (G2)
end in UNRESOLVED. REFUTE follows only from affirmed, scoreable answers: no basic
audio evidence (G3), no candidate of any polarity (G4u), valid-time evidence
against every candidate's anchoring (G8b), or affirmed laughter conflicting with a
valid human manipulation check (G10b).

**Stopping:** one run, no prompt iteration; stop at the outcome. Any follow-up
model or run must target the named UNRESOLVED reason. SUPPORT only licenses
designing the held-out evaluation (§9).

## 6. Descriptive report (never changes the outcome)

T1/T2/T3/T4/T6a/T6b/T7/T8 matches with stage and time; D-set padding
classifications using §4.4 with each case's target; proxy conflicts; recognition
flag; both scorers' raw disagreement counts.

## 7. Worked outcomes (hypothetical answers, not model outputs)

Defaults unless stated: G0–G1 pass; scorers agree. `drum-entry.core` A: "[4 s]
kick and snare come in; the same groove continues" (T1, T2). `within-passage.core`:
"[0–10 s] a steady guitar figure, no vocals, no change" (T3, no contradiction).
Verse padded clips repeat the core description with laughter at source 125
(pre "[10 s]", post "[6 s]"). The manipulation check is valid. A retry (§3.3) is
assumed taken wherever eligible and to return the same content.

| # | Scenario (differences from defaults) | Gate path | Outcome |
| --- | --- | --- | --- |
| W1 valid success | Verse core A: "[0–5 s] a man raps in a spoken style; [6 s] he stops and laughs repeatedly." Resum: "[6 s] a man laughs". Novocals: "[0–13 s] instrumental loop, no voice". | G3 ✓ G4 ✓ G5 ✓ G6 ✓ G7 (125) ✓ G8 ✓ G9 ✓ G10a ✓ G10b ✓ G10c ✓ G11 ✓ | **SUPPORT** |
| W2 Stage-B-only success | As W1, but Stage A says only "[0–13 s] rap vocal over bass and drums"; Stage B Q2: "rapping, then he laughs at 6 s". | G4 ✓ G5 ✓ G6 ✗ | **UNRESOLVED(prompted-only)** |
| W3 all abstentions | Every clip, both stages: "unsure" (retry returns the same). | G1 ✓ G3 ✗ | **REFUTE(positive-control)** |
| W4 missing timestamps | Verse core A: "a man raps, then he laughs near the end"; B gives no times; retry still untimed. | G4–G6 ✓ G7 ✗ | **UNRESOLVED(core-time-unverified)** |
| W5 scorer disagreement resolved | Verse core A: "[0–5 s] rap; [6 s] a laughing sample plays". Scorer 1 maps T5-full (→ SUPPORT); scorer 2 maps U (→ REFUTE(no-candidate)). §4.2: laughter attributed to a sample is U. | G2 adjudicates U → G3 ✓ G4 ✗ | **REFUTE(no-candidate)** |
| W5b disagreement rubric cannot decide | Verse core A: "[0–5 s] a man raps; [6 s] his rap gives way to a laughing loop". Scorer 1 reads "loop" as the voice's laughter (T5-full); scorer 2 as a sample (U). §4.2 does not define "loop". | Outcomes differ; rubric text cannot decide | **UNRESOLVED(scoring)** |
| W6 required control removed | Run omitted `verse-ending.novocals` for runtime reasons. | G1 ✗ | **UNRESOLVED(incomplete)** |
| W6b descriptive clips removed | As W1, D omitted. | as W1 | **SUPPORT** |
| W7 constant continuation | Every clip: "one continuing part, no change" (retry returns the same). | G3 ✗ (T2 requires T1; no T7) | **REFUTE(positive-control)** |
| W8 unverified bleed | As W1, but no manipulation check was collected. | G9 ✗ | **UNRESOLVED(control-unverified)** |
| W8b bleed audible | As W1, but the check gives novocals = yes. | G9 ✗ | **UNRESOLVED(control-invalid)** |
| W8c model hears laughter after vocal removal | As W1, but novocals: "[6 s] faint laughter in the background"; check valid. | G10b ✗ | **REFUTE(ablation-conflict)** |
| W9 complete valid failure | Verse core A: "[0–13 s] rap vocal continues, getting quieter after 6 s" (T4 only). | G3 ✓ G4 ✗ | **REFUTE(no-candidate)** |
| W10 partial target | Verse core A: "[0–5 s] rapping; [6 s] vocal ad-libs and shouts". | G4 ✓ G5 ✗ | **UNRESOLVED(partial-target)** |
| W11 timed mismatch | As W1, but pre: "[0–3 s] a man raps; [4–5 s] he stops and laughs" (source 119–120, reference 119, Δ = 6 s, interval does not contain 125) and no other laughter claim. | G8 ✗ G8b ✓ | **REFUTE(unanchored)** |
| W12 false alarm | As W1, but within-passage core A: "[5 s] a new section begins as the guitar drops out". | G11 ✗ | **UNRESOLVED(false-alarm)** |

Revision 3 cases (same defaults; W1's clips unless stated):

| # | Scenario | Gate path | Outcome |
| --- | --- | --- | --- |
| W13 A laughter-only + B preceding rap (R1) | Core A: "[6 s] a man laughs." Core B Q2: "At 0 s the man raps before laughing at 6 s." | G4 ✓ G5 ✓ (full via A+B) G6 ✗ (A alone is partial; B cannot supply lead delivery for T5-full-A) | **UNRESOLVED(prompted-only)** |
| W14 complete transition wholly in A (R1) | Core A: "[0–5 s] a man raps; then [6 s] he laughs." Core B Q2: "unsure". | G4 ✓ G5 ✓ G6 ✓ (all predicates in A) G7 ✓ … G11 ✓ | **SUPPORT** |
| W15 explicit absence in ablation (R2) | Novocals: "[0–13 s] instrumental; no laughter is audible." | G10a ✓ G10b ✓ (negated) G10c ✓ G11 ✓ | **SUPPORT** |
| W15b explicit absence in core (R2) | Core A: "[0–13 s] a man raps; no laughter is audible." No other vocal-manner claim. | G3 ✓ G4 ✗ G4u ✗ | **REFUTE(no-candidate)** |
| W16 uncertain core candidate (R2) | Core A: "[0–5 s] a man raps; around 6 s, I'm unsure whether the man laughs." No affirmed T5 anywhere. | G4 ✗ G4u ✓ | **UNRESOLVED(uncertain-candidate)** |
| W17 uncertain ablation (R2) | Novocals: "[6 s] maybe some laughter?" | G10a ✓ G10b ✓ G10c ✗ | **UNRESOLVED(ablation-uncertain)** |
| W18 affirmed laughter sample in ablation (R2) | Novocals: "[6 s] a laughter sample is audible." | G10b ✗ (source attribution is not an exemption) | **REFUTE(ablation-conflict)** |
| W19 out-of-bounds padded interval (R3, Team 1's example) | Core A: "[0–5 s] a man raps; then [13 s] he laughs" (valid, source 132). Post (D = 17): "[0–3 s] a man raps; then [12–20 s] he laughs": 20 > 17, invalid, no valid padded match. Pre: "[0–3 s] a man raps; then [17 s] he laughs" (source 132, matched). | Candidate 132: pre matched, post unverified → G8 ✗ G8b ✗ | **UNRESOLVED(anchoring-unverified)** |
| W20 reversed core interval (R3) | Core A: "[0–5 s] a man raps, then [9–6 s] he laughs." Full-A via the ordering phrase, but the time is invalid, so there is no core candidate. | G4–G6 ✓ G7 ✗ | **UNRESOLVED(core-time-unverified)** |
| W21 valid and invalid core candidates together (R3) | Core A: "[0–5 s] a man raps, then [25 s] he laughs; [6 s] he laughs again." The [25 s] claim is invalid (D = 13), full-A via the ordering phrase. The [6 s] claim is valid, full-A via the earlier valid rap time. One candidate: 125. | G7 ✓ (125 only) G8 ✓ … G11 ✓ | **SUPPORT** |
| W21b invalid time cannot rescue a mismatch (R3) | As W21, but pre: "[0–3 s] a man raps; then [4–5 s] he laughs" (source 119, Δ = 6, valid) and "[30 s] he laughs" (invalid, D = 17). Post matched. | Candidate 125: pre mismatched (the invalid claim is ignored) → refuted; the only candidate → G8 ✗ G8b ✓ | **REFUTE(unanchored)** |

W5b is intentional: a rubric gap ends in UNRESOLVED rather than scorer majority.
Team 1 may close such gaps before G0; after G0 no rule may be added.

## 8. Validation performed

- The lead checked every note quotation against
  `benchmark/feedback/listening-examples-01.json`, and checked the baseline and
  MuQ facts against doc 23/24 and `music-representation-02/report.md`.
- A Sonnet worker traced W1–W12 (16 scenarios) through §4–§5 literally. Its
  outcomes matched §7 for all 16, and it confirmed the gate chain is exhaustive:
  every path ends in exactly one outcome.
- It found one defect, now fixed. Atomic segmentation split lead delivery and
  laughter into separate claims, so per-claim grading would have made W1, W4, W6b,
  W8–W8c, W10 and W11 stop earlier at G4/G5. §4.2 now grades T5 across the
  aggregated answer. The worker also checked that the §7 defaults are not
  retry-eligible and hold no contradiction.
- Limit of this check: the worker could not hide §7's outcome columns while
  reading the scenarios, so the trace was not blind to the expected answers.
  This checks consistency, not an independent derivation.
- Documentation only; no tests, audio or model execution apply.

Revision 3 (2026-09-14):

- A fresh Sonnet worker traced all 22 scenarios (W1–W21b), blind to the expected
  results. It read only §0–§6 plus a separate file containing just the defaults
  and scenario text. Its stopping gate and outcome matched §7 in **22/22** cases.
  Its probes confirmed: Stage B cannot supply a T5-full-A predicate; negated or
  uncertain claims cannot earn credit or cause REFUTE(ablation-conflict); invalid
  or untimed times cannot satisfy G7, create a padded match or produce a
  mismatch; every gate path ends in exactly one outcome; the defaults break no rule.
- The worker found two gaps, now fixed without changing any §7 outcome. The core
  candidate time c for an interval was undefined; it is now the interval start
  (§4.4). The boundary between a negated claim becoming a contradiction and it
  only cancelling a target was implicit; it is now closed in §4.3 step 2.
- Limit: a few scenario texts carry analyst hints (for example "valid"). The
  worker was told these were not authoritative and to re-derive them. Scorer and
  model agreement is consistency, not auditory truth. Documentation only.

## 9. S1–S5 and R1–R3 status and remaining limits

- S1, S2, S4, S5: resolved by §3–§5 as tabled above.
- R1: closed by T5-full-A (§4.2, G6; W13/W14). R2: closed by recorded polarity,
  the negation boundary, G4u/G10c and source-agnostic G10a/G10b (W15–W18).
  R3: closed by duration validation and per-candidate anchoring (§4.3–4.4, G7/G8;
  W19–W21b).
- Deliberately open: the non-voice-source category in §4.2 (sample, effect, …)
  is not a closed list. A borderline noun such as "loop" ends at
  UNRESOLVED(scoring) (W5b). Team 1 may close it before G0.
- Polarity cue lists are frozen English lists; a hedge outside them is scored
  affirmed. Scorer disagreement over such a hedge goes to G2.
- S3: resolved procedurally; **control validity cannot be established from
  current evidence and remains UNRESOLVED** until a §3.4 check exists.
- Familiarity: a negative recognition probe does not show unfamiliarity. The user,
  who would likely perform the manipulation check, knows the song and may expect
  laughter in bleed; record this as a limit, not a gate.
- The resum clip controls separation artifacts, not lost mix interactions; the
  stems are a model separation, not a studio multitrack.
- Excerpts were chosen around suspected changes; padding weakens but does not
  remove this cue. The `none` clip is the only false-alarm pressure.
- The B boundary is a judgment frozen here; a future voice/laughter detector would
  move T5 into B.
- SUPPORT rests on one proposition in one familiar song. This is intended: it is
  only permission to design held-out work. Greedy repeated outputs are
  reproducibility, not confirmation (doc 24).

**Future held-out evaluation (distinct; only after SUPPORT):** material unlikely
to be in pretraining; labels collected before any model output; at least two
raters; timed vocal-manner judgments; explicit non-change excerpts; manipulation
checks for every ablation; rubric frozen beforehand; none of the four development
cases reused.

**Execution constraints:** outputs go to a new `outputs/reviews/semantic-probe-01/`
with a manifest binding clip hashes, intervals, order seed and mapping, prompts,
decoding, model revision, raw and retry replies, both scorers' segmentation and
mapping files, the manipulation check and this document's hash. Any runtime
accepting ≤17s audio with text suffices; doc 26 governs feasibility, currently
recorded by Team 1 as no demonstrated route.
