# Semantic probe 02 — frozen control amendment

Status: **local preregistration amendment, 2026-09-15**. This document is the
only design change for `semantic-probe-02`. It incorporates
[`27_semantic_experiment_design.md`](27_semantic_experiment_design.md) exactly
except where this amendment expressly replaces its `verse-ending.novocals`
stimulus and manipulation-check binding. All prompts, eight R clip identifiers,
order seed, decoding, retry, scoring, gate order, targets and prohibitions stay
unchanged.

## Why an amendment is necessary

The original `semantic-probe-01` no-vocals clip was the sum of Demucs bass,
drums and other stems. Its blind check was invalid: the listener heard laughter
in both its resum-positive and no-vocals clips. That package remains frozen and
cannot be repaired in place.

Team 1 then made one local, bounded 119–132s separation with Kimberley Jensen's
Mel-Band RoFormer two-stem vocal model. The model's `other` output passed a new
neutral-name listener check: candidate-other = no laughter, resum-positive =
yes laughter. This only validates a listener-facing counterfactual for this
excerpt. It does not establish a truth stem, separation accuracy, song meaning,
or an authorization to use Music Flamingo.

## Replaced stimulus

`verse-ending.novocals` retains its id, original-song interval (119–132s),
duration (13s), position in the seeded order and all downstream rubric/gate
roles from doc 27. Its source recipe is replaced:

| Field | Frozen value |
| --- | --- |
| source kind | `external_control` |
| control name | `kim-mel-band-roformer-other` |
| source file | `outputs/reviews/stem-separation-control-01/separation-attempt-02/gorillaz_119_132_original_(other)_vocals_mel_band_roformer.wav` |
| source SHA-256 | `14d418a527f2882441467b0053d46bd6dc112bcc66852829cc1c7de3eab73612` |
| file-local read | frames 0–573,300 (0–13s) |
| original-song alignment | 119–132s |
| format | 44.1 kHz, stereo, PCM-24 WAV, exactly 573,300 frames |

Every other R stimulus keeps the exact source recipe in doc 27. In particular,
`verse-ending.resum` remains the four-Demucs-stem sum and is the positive
condition for the manipulation check.

## Manipulation-check binding

The completed control record is
`outputs/reviews/stem-separation-control-01/candidate-manipulation-check.json`,
SHA-256 `b9f09d7da82f81d125cb61ca8fefb86762834966deae3020a803d6269a3e96db`.
Its `/dev/urandom`-recorded neutral order mapped Clip A to the candidate other
output and Clip B to the resum-positive output. The raw listener response was
`A: no, b: yes`; normalized result: candidate-other = no,
resum-positive = yes. This passes doc 27's polarity condition, now bound to the
exact replacement WAV above.

`semantic-probe-02` must fingerprint both that record and the replacement WAV,
and must produce a new eight-clip manifest and clip hashes. If either hash,
format, duration or alignment differs, G0 is not complete and no model output
may be generated. The existing listener record is valid because it was collected
without model output and uses exactly the output that the new package binds.

## Stage-B transport binding

The private endpoint is stateless across HTTP requests. For every clip, Stage A
is one authenticated request containing its frozen Stage-A prompt and audio.
Stage B is a second authenticated request containing the same audio plus an
explicit reconstructed transcript: the exact Stage-A user message, the exact
Stage-A model response as an assistant message, then the frozen Stage-B user
message. This is the transport implementation of doc 27's “same context” rule;
it is not a new prompt and it does not add any human text. The runner records
both raw replies. If the frozen retry condition applies, it reruns that complete
two-turn transcript once in a fresh context, as doc 27 requires.

## Limits and next authorization boundary

The resulting package is a local, no-run preregistration only. It does not
upload source or derived audio, call a remote endpoint, spend money, execute
Music Flamingo, promote Mel-Band RoFormer for general separation, or process a
full song. A separate explicit human authorization remains required before any
external audio transfer or paid model execution.
