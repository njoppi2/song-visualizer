# Private Music Flamingo Modal wrapper

This is a private feasibility wrapper. An ordinary local import of `app.py`
does not import Modal, authenticate, call a cloud API, download weights, or
load the model. With `SONGVIZ_MODAL_ENABLE_APP=1`, the deployable endpoint
functions are defined at module scope. Modal worker imports use its container
arguments marker to define the same global functions even though they do not
inherit that local deployment variable; the fake-SDK builder exists only for
offline tests.
The first contract accepts exactly one base64-encoded uncompressed PCM WAV
(`audio/wav`), at most 20 MiB and 60 seconds, plus one non-empty prompt. It
rejects unknown fields, unsupported types, malformed WAV, unsupported channel/
sample-rate values, excess size, and excess duration before inference.

`POST /analyze` and `GET /health` both use Modal proxy authentication; neither
route is public and Modal rejects unauthenticated traffic before a worker is
started. Health deliberately returns only `status`, `ready: false`, and schema
version—it is not a model-ready claim and contains no model, account, or
credential data.

The `health` function uses its own deliberately small image containing only
`fastapi==0.115.14`; it cannot import Torch, Transformers, SoundFile, the model,
or a GPU runtime. The GPU `analyze` function alone fixes the documented model
identifier and revision, uses one L4 worker at most (`max_containers=1`,
`min_containers=0`, no buffer), a 60-second scale-down window, and a 300-second
request/startup ceiling. There is no volume, queue, background task, external
log sink, or permanent warm worker.
A model/processor load happens only on the first authorized `POST /analyze` in
each container, then remains in that container's process-local memory for later
requests until Modal scales it down. This avoids repeated 16.5-GB loads during
the 60-second warm window; it is not a persistent volume/cache and disappears
when the container does.

Successful analysis responses retain the pinned model and validated input
metadata and add non-sensitive per-request execution metadata:
`inference_elapsed_seconds` (wall-clock time including any first-request load)
and `model_cache_miss` (whether that request loaded the container-local runtime).

`client.invoke_once` is the intended cold-start policy: it can retry only the
authenticated `GET /health` on HTTP 503 within a 120-second overall deadline,
then emits exactly one `POST /analyze`. It never retries a POST, whose execution
state could be ambiguous after a transport failure. If that POST returns Modal's
long-running HTTP 303 response, capture its `Location` header and call
`client.follow_result_redirect`: it permits one same-origin `GET` to the result
URL and never issues another POST or sends proxy credentials to a foreign origin.

## Local checks

No Modal package, credentials, network, model weights, or audio file is needed:

```bash
python -m pytest -q tests/test_modal_music_flamingo.py
```

## Deployment gate (do not run without human authorization)

The human must first provide Modal access, a small spending limit, the selected
GPU/region and cost-stop threshold, and separately authorize the feasibility
call. This pinned public checkpoint is downloaded anonymously on the first
authorized request; no Hugging Face account or secret is needed for this route.
If a later revision is gated, do not add a token ad hoc: make a new scoped
security decision and use a Modal Secret rather than a shell, file or source
tree.

Only after those gates, from the repository root, explicitly opt in:

```bash
SONGVIZ_MODAL_ENABLE_APP=1 modal deploy -m experiments.modal_music_flamingo.app
```

Use a Modal proxy token at invocation time (for example through Modal's
authenticated client/CLI); do not paste it into this repository. Before sending
any source-song audio, obtain the separate audio-upload authorization described
in doc 24. Record only redacted config, timings, observed lifecycle behavior,
cost, source hashes, and approved outputs in the later evidence package.

## Teardown

After the bounded run, stop the exact deployed app (this is destructive and
cannot be restarted):

```bash
modal app stop songviz-music-flamingo-private
```

Do not create or retain a Modal Volume for this first route. Revoke the proxy
credential in Modal if the authorized run requires that response.
