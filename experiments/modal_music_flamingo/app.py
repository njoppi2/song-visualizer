"""Private Modal application definition; inert unless explicitly opted in.

Importing this module neither imports Modal nor loads/downloads model weights.
Set SONGVIZ_MODAL_ENABLE_APP=1 only for a separately authorized deployment.
"""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Callable

from .schema import MODEL_ID, MODEL_REVISION, health_payload, response_payload, validate_request

APP_NAME = "songviz-music-flamingo-private"
GPU = "L4"
MAX_CONTAINERS = 1
MIN_CONTAINERS = 0
SCALEDOWN_WINDOW_SECONDS = 60
REQUEST_TIMEOUT_SECONDS = 300
# Frozen by docs/27_semantic_experiment_design.md §3.2.  Keep this public
# module-level setting so an offline execution-contract check can bind the
# runner's declared decoding settings to the source that will be deployed.
MAX_NEW_TOKENS = 400


_model_runtime: tuple[Any, Any] | None = None


@dataclass(frozen=True)
class InferenceResult:
    text: str
    elapsed_seconds: float
    cache_miss: bool
    stopped_at_max_new_tokens: bool | None = None


def _load_pinned_runtime() -> tuple[Any, Any]:
    """Import ML dependencies and load weights only on the first worker request."""
    import torch
    from transformers import AutoProcessor, MusicFlamingoForConditionalGeneration

    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = MusicFlamingoForConditionalGeneration.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, device_map="auto", torch_dtype=torch.bfloat16
    )
    return processor, model


_runtime_loader: Callable[[], tuple[Any, Any]] = _load_pinned_runtime


def _get_pinned_runtime() -> tuple[tuple[Any, Any], bool]:
    """Return the lazy, process-local runtime; Modal scale-down discards it."""
    global _model_runtime
    if _model_runtime is None:
        _model_runtime = _runtime_loader()
        return _model_runtime, True
    return _model_runtime, False


def _set_runtime_loader_for_test(loader: Callable[[], tuple[Any, Any]]) -> None:
    """Inject a no-ML loader for focused local tests only."""
    global _runtime_loader
    _runtime_loader = loader


def _clear_runtime_cache_for_test() -> None:
    """Clear process-local state so focused tests never retain fake runtimes."""
    global _model_runtime, _runtime_loader
    _model_runtime = None
    _runtime_loader = _load_pinned_runtime


def _conversation_for_request(
    audio_path: str, prompt: str, prior_response: str | None = None, followup_prompt: str | None = None,
) -> list[dict[str, object]]:
    """Build the exact one- or two-turn transcript sent to the processor."""
    conversation: list[dict[str, object]] = [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "audio", "path": audio_path},
    ]}]
    if prior_response is not None and followup_prompt is not None:
        conversation.extend((
            {"role": "assistant", "content": [{"type": "text", "text": prior_response}]},
            {"role": "user", "content": [{"type": "text", "text": followup_prompt}]},
        ))
    return conversation


def _infer_pinned_model(
    audio_bytes: bytes, prompt: str, prior_response: str | None = None, followup_prompt: str | None = None,
) -> InferenceResult:
    """Load and invoke the exact revision only inside an authorized worker call."""
    # A temporary, per-request file avoids a persistent volume/cache and is removed
    # before the request returns.  It is never created during local contract tests.
    started = time.perf_counter()
    with tempfile.NamedTemporaryFile(suffix=".wav") as audio_file:
        audio_file.write(audio_bytes)
        audio_file.flush()
        (processor, model), cache_miss = _get_pinned_runtime()
        conversation = _conversation_for_request(
            audio_file.name, prompt, prior_response, followup_prompt
        )
        inputs = processor.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=True, return_dict=True
        ).to(model.device)
        inputs["input_features"] = inputs["input_features"].to(model.dtype)
        outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        generated = outputs[:, inputs.input_ids.shape[1]:]
        eos_ids = getattr(model.generation_config, "eos_token_id", None)
        eos_set = set(eos_ids if isinstance(eos_ids, (list, tuple, set)) else (() if eos_ids is None else (eos_ids,)))
        stopped_at_max = bool(generated.shape[1] >= MAX_NEW_TOKENS and not any(int(token) in eos_set for token in generated.flatten()))
        text = processor.batch_decode(
            generated, skip_special_tokens=True
        )[0]
    return InferenceResult(text, round(time.perf_counter() - started, 6), cache_miss, stopped_at_max)


def _analyze_payload(
    payload: dict[str, object], infer: Callable[..., str | InferenceResult] = _infer_pinned_model
) -> dict[str, object]:
    request = validate_request(payload)
    if request.followup_prompt is None:
        outcome = infer(request.audio_bytes, request.prompt)
    else:
        outcome = infer(request.audio_bytes, request.prompt, request.prior_response, request.followup_prompt)
    if isinstance(outcome, InferenceResult):
        execution = {
            "inference_elapsed_seconds": outcome.elapsed_seconds,
            "model_cache_miss": outcome.cache_miss,
        }
        if outcome.stopped_at_max_new_tokens is not None:
            execution["stopped_at_max_new_tokens"] = outcome.stopped_at_max_new_tokens
        return response_payload(
            request,
            outcome.text,
            execution=execution,
        )
    return response_payload(request, outcome)


def _create_ml_image(modal_module: Any) -> Any:
    """Return the heavyweight image used exclusively by GPU model inference."""
    return modal_module.Image.debian_slim(python_version="3.10").pip_install(
        "torch==2.10.0",
        "transformers==5.17.0",
        "accelerate==1.15.0",
        "soundfile==0.13.1",
        # Transformers' audio loader imports librosa when decoding the bounded
        # request WAV. Keep it on the ML image only: health must stay tiny.
        "librosa==0.11.0",
        # Modal validates FastAPI endpoints against dependencies installed in
        # the remote image, rather than the local deployment environment.
        "fastapi==0.115.14",
    )


def _create_health_image(modal_module: Any) -> Any:
    """Return the deliberately small private-health image, with no ML stack."""
    return modal_module.Image.debian_slim(python_version="3.10").pip_install(
        # Modal validates FastAPI endpoint dependencies in the remote image.
        "fastapi==0.115.14",
    )


def _create_modal_app(modal_module: Any) -> tuple[Any, Any]:
    """Create the app and its separate health image only after opt-in/testing."""
    return (
        modal_module.App(APP_NAME, image=_create_ml_image(modal_module)),
        _create_health_image(modal_module),
    )


def build_modal_app_for_test(
    modal_module: Any,
    *,
    infer: Callable[..., str | InferenceResult] = _infer_pinned_model,
) -> Any:
    """Offline fake-SDK configuration checker; do not pass the real Modal SDK."""
    app, health_image = _create_modal_app(modal_module)

    @app.function(
        gpu=GPU,
        max_containers=MAX_CONTAINERS,
        min_containers=MIN_CONTAINERS,
        buffer_containers=0,
        scaledown_window=SCALEDOWN_WINDOW_SECONDS,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    @modal_module.fastapi_endpoint(method="POST", requires_proxy_auth=True, docs=False)
    def analyze(payload: dict[str, object]) -> dict[str, object]:
        return _analyze_payload(payload, infer)

    @app.function(
        image=health_image,
        max_containers=MAX_CONTAINERS,
        min_containers=MIN_CONTAINERS,
        buffer_containers=0,
        scaledown_window=SCALEDOWN_WINDOW_SECONDS,
        timeout=30,
    )
    @modal_module.fastapi_endpoint(method="GET", requires_proxy_auth=True, docs=False)
    def health() -> dict[str, object]:
        return health_payload()

    return app


def build_enabled_modal_app() -> Any:
    """Return the opt-in global app; Modal requires endpoint functions globally."""
    if os.environ.get("SONGVIZ_MODAL_ENABLE_APP") != "1":
        raise RuntimeError("refusing to create a Modal app without SONGVIZ_MODAL_ENABLE_APP=1")
    return app


def _modal_app_should_be_defined() -> bool:
    """Allow app definitions in an opted-in deploy process and Modal workers only.

    Modal workers import the mounted source afresh.  They do not inherit the
    deploy shell's opt-in variable, but the SDK's container entrypoint always
    supplies ``MODAL_CONTAINER_ARGUMENTS_PATH`` before hydrating user code.
    Keeping that worker marker here makes the globally named endpoint functions
    available to source imports while ordinary local imports remain inert.
    """
    return (
        os.environ.get("SONGVIZ_MODAL_ENABLE_APP") == "1"
        or bool(os.environ.get("MODAL_CONTAINER_ARGUMENTS_PATH"))
    )


# Modal's CLI imports this name when the explicit environment opt-in is present.
# These functions must be lexically global; Modal rejects nested Functions unless
# serialized=True, which is intentionally avoided for a transparent deployable app.
# Worker imports use the SDK's container-arguments marker instead of inheriting
# SONGVIZ_MODAL_ENABLE_APP from the local deployment shell.
if _modal_app_should_be_defined():
    import modal

    app, _health_image = _create_modal_app(modal)

    @app.function(
        gpu=GPU,
        max_containers=MAX_CONTAINERS,
        min_containers=MIN_CONTAINERS,
        buffer_containers=0,
        scaledown_window=SCALEDOWN_WINDOW_SECONDS,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True, docs=False)
    def analyze(payload: dict[str, object]) -> dict[str, object]:
        return _analyze_payload(payload)

    @app.function(
        image=_health_image,
        max_containers=MAX_CONTAINERS,
        min_containers=MIN_CONTAINERS,
        buffer_containers=0,
        scaledown_window=SCALEDOWN_WINDOW_SECONDS,
        timeout=30,
    )
    @modal.fastapi_endpoint(method="GET", requires_proxy_auth=True, docs=False)
    def health() -> dict[str, object]:
        return health_payload()
else:
    app = None
