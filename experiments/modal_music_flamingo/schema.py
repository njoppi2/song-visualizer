"""Dependency-free request and response contract for the Modal wrapper.

Only PCM WAV is accepted in the first feasibility route.  That deliberately
keeps type and duration validation local and deterministic, before a GPU worker
or a model import can be reached.
"""

from __future__ import annotations

import base64
import binascii
import io
import wave
from dataclasses import dataclass
from typing import Any, Mapping

MODEL_ID = "nvidia/music-flamingo-2601-hf"
MODEL_REVISION = "6b5be086d52f65a1e204cb0faf70bf54e2741ecd"
SCHEMA_VERSION = "songviz.music_flamingo.v1"
MAX_AUDIO_BYTES = 20 * 1024 * 1024
MAX_AUDIO_SECONDS = 60.0
MAX_PROMPT_CHARS = 1_000
MAX_PRIOR_RESPONSE_CHARS = 8_000
ALLOWED_CONTENT_TYPE = "audio/wav"


class RequestValidationError(ValueError):
    """Raised for input that must not schedule model work."""


@dataclass(frozen=True)
class AnalysisRequest:
    audio_bytes: bytes
    content_type: str
    prompt: str
    duration_seconds: float
    sample_rate_hz: int
    channels: int
    prior_response: str | None = None
    followup_prompt: str | None = None


def _reject(message: str) -> None:
    raise RequestValidationError(message)


def _decode_audio(value: object) -> bytes:
    if not isinstance(value, str) or not value:
        _reject("audio_base64 must be a non-empty base64 string")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        _reject("audio_base64 is not valid base64")
    if not decoded:
        _reject("audio payload is empty")
    if len(decoded) > MAX_AUDIO_BYTES:
        _reject(f"audio payload exceeds {MAX_AUDIO_BYTES} bytes")
    return decoded


def validate_request(payload: Mapping[str, Any]) -> AnalysisRequest:
    """Validate a JSON request without importing Modal or ML dependencies."""
    if not isinstance(payload, Mapping):
        _reject("request body must be a JSON object")
    unexpected = set(payload) - {"audio_base64", "content_type", "prompt", "prior_response", "followup_prompt"}
    if unexpected:
        _reject("unknown request field")
    if payload.get("content_type") != ALLOWED_CONTENT_TYPE:
        _reject("only content_type audio/wav is accepted in this feasibility route")
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        _reject("prompt must be a non-empty string")
    if len(prompt) > MAX_PROMPT_CHARS:
        _reject(f"prompt exceeds {MAX_PROMPT_CHARS} characters")
    prior_response = payload.get("prior_response")
    followup_prompt = payload.get("followup_prompt")
    if (prior_response is None) != (followup_prompt is None):
        _reject("prior_response and followup_prompt must be supplied together")
    if prior_response is not None:
        if not isinstance(prior_response, str) or not prior_response.strip():
            _reject("prior_response must be a non-empty string when supplied")
        if len(prior_response) > MAX_PRIOR_RESPONSE_CHARS:
            _reject(f"prior_response exceeds {MAX_PRIOR_RESPONSE_CHARS} characters")
        if not isinstance(followup_prompt, str) or not followup_prompt.strip():
            _reject("followup_prompt must be a non-empty string when supplied")
        if len(followup_prompt) > MAX_PROMPT_CHARS:
            _reject(f"followup_prompt exceeds {MAX_PROMPT_CHARS} characters")
    audio_bytes = _decode_audio(payload.get("audio_base64"))
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav:
            channels = wav.getnchannels()
            sample_rate_hz = wav.getframerate()
            frames = wav.getnframes()
            sample_width = wav.getsampwidth()
            compression = wav.getcomptype()
    except (EOFError, wave.Error):
        _reject("audio/wav payload is not a readable WAV file")
    if compression != "NONE" or sample_width not in (1, 2, 3, 4):
        _reject("only uncompressed PCM WAV is accepted")
    if channels not in (1, 2):
        _reject("WAV must contain one or two channels")
    if not 8_000 <= sample_rate_hz <= 48_000:
        _reject("WAV sample rate must be between 8000 and 48000 Hz")
    duration_seconds = frames / sample_rate_hz
    if duration_seconds <= 0 or duration_seconds > MAX_AUDIO_SECONDS:
        _reject(f"WAV duration must be greater than 0 and at most {MAX_AUDIO_SECONDS:g} seconds")
    return AnalysisRequest(audio_bytes, ALLOWED_CONTENT_TYPE, prompt, duration_seconds, sample_rate_hz, channels, prior_response, followup_prompt)


def health_payload() -> dict[str, object]:
    """Return intentionally non-sensitive health metadata without model readiness."""
    return {"status": "ok", "ready": False, "schema_version": SCHEMA_VERSION}


def response_payload(
    request: AnalysisRequest, model_response: str, *, execution: Mapping[str, object] | None = None
) -> dict[str, object]:
    if not isinstance(model_response, str):
        raise TypeError("model_response must be text")
    runtime: dict[str, object] = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "input": {
            "content_type": request.content_type,
            "duration_seconds": round(request.duration_seconds, 6),
            "sample_rate_hz": request.sample_rate_hz,
            "channels": request.channels,
            "followup_turn": request.followup_prompt is not None,
        },
    }
    if execution is not None:
        runtime["execution"] = dict(execution)
    return {
        "schema_version": SCHEMA_VERSION,
        "response": model_response,
        "runtime": runtime,
    }
