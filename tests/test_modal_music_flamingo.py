import base64
import importlib
import io
import sys
import wave

import pytest

from experiments.modal_music_flamingo import app as modal_app
from experiments.modal_music_flamingo.client import (
    ColdStartTimeout,
    ResultRedirectError,
    follow_result_redirect,
    invoke_once,
)
from experiments.modal_music_flamingo.schema import (
    MAX_AUDIO_SECONDS,
    MAX_PRIOR_RESPONSE_CHARS,
    MODEL_REVISION,
    RequestValidationError,
    health_payload,
    response_payload,
    validate_request,
)


def test_production_generation_budget_is_the_frozen_400_tokens_per_stage():
    assert modal_app.MAX_NEW_TOKENS == 400


def wav_bytes(*, seconds=1, sample_rate=8000, channels=1):
    stream = io.BytesIO()
    with wave.open(stream, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * int(seconds * sample_rate * channels))
    return stream.getvalue()


def payload(**changes):
    request = {
        "content_type": "audio/wav",
        "audio_base64": base64.b64encode(wav_bytes()).decode(),
        "prompt": "Describe only what is audible.",
    }
    request.update(changes)
    return request


def test_schema_accepts_one_bounded_wav_and_returns_pinned_metadata():
    request = validate_request(payload())
    assert request.duration_seconds == 1
    result = response_payload(request, "A bounded answer.")
    assert result["runtime"]["model_revision"] == MODEL_REVISION
    assert result["runtime"]["input"]["channels"] == 1
    result = response_payload(
        request, "A bounded answer.", execution={"inference_elapsed_seconds": 1.25, "model_cache_miss": True}
    )
    assert result["runtime"]["execution"] == {
        "inference_elapsed_seconds": 1.25, "model_cache_miss": True
    }


def test_schema_accepts_followup_only_as_a_complete_prior_turn_pair():
    request = validate_request(payload(prior_response="Stage A answer.", followup_prompt="Stage B question."))
    assert request.prior_response == "Stage A answer."
    assert request.followup_prompt == "Stage B question."
    assert response_payload(request, "Stage B answer.")["runtime"]["input"]["followup_turn"] is True


@pytest.mark.parametrize("changes", [
    {"prior_response": "stage a"},
    {"followup_prompt": "stage b"},
    {"prior_response": "", "followup_prompt": "stage b"},
    {"prior_response": "stage a", "followup_prompt": " "},
    {"prior_response": "x" * (MAX_PRIOR_RESPONSE_CHARS + 1), "followup_prompt": "stage b"},
])
def test_schema_rejects_partial_or_oversize_followup_turn(changes):
    with pytest.raises(RequestValidationError):
        validate_request(payload(**changes))


@pytest.mark.parametrize("changes", [
    {"content_type": "audio/mpeg"},
    {"audio_base64": "not base64!"},
    {"audio_base64": base64.b64encode(b"not-a-wav").decode()},
    {"prompt": "   "},
    {"unexpected": "field"},
    {"audio_base64": base64.b64encode(wav_bytes(seconds=MAX_AUDIO_SECONDS + 1)).decode()},
])
def test_schema_rejects_bad_input_before_inference(changes):
    with pytest.raises(RequestValidationError):
        validate_request(payload(**changes))


def test_import_is_inert_without_modal_or_opt_in(monkeypatch):
    monkeypatch.delenv("SONGVIZ_MODAL_ENABLE_APP", raising=False)
    sys.modules.pop("modal", None)
    module = importlib.reload(modal_app)
    assert module.app is None
    assert "modal" not in sys.modules
    with pytest.raises(RuntimeError, match="SONGVIZ_MODAL_ENABLE_APP"):
        module.build_enabled_modal_app()


def test_runtime_cache_reuses_injected_loader_without_ml_imports():
    calls = []
    expected = (object(), object())

    def loader():
        calls.append("load")
        return expected

    modal_app._clear_runtime_cache_for_test()
    modal_app._set_runtime_loader_for_test(loader)
    try:
        assert modal_app._get_pinned_runtime() == (expected, True)
        assert modal_app._get_pinned_runtime() == (expected, False)
        assert calls == ["load"]
        assert "transformers" not in sys.modules
    finally:
        modal_app._clear_runtime_cache_for_test()


class FakeImage:
    def __init__(self):
        self.packages = ()

    def pip_install(self, *packages):
        self.packages = packages
        return self


class FakeApp:
    def __init__(self, name, *, image):
        self.name, self.image, self.functions = name, image, []

    def function(self, **options):
        def decorate(function):
            self.functions.append((function, options))
            return function
        return decorate


class FakeModal:
    class Image:
        @staticmethod
        def debian_slim(**kwargs):
            return FakeImage()

    class Secret:
        @staticmethod
        def from_name(name, **kwargs):
            return name, kwargs

    @staticmethod
    def App(name, *, image):
        return FakeApp(name, image=image)

    @staticmethod
    def fastapi_endpoint(**options):
        def decorate(function):
            function.endpoint_options = options
            return function
        return decorate


def test_modal_definition_is_private_pinned_and_scales_to_zero():
    app = modal_app.build_modal_app_for_test(FakeModal, infer=lambda _audio, _prompt: "ok")
    assert app.name == modal_app.APP_NAME
    assert app.image.packages == (
        "torch==2.10.0",
        "transformers==5.17.0",
        "accelerate==1.15.0",
        "soundfile==0.13.1",
        "librosa==0.11.0",
        "fastapi==0.115.14",
    )
    assert len(app.functions) == 2
    for function, options in app.functions:
        assert options["max_containers"] == 1
        assert options["min_containers"] == 0
        assert options["buffer_containers"] == 0
        assert function.endpoint_options == {
            "method": "POST" if function.__name__ == "analyze" else "GET",
            "requires_proxy_auth": True,
            "docs": False,
        }
    analyze = next(function for function, _ in app.functions if function.__name__ == "analyze")
    analyze_options = next(options for function, options in app.functions if function.__name__ == "analyze")
    health_options = next(options for function, options in app.functions if function.__name__ == "health")
    assert "secrets" not in analyze_options
    assert "image" not in analyze_options
    assert health_options["image"].packages == ("fastapi==0.115.14",)
    assert not set(health_options["image"].packages) & {
        "torch==2.10.0",
        "transformers==5.17.0",
        "accelerate==1.15.0",
        "soundfile==0.13.1",
        "librosa==0.11.0",
    }
    assert analyze(payload())["response"] == "ok"
    assert health_payload() == {
        "status": "ok", "ready": False, "schema_version": "songviz.music_flamingo.v1"
    }


def test_modal_worker_source_import_defines_endpoints_without_deploy_opt_in(monkeypatch):
    """Modal workers import source afresh and only receive container arguments."""
    monkeypatch.delenv("SONGVIZ_MODAL_ENABLE_APP", raising=False)
    monkeypatch.setenv("MODAL_CONTAINER_ARGUMENTS_PATH", "/modal/container-arguments.pb")
    monkeypatch.setitem(sys.modules, "modal", FakeModal)
    module = importlib.reload(modal_app)
    try:
        assert module._modal_app_should_be_defined() is True
        assert module.app.name == module.APP_NAME
        assert module.analyze.__name__ == "analyze"
        assert module.health.__name__ == "health"
        health_options = next(
            options for function, options in module.app.functions if function.__name__ == "health"
        )
        assert health_options["image"].packages == ("fastapi==0.115.14",)
        assert "gpu" not in health_options
    finally:
        monkeypatch.delenv("MODAL_CONTAINER_ARGUMENTS_PATH", raising=False)
        monkeypatch.delitem(sys.modules, "modal", raising=False)
        importlib.reload(module)


def test_modal_analysis_exposes_non_sensitive_runtime_metadata():
    outcome = modal_app.InferenceResult("ok", 2.5, True)
    app = modal_app.build_modal_app_for_test(FakeModal, infer=lambda _audio, _prompt: outcome)
    analyze = next(function for function, _ in app.functions if function.__name__ == "analyze")
    assert analyze(payload())["runtime"]["execution"] == {
        "inference_elapsed_seconds": 2.5,
        "model_cache_miss": True,
    }


def test_modal_analysis_passes_prior_turn_only_for_followup_requests():
    calls = []

    def infer(*args):
        calls.append(args)
        return "ok"

    app = modal_app.build_modal_app_for_test(FakeModal, infer=infer)
    analyze = next(function for function, _ in app.functions if function.__name__ == "analyze")
    analyze(payload())
    analyze(payload(prior_response="first answer", followup_prompt="second question"))
    assert [len(args) for args in calls] == [2, 4]
    assert calls[1][2:] == ("first answer", "second question")


def test_production_stage_b_conversation_is_the_exact_reconstructed_transcript():
    stage_a = "Frozen Stage A prompt"
    stage_a_reply = "Exact Stage A model response"
    stage_b = "Frozen Stage B prompt"
    assert modal_app._conversation_for_request("/tmp/clip.wav", stage_a, stage_a_reply, stage_b) == [
        {"role": "user", "content": [
            {"type": "text", "text": stage_a},
            {"type": "audio", "path": "/tmp/clip.wav"},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": stage_a_reply}]},
        {"role": "user", "content": [{"type": "text", "text": stage_b}]},
    ]


def test_client_retries_only_health_then_posts_once():
    calls = []
    health_responses = iter([(503, {}), (200, health_payload())])

    def request(method, url, headers, body):
        calls.append((method, url, headers, body))
        return next(health_responses) if method == "GET" else (200, {"response": "ok"})

    result = invoke_once(
        base_url="https://private.example/",
        payload=payload(),
        headers={"Modal-Key": "caller-owned", "Modal-Secret": "caller-owned"},
        request=request,
        deadline_seconds=5,
        monotonic=iter([0, 0, 0, 0]).__next__,
        sleep=lambda _seconds: None,
    )
    assert result == {"response": "ok"}
    assert [call[0] for call in calls] == ["GET", "GET", "POST"]
    assert calls[-1][2]["Content-Type"] == "application/json"


def test_client_does_not_post_after_cold_start_deadline():
    calls = []

    def request(method, url, headers, body):
        calls.append(method)
        return 503, {}

    with pytest.raises(ColdStartTimeout):
        invoke_once(
            base_url="https://private.example",
            payload=payload(), headers={}, request=request, deadline_seconds=1,
            monotonic=iter([0, 1]).__next__, sleep=lambda _seconds: None,
        )
    assert calls == ["GET"]


def test_client_follows_modal_result_redirect_with_get_only_once():
    calls = []

    def request(method, url, headers, body):
        calls.append((method, url, dict(headers), body))
        return 200, {"response": "OK"}

    result = follow_result_redirect(
        original_url="https://private.example/analyze",
        location="/analyze?modal-result=request-token",
        headers={"Modal-Key": "caller-owned", "Modal-Secret": "caller-owned"},
        request=request,
    )

    assert result == {"response": "OK"}
    assert calls == [
        (
            "GET",
            "https://private.example/analyze?modal-result=request-token",
            {"Modal-Key": "caller-owned", "Modal-Secret": "caller-owned"},
            None,
        )
    ]


@pytest.mark.parametrize("location", [None, "https://elsewhere.example/result", "//elsewhere.example/result"])
def test_client_rejects_unsafe_or_missing_result_redirect_without_request(location):
    calls = []

    with pytest.raises(ResultRedirectError):
        follow_result_redirect(
            original_url="https://private.example/analyze",
            location=location,
            headers={"Modal-Key": "caller-owned", "Modal-Secret": "caller-owned"},
            request=lambda *args: calls.append(args),
        )

    assert calls == []


def test_client_does_not_post_again_when_result_get_fails():
    calls = []

    def request(method, url, headers, body):
        calls.append((method, url, body))
        return 500, {"detail": "not ready"}

    with pytest.raises(ResultRedirectError, match="HTTP 500; POST was not retried"):
        follow_result_redirect(
            original_url="https://private.example/analyze",
            location="?modal-result=request-token",
            headers={},
            request=request,
        )

    assert calls == [("GET", "https://private.example/analyze?modal-result=request-token", None)]
