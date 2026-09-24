from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments import run_semantic_music_flamingo as runner


STAGE_A = "Describe the clip with clip-relative times."
STAGE_B = "Ask the frozen Stage-B questions."


def frozen_prompts() -> dict[str, object]:
    return {
        "stage_a": STAGE_A,
        "stage_b": STAGE_B,
        "decoding": {"do_sample": False, "framework_seed": 0, "max_new_tokens_per_stage": 400},
        "retry": {
            "fresh_context": True,
            "maximum_per_clip": 1,
            "stage_a_append": runner.RETRY_STAGE_A_APPEND,
        },
    }


def test_retry_prompt_is_distinct_from_the_historical_unchanged_prompt() -> None:
    prompts = frozen_prompts()
    # The former unchanged retry would fail this protocol predicate: it does not
    # contain the required frozen timestamp-format instruction.
    assert runner.RETRY_STAGE_A_APPEND not in prompts["stage_a"]
    retry_prompt = runner.retry_stage_a_prompt(prompts)
    assert retry_prompt == f"{STAGE_A}\n{runner.RETRY_STAGE_A_APPEND}"
    assert retry_prompt.count(runner.RETRY_STAGE_A_APPEND) == 1


def test_turns_binds_exact_stage_a_user_prompt_and_reply_into_stage_b(monkeypatch, tmp_path: Path) -> None:
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"local-only-test-audio")
    sent: list[dict[str, object]] = []

    def fake_post_once(_modal_bin: str, _url: str, payload: dict[str, object]) -> dict[str, object]:
        sent.append(payload)
        return {"response": "Stage A exact reply [2 s]"} if len(sent) == 1 else {"response": "Stage B reply [2 s]"}

    monkeypatch.setattr(runner, "post_once", fake_post_once)
    result = runner.turns("unused", "https://example.invalid/analyze", audio, frozen_prompts())
    assert result["retry_required"] is False
    assert sent[0]["prompt"] == STAGE_A
    assert sent[1] == {
        **sent[0],
        "prior_response": "Stage A exact reply [2 s]",
        "followup_prompt": STAGE_B,
    }


def test_retry_turn_uses_suffix_once_and_reconstructs_its_own_stage_a_transcript(monkeypatch, tmp_path: Path) -> None:
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"local-only-test-audio")
    sent: list[dict[str, object]] = []

    def fake_post_once(_modal_bin: str, _url: str, payload: dict[str, object]) -> dict[str, object]:
        sent.append(payload)
        return {"response": "Retry Stage A [3 s]"} if len(sent) == 1 else {"response": "Retry Stage B [3 s]"}

    monkeypatch.setattr(runner, "post_once", fake_post_once)
    runner.turns("unused", "https://example.invalid/analyze", audio, frozen_prompts(), retry=True)
    assert sent[0]["prompt"] == f"{STAGE_A}\n{runner.RETRY_STAGE_A_APPEND}"
    assert sent[0]["prompt"].count(runner.RETRY_STAGE_A_APPEND) == 1
    assert sent[1]["prompt"] == sent[0]["prompt"]
    assert sent[1]["prior_response"] == "Retry Stage A [3 s]"
    assert sent[1]["followup_prompt"] == STAGE_B


def test_execution_contract_hashes_local_sources_and_frozen_effective_settings(tmp_path: Path) -> None:
    prompts = frozen_prompts()
    for name, value in {
        "manifest.json": {"kind": "test"},
        "order_mapping.json": {"clips": []},
        "prompts.json": prompts,
    }.items():
        (tmp_path / name).write_text(json.dumps(value), encoding="utf-8")
    contract = runner.execution_contract(tmp_path, prompts)
    assert contract["scope"].startswith("local source/config provenance")
    assert contract["effective_settings"] == {
        "do_sample": False,
        "framework_seed": 0,
        "max_new_tokens_per_stage": 400,
        "model_id": "nvidia/music-flamingo-2601-hf",
        "model_revision": "6b5be086d52f65a1e204cb0faf70bf54e2741ecd",
    }
    assert contract["retry"]["retry_stage_a_sha256"] == hashlib.sha256(
        runner.retry_stage_a_prompt(prompts).encode("utf-8")
    ).hexdigest()
    assert set(contract["source_files"]) == {"runner", "modal_app", "modal_schema"}
    assert all(len(row["sha256"]) == 64 for row in contract["source_files"].values())
