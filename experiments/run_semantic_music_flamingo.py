"""Execute only the hash-bound semantic-probe-02 package through private Modal.

The runner sends no request until its package checks and private health check
pass.  It makes Stage B a reconstructed follow-up conversation containing the
exact Stage-A response, never retries an individual POST, and permits at most
one complete two-turn retry per clip under doc 27's mechanical rule.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any
from urllib.parse import urljoin, urlsplit

from experiments.modal_music_flamingo import app as modal_app
from experiments.modal_music_flamingo.schema import MODEL_ID, MODEL_REVISION


TIME_RE = re.compile(r"\b\d{1,3}(?:\.\d+)?\s*(?:s|sec|secs|second|seconds)\b|\b\d{1,2}:\d{2}\b", re.I)
EXPECTED_CLIPS = [f"clip_{number:02d}.wav" for number in range(1, 9)]
ROOT = Path(__file__).resolve().parents[1]
RETRY_STAGE_A_APPEND = "Write each observation as `[start–end s] description` or `[time s] description`."


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_fingerprint(path: Path) -> dict[str, object]:
    """Fingerprint a local input without claiming it was previously deployed."""
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha256(resolved)}


def retry_stage_a_prompt(prompts: dict[str, Any]) -> str:
    """Build the sole permitted retry prompt, with its suffix exactly once."""
    stage_a = prompts.get("stage_a")
    retry = prompts.get("retry")
    suffix = retry.get("stage_a_append") if isinstance(retry, dict) else None
    if not isinstance(stage_a, str) or not isinstance(suffix, str) or suffix != RETRY_STAGE_A_APPEND:
        raise ValueError("prompts lack the frozen Stage-A retry suffix")
    if stage_a.count(suffix):
        raise ValueError("initial Stage-A prompt already contains the retry suffix")
    return f"{stage_a}\n{suffix}"


def execution_contract(package_dir: Path, prompts: dict[str, Any]) -> dict[str, object]:
    """Return reusable local provenance for a *future* semantic execution.

    This records source and frozen-input bytes plus effective settings.  It is a
    source/config contract only: it intentionally makes no assertion about a
    remote deployment or any prior run.
    """
    decoding = prompts.get("decoding")
    retry = prompts.get("retry")
    if not isinstance(decoding, dict) or not isinstance(retry, dict):
        raise ValueError("prompts lack frozen decoding or retry settings")
    effective = {
        "do_sample": decoding.get("do_sample"),
        "framework_seed": decoding.get("framework_seed"),
        "max_new_tokens_per_stage": decoding.get("max_new_tokens_per_stage"),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
    if effective != {
        "do_sample": False,
        "framework_seed": 0,
        "max_new_tokens_per_stage": modal_app.MAX_NEW_TOKENS,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }:
        raise ValueError("frozen decoding settings do not match the production wrapper")
    retry_prompt = retry_stage_a_prompt(prompts)
    return {
        "kind": "songviz-semantic-execution-contract",
        "schema_version": 1,
        "scope": "local source/config provenance for a future run; not evidence of deployed or prior-run state",
        "package_files": {
            name: file_fingerprint(package_dir / name)
            for name in ("manifest.json", "order_mapping.json", "prompts.json")
        },
        "source_files": {
            "runner": file_fingerprint(Path(__file__)),
            "modal_app": file_fingerprint(ROOT / "experiments" / "modal_music_flamingo" / "app.py"),
            "modal_schema": file_fingerprint(ROOT / "experiments" / "modal_music_flamingo" / "schema.py"),
        },
        "effective_settings": effective,
        "retry": {
            "fresh_context": retry.get("fresh_context"),
            "maximum_per_clip": retry.get("maximum_per_clip"),
            "stage_a_append": RETRY_STAGE_A_APPEND,
            "retry_stage_a_sha256": hashlib.sha256(retry_prompt.encode("utf-8")).hexdigest(),
        },
        "stage_b_transcript": ["exact_stage_a_user", "exact_stage_a_assistant", "frozen_stage_b_user"],
    }


def load_package(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = json.loads((path / "manifest.json").read_text())
    order = json.loads((path / "order_mapping.json").read_text())
    prompts = json.loads((path / "prompts.json").read_text())
    if manifest.get("kind") != "songviz-semantic-preregistration":
        raise ValueError("unexpected package kind")
    if not manifest.get("declared_inputs", {}).get("replacement_control_record"):
        raise ValueError("semantic-probe-02 replacement control is not bound")
    if [row.get("file_name") for row in order.get("clips", [])] != EXPECTED_CLIPS:
        raise ValueError("unexpected clip order")
    for row in manifest.get("generated_clips", []):
        clip = path / row["path"]
        if not clip.is_file() or sha256(clip) != row["sha256"]:
            raise ValueError(f"clip fingerprint mismatch: {row.get('file_name')}")
    if len(manifest.get("generated_clips", [])) != 8:
        raise ValueError("package must have exactly eight clips")
    return manifest, order, prompts


def modal_request(modal_bin: str, method: str, url: str, payload: dict[str, Any] | None) -> tuple[int, dict[str, str], dict[str, Any]]:
    token_id = os.environ.get("SONGVIZ_MODAL_PROXY_TOKEN_ID")
    token_secret = os.environ.get("SONGVIZ_MODAL_PROXY_TOKEN_SECRET")
    if not token_id or not token_secret:
        raise RuntimeError("missing ephemeral SONGVIZ_MODAL_PROXY_TOKEN_ID/SECRET")
    with tempfile.TemporaryDirectory(prefix="songviz-modal-") as temp:
        root = Path(temp)
        headers, body = root / "headers.txt", root / "body.json"
        command = ["curl", "-sS", "-D", str(headers), "-o", str(body), "-w", "%{http_code}", "-X", method,
                   "-H", f"Modal-Key: {token_id}", "-H", f"Modal-Secret: {token_secret}", url]
        if payload is not None:
            request = root / "request.json"
            request.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
            command.extend(("-H", "Content-Type: application/json", "--data-binary", f"@{request}"))
        completed = subprocess.run(command, check=False, text=True, capture_output=True, timeout=360)
        if completed.returncode:
            raise RuntimeError(f"modal curl failed ({completed.returncode}): {completed.stderr.strip()}")
        status = int(completed.stdout.strip())
        parsed_headers: dict[str, str] = {}
        for line in headers.read_text(encoding="latin-1").splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                parsed_headers[key.lower()] = value.strip()
        raw = body.read_text(encoding="utf-8") if body.exists() else ""
        try:
            parsed = json.loads(raw) if raw else {}
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"endpoint returned non-JSON HTTP {status}") from exc
    return status, parsed_headers, parsed


def post_once(modal_bin: str, analyze_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    url = analyze_url.rstrip("/")
    status, headers, response = modal_request(modal_bin, "POST", url, payload)
    if status == 303:
        location = headers.get("location")
        resolved = urljoin(url, location or "")
        original, target = urlsplit(url), urlsplit(resolved)
        if not location or target.scheme != original.scheme or target.netloc != original.netloc:
            raise RuntimeError("unsafe or absent 303 result Location; POST was not retried")
        status, _, response = modal_request(modal_bin, "GET", resolved, None)
    if status != 200:
        raise RuntimeError(f"analysis returned HTTP {status}; POST was not retried")
    if not isinstance(response.get("response"), str):
        raise RuntimeError("analysis response lacks text")
    return response


def turns(modal_bin: str, analyze_url: str, audio: Path, prompts: dict[str, Any], *, retry: bool = False) -> dict[str, Any]:
    encoded = base64.b64encode(audio.read_bytes()).decode("ascii")
    stage_a_prompt = retry_stage_a_prompt(prompts) if retry else prompts["stage_a"]
    first = {"content_type": "audio/wav", "audio_base64": encoded, "prompt": stage_a_prompt}
    stage_a = post_once(modal_bin, analyze_url, first)
    second = {**first, "prior_response": stage_a["response"], "followup_prompt": prompts["stage_b"]}
    stage_b = post_once(modal_bin, analyze_url, second)
    retry = (
        not stage_a["response"].strip()
        or not stage_b["response"].strip()
        or bool(stage_a.get("runtime", {}).get("execution", {}).get("stopped_at_max_new_tokens"))
        or bool(stage_b.get("runtime", {}).get("execution", {}).get("stopped_at_max_new_tokens"))
        or not TIME_RE.search(stage_a["response"] + "\n" + stage_b["response"])
    )
    return {"stage_a": stage_a, "stage_b": stage_b, "retry_required": retry}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--health-url", required=True)
    parser.add_argument("--analyze-url", required=True)
    parser.add_argument("--modal-bin", required=True)
    args = parser.parse_args()
    if args.result_dir.exists():
        raise FileExistsError(f"refusing to overwrite result directory: {args.result_dir}")
    manifest, order, prompts = load_package(args.package_dir)
    health_status, _, health = modal_request(args.modal_bin, "GET", args.health_url.rstrip("/"), None)
    if health_status != 200 or health != {"status": "ok", "ready": False, "schema_version": "songviz.music_flamingo.v1"}:
        raise RuntimeError("private health did not pass; no analysis POST sent")
    args.result_dir.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    run_header = {
        "package_manifest_sha256": sha256(args.package_dir / "manifest.json"),
        "health_url": args.health_url,
        "analyze_url": args.analyze_url,
        "health": health,
        "execution_contract": execution_contract(args.package_dir, prompts),
    }
    (args.result_dir / "run.json").write_text(json.dumps({**run_header, "records": records}, indent=2) + "\n")
    for row in order["clips"]:
        audio = args.package_dir / "clips" / row["file_name"]
        initial = turns(args.modal_bin, args.analyze_url, audio, prompts)
        selected = initial
        retry = None
        if initial["retry_required"]:
            retry = turns(args.modal_bin, args.analyze_url, audio, prompts, retry=True)
            selected = retry
        record = {"clip_file": row["file_name"], "clip_id": row["clip_id"], "audio_sha256": sha256(audio), "initial": initial, "retry": retry, "selected": "retry" if retry else "initial"}
        records.append(record)
        (args.result_dir / "run.json").write_text(json.dumps({**run_header, "records": records}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
