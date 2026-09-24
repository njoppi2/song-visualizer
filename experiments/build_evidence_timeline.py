"""Build a provenance-bound, full-song evidence timeline from frozen reviews."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from urllib.parse import quote, unquote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from songviz.ingest import sha256_file

CODE = (
    "experiments/build_evidence_timeline.py",
    "experiments/templates/evidence_timeline.html",
    "experiments/check_evidence_timeline.cjs",
)


def _json(value: object, *, embedded: bool = False) -> str:
    text = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":") if embedded else None)
    return text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026") if embedded else text + "\n"


def record(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _resolve(row: dict, package: Path) -> Path:
    raw = Path(row["path"])
    return raw if raw.is_absolute() else (ROOT / raw if (ROOT / raw).exists() else package / raw)


def verify_records(rows: list[dict], package: Path) -> None:
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str) or not isinstance(row.get("sha256"), str):
            raise ValueError("Malformed frozen fingerprint record")
        path = _resolve(row, package)
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise ValueError(f"Changed or missing frozen input: {path}")


def verify_package(package: Path, kind: str | None) -> tuple[dict, list[Path]]:
    manifest_path = package / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if kind is not None and manifest.get("kind") != kind:
        raise ValueError(f"Unexpected frozen package kind: {manifest.get('kind')}")
    rows: list[dict] = []
    for name in ("sources", "outputs", "input_snapshots"):
        entries = manifest.get(name, [])
        if entries is None:
            entries = []
        if not isinstance(entries, list):
            raise ValueError(f"Malformed {name} in {manifest_path}")
        rows.extend(entries)
    verify_records(rows, package)
    return manifest, [manifest_path, *[_resolve(row, package) for row in rows]]


def _bound_file(manifest: dict, package: Path, name: str) -> Path:
    matches = [row for field in ("outputs", "sources", "input_snapshots") for row in (manifest.get(field) or [])
               if _resolve(row, package).resolve() == (package / name).resolve()]
    if len(matches) != 1:
        raise ValueError(f"{name} is not uniquely bound by {package}/manifest.json")
    return package / name


def _audio_from_review(review: dict, audio_package: Path, audio_manifest: dict) -> Path:
    encoded = review.get("audio_path")
    digest = review.get("audio_sha256")
    if not isinstance(encoded, str) or not isinstance(digest, str):
        raise ValueError("Listening review has no bound audio")
    audio = (audio_package / unquote(encoded)).resolve()
    candidates = [row for row in (audio_manifest.get("outputs") or []) + (audio_manifest.get("sources") or [])
                  if _resolve(row, audio_package).resolve() == audio]
    if len(candidates) != 1 or candidates[0].get("sha256") != digest or not audio.is_file() or sha256_file(audio) != digest:
        raise ValueError("Audio path/fingerprint is not bound to the frozen structure review")
    return audio


def _local_context(role: dict) -> dict:
    times, curves, stems = role.get("times_s"), role.get("curves"), role.get("stem_names")
    if not isinstance(times, list) or not isinstance(curves, list) or not isinstance(stems, list):
        raise ValueError("Malformed role-context data")
    scales = []
    for curve in curves:
        if not isinstance(curve, dict) or not isinstance(curve.get("scale_beats"), int) or not isinstance(curve.get("samples"), list):
            raise ValueError("Malformed local-context curve")
        if len(curve["samples"]) != len(times):
            raise ValueError("Local-context samples do not align to timeline")
        # Preserve null values: unsupported values are unavailable, never zero.
        scales.append({"scale_beats": curve["scale_beats"], "samples": curve["samples"]})
    return {"times_s": times, "scales": scales, "audibility_floors": role.get("audibility_floors"),
            "descriptors": ["mean_rms", "active_fraction", "rms_power_share", "spectral_concentration", "adjacent_spectral_change"]}


def _recurrence(data: dict) -> tuple[dict, list[dict]]:
    results = data.get("results")
    if not isinstance(results, list):
        raise ValueError("Malformed recurrence result")
    scales, paired = [], []
    for result in results:
        if not isinstance(result, dict):
            raise ValueError("Malformed recurrence scale")
        spans, pairs = result.get("spans"), result.get("pairs")
        if not isinstance(spans, list) or not isinstance(pairs, list):
            raise ValueError("Recurrence has no spans/pairs")
        # The frozen package has thousands of pairs.  Keep a deterministic,
        # inspectable listening set rather than making a page ship the whole
        # analytical cache.  Context remains complete and points at the source.
        selected = sorted(pairs, key=lambda item: (-float(item.get("similarity", -2)), int(item.get("a", -1)), int(item.get("b", -1))))[:12]
        scale = {key: result.get(key) for key in ("scale_beats", "stride_beats", "spans", "context")}
        scale["pairs"] = selected
        scales.append(scale)
        for pair in selected:
            if not isinstance(pair, dict) or not isinstance(pair.get("a"), int) or not isinstance(pair.get("b"), int):
                raise ValueError("Malformed recurrence pair")
            a, b = pair["a"], pair["b"]
            if not (0 <= a < len(spans) and 0 <= b < len(spans)):
                raise ValueError("Recurrence pair references a missing span")
            paired.append({"scale_beats": result.get("scale_beats"), **pair,
                           "a": {"span_index": a, **spans[a]}, "b": {"span_index": b, **spans[b]},
                           "prior": {"start_s": spans[a]["start_s"], "end_s": spans[a]["end_s"]},
                           "target": {"start_s": spans[b]["start_s"], "end_s": spans[b]["end_s"]}})
    return {"scales": scales}, paired


def build(*, role_context: Path, structure_evaluation: Path, listening: Path, audio_review: Path,
          feedback: Path, out: Path) -> None:
    role_context, structure_evaluation, listening, audio_review, feedback, out = (
        p.resolve() for p in (role_context, structure_evaluation, listening, audio_review, feedback, out))
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    if not feedback.is_file():
        raise ValueError("Missing raw listening feedback")
    template = ROOT / CODE[1]
    if not template.is_file():
        raise FileNotFoundError(f"Evidence-timeline template is required before build: {template}")
    role_manifest, role_files = verify_package(role_context, "songviz-role-context-review")
    eval_manifest, eval_files = verify_package(structure_evaluation, "songviz-structural-development-evaluation")
    listen_manifest, listen_files = verify_package(listening, "songviz-listening-examples")
    audio_manifest, audio_files = verify_package(audio_review, None)
    role_path = _bound_file(role_manifest, role_context, "role-context.json")
    listen_path = _bound_file(listen_manifest, listening, "review.json")
    recurrence_path = _bound_file(eval_manifest, structure_evaluation, "recurrence.json")
    role, listen_data, recurrence_data = (json.loads(p.read_text()) for p in (role_path, listen_path, recurrence_path))
    feedback_data = json.loads(feedback.read_text())
    feedback_digest = sha256_file(feedback)
    feedback_bound = [row for manifest, package in ((role_manifest, role_context), (listen_manifest, listening))
                      for field in ("sources", "outputs", "input_snapshots") for row in (manifest.get(field) or [])
                      if isinstance(row, dict) and row.get("sha256") == feedback_digest
                      and _resolve(row, package).resolve() == feedback]
    if len(feedback_bound) != 1:
        raise ValueError("Raw listening feedback is not uniquely bound by the frozen reviews")
    answers = feedback_data.get("answers")
    if not isinstance(answers, list):
        raise ValueError("Raw listening feedback has no answers")
    answer_ids = {item.get("example_id") for item in answers if isinstance(item, dict)}
    excerpts = listen_data.get("examples")
    if not isinstance(excerpts, list) or not all(item.get("id") in answer_ids for item in excerpts if isinstance(item, dict)):
        raise ValueError("Listening examples are not bound to raw feedback answers")
    audio = _audio_from_review(listen_data, audio_review, audio_manifest)
    recurrence, paired = _recurrence(recurrence_data)
    title = listen_data.get("song_title")
    duration = listen_data.get("duration_s")
    if not isinstance(title, str) or not isinstance(duration, (int, float)):
        raise ValueError("Listening review metadata is incomplete")
    unavailable = sorted({float(times[index]) for curve in role.get("curves", []) for index, sample in enumerate(curve.get("samples", [])) if sample is None
                          for times in [role.get("times_s", [])] if index < len(times)})
    data = {"schema_version": 1, "kind": "songviz-evidence-timeline", "title": title, "duration_s": duration,
            "audio_path": quote(Path(os.path.relpath(audio, out)).as_posix(), safe="/"), "audio_sha256": sha256_file(audio),
            "excerpts": excerpts, "stem_names": role.get("stem_names"), "local_context": _local_context(role),
            "unavailable_probes": [{"time_s": value} for value in unavailable],
            "recurrence": recurrence, "paired_listening": paired,
            "feedback": {"source": {"path": str(feedback), "sha256": feedback_digest}, "answers": answers},
            "provenance": {"frozen_packages": {"role_context": record(role_context / "manifest.json"),
                           "structure_evaluation": record(structure_evaluation / "manifest.json"), "listening": record(listening / "manifest.json"),
                           "audio_review": record(audio_review / "manifest.json")},
                           "limitations": ["Stem descriptors are acoustic measurements, not musical-role or importance labels.",
                                           "Null local-context values mean unavailable support; they are not zero.",
                                           "Recurrence/history pairs are offline, windowed comparison evidence."]}}
    consumed = [*role_files, *eval_files, *listen_files, *audio_files, feedback, *[ROOT / path for path in CODE]]
    # Eliminate duplicates while preserving deterministic absolute path order.
    inputs = sorted({p.resolve() for p in consumed}, key=str)
    for path in inputs:
        if not path.is_file():
            raise ValueError(f"Required source/snapshot is missing: {path}")
    out.mkdir(parents=True)
    for relative in CODE:
        destination = out / "inputs" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    for source, name in ((role_context / "manifest.json", "role-context-manifest.json"), (structure_evaluation / "manifest.json", "structure-evaluation-manifest.json"),
                         (listening / "manifest.json", "listening-manifest.json"), (audio_review / "manifest.json", "audio-manifest.json"), (feedback, "listening-feedback.json")):
        shutil.copy2(source, out / "inputs" / name)
    timeline = out / "evidence-timeline.json"
    timeline.write_text(_json(data))
    page = template.read_text().replace("{{TIMELINE_JSON}}", _json(data, embedded=True)).replace("{{TIMELINE_SHA}}", sha256_file(timeline))
    if "{{TIMELINE_JSON}}" in page or "{{TIMELINE_SHA}}" in page:
        raise ValueError("Evidence-timeline template tokens were not resolved")
    (out / "index.html").write_text(page)
    manifest = {"schema_version": 1, "kind": "songviz-evidence-timeline", "sources": [record(p) for p in inputs],
                "input_snapshots": [record(p) for p in sorted((out / "inputs").rglob("*")) if p.is_file()],
                "outputs": [record(timeline), record(out / "index.html")],
                "page_integrity": "index.html is derived from snapshotted template and evidence-timeline.json SHA-256; manifest does not self-reference."}
    (out / "manifest.json").write_text(_json(manifest))
    print(f"Ready: {out / 'index.html'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role-context", type=Path, default=ROOT / "outputs/reviews/role-context-02")
    parser.add_argument("--structure-evaluation", type=Path, default=ROOT / "outputs/reviews/structure-evaluation-03")
    parser.add_argument("--listening", type=Path, default=ROOT / "outputs/reviews/listening-examples-01")
    parser.add_argument("--audio-review", type=Path, default=ROOT / "outputs/reviews/structure-review-03")
    parser.add_argument("--feedback", type=Path, default=ROOT / "benchmark/feedback/listening-examples-01.json")
    parser.add_argument("--out", type=Path, required=True)
    build(**vars(parser.parse_args()))
