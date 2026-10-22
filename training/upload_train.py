"""Persistent full-file fine-tuning worker, launched by the voice library."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

from training.self_train import DATA, PYTHON, ROOT, TRAIN_PYTHON, command, register
from training.pipeline import sha256_file
from training.uploaded_audio import build, load, now, save
from server.target_voice import process as extract_target_voice


def combined_key(paths: list[Path], reference: Path | None = None) -> str:
    checksums = [sha256_file(path) for path in paths]
    if len(paths) == 1 and reference is None:
        return checksums[0]
    if reference is not None:
        checksums.append(sha256_file(reference))
    return hashlib.sha256("\n".join(checksums).encode()).hexdigest()


def join_audio(paths: list[Path], output: Path) -> Path:
    """Join files at 24 kHz, preserving clean source bandwidth where available."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="join-training-", dir=output.parent) as temporary:
        with sf.SoundFile(output, mode="w", samplerate=24000, channels=1, subtype="PCM_16") as out:
            for index, path in enumerate(paths):
                normalized = Path(temporary) / f"{index:04d}.wav"
                subprocess.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(path), "-vn", "-ac", "1",
                                "-ar", "24000", "-c:a", "pcm_s16le", str(normalized)], check=True, timeout=600)
                with sf.SoundFile(normalized) as source:
                    while True:
                        block = source.read(24000 * 10, dtype="float32")
                        if not len(block):
                            break
                        out.write(block)
                if index < len(paths) - 1:
                    out.write(np.zeros(12000, dtype=np.float32))
    return output


def prior_evaluation(manifest: dict) -> str | None:
    """Find an already evaluated split run from this exact uploaded file."""
    candidates = []
    for status_path in (DATA / "training_runs").glob("*/status.json"):
        try:
            state = json.loads(status_path.read_text(encoding="utf-8"))
            prior = json.loads((Path(state["dataset"]) / "manifest.json").read_text(encoding="utf-8"))
            if (state["status"] == "evaluated" and prior.get("archive_sha256") == manifest["archive_sha256"]
                    and prior.get("clean_before_training", False) == manifest.get("clean_before_training", False)):
                candidates.append((prior.get("voice_id") == manifest.get("voice_id"),
                                   "filtered" not in state["run"] and "vocals" not in state["run"], state["run"]))
        except (OSError, KeyError, ValueError):
            continue
    return max(candidates)[2] if candidates else None


def evaluation_dataset(full: Path, target: Path) -> dict:
    """Prepare a temporary disjoint split for pre-final validation."""
    rows = [json.loads(line) for line in (full / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(rows) < 22:
        raise ValueError("Cần ít nhất 22 đoạn cho đánh giá tách riêng; vẫn có thể train toàn bộ file.")
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    (target / "raw_audio").mkdir()
    for row in rows:
        os.link(full / "raw_audio" / row["file_name"], target / "raw_audio" / row["file_name"])
    ordered = sorted(rows, key=lambda row: hashlib.sha256(row["file_name"].encode()).hexdigest())
    test_count = max(5, min(20, round(len(rows) * .1)))
    validation_count = max(2, min(10, round(len(rows) * .05)))
    for index, row in enumerate(ordered):
        row["split"] = "test" if index < test_count else "validation" if index < test_count + validation_count else "train"
    for split in ("train", "validation", "test"):
        with (target / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in ordered:
                if row["split"] == split:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (target / "train_metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="|")
        writer.writerow(["file_name", "text"])
        writer.writerows((row["file_name"], row["text"]) for row in ordered if row["split"] == "train")
    manifest = json.loads((full / "manifest.json").read_text(encoding="utf-8"))
    manifest["full_training"] = False
    manifest["counts"] = {"train": len(rows) - test_count - validation_count,
                          "validation": validation_count, "test": test_count}
    manifest["preparation"] = "Disjoint preliminary evaluation; final model is retrained using all clips"
    (target / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def run(voice_id: str) -> int:
    if len(voice_id) != 32 or any(c not in "0123456789abcdef" for c in voice_id):
        raise ValueError("Invalid voice ID")
    voices = json.loads((DATA / "voices.json").read_text(encoding="utf-8"))
    voice = next((item for item in voices if item["id"] == voice_id), None)
    if not voice or not voice.get("source_file"):
        raise ValueError("Full original audio missing")
    source = DATA / "voices/sources" / voice["source_file"]
    if not source.is_file():
        raise FileNotFoundError(source)
    source_paths = [source]
    for row in voice.get("training_files", []):
        filename = row.get("file", "")
        if not filename or Path(filename).name != filename:
            raise ValueError("Invalid additional training file")
        extra = DATA / "voices/training" / voice_id / filename
        if not extra.is_file():
            raise FileNotFoundError(extra)
        source_paths.append(extra)
    work = DATA / "upload_training" / voice_id
    clean = bool((load(voice_id) or {}).get("clean_before_training", False))
    reference = DATA / "voices" / f"{voice_id}.wav"
    source_key = combined_key(source_paths, reference if clean else None)
    dataset = DATA / "prepared" / f"upload_{voice_id}_{'clean_v3' if clean else 'full_v2'}"
    lock_file = DATA / "auto_train/worker.lock"
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    attempt = (load(voice_id) or {}).get("attempt", 1)
    try:
        save(voice_id, status="preparing", stage="decoding", source_name=voice["source"],
             source_seconds=round(voice["duration"] + sum(row["duration"] for row in voice.get("training_files", [])), 1),
             source_file_count=len(source_paths), name=voice["name"], locale=voice["locale"],
             generated_wer=None, reference_wer=None, speaker_similarity=None,
             model_selection=None, previous_run=None)
        if ((dataset / "manifest.json").is_file()
                and json.loads((dataset / "manifest.json").read_text()).get("archive_sha256") == source_key):
            manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
        else:
            if dataset.exists():
                shutil.rmtree(dataset)
            with lock_file.open("a+") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                prepared_sources = []
                selected_chunks = total_chunks = 0
                if clean:
                    save(voice_id, stage="filtering_speaker")
                    for index, path in enumerate(source_paths):
                        selected = extract_target_voice(path, reference, work / "filtered" / str(index),
                                                        enhance=True, allow_single_speaker_fallback=False)
                        selected_chunks += selected["selected_chunks"]
                        total_chunks += selected["total_chunks"]
                        if selected["selected_chunks"]:
                            prepared_sources.append(Path(selected["target"]))
                    if not prepared_sources:
                        raise ValueError("Không tìm thấy giọng mục tiêu trong các file; cần mẫu tham chiếu sạch hơn.")
                    save(voice_id, selected_chunks=selected_chunks, total_chunks=total_chunks)
                else:
                    prepared_sources = source_paths
                training_source = (prepared_sources[0] if len(prepared_sources) == 1
                                   else join_audio(prepared_sources, work / "combined_training.wav"))
                manifest = build(voice, training_source, dataset, lambda **fields: save(voice_id, **fields))
                if clean:
                    for index in range(len(source_paths)):
                        filtered_dir = work / "filtered" / str(index)
                        shutil.rmtree(filtered_dir / "stems", ignore_errors=True)
                        (filtered_dir / "mixture.wav").unlink(missing_ok=True)
                        (filtered_dir / "reference.wav").unlink(missing_ok=True)
                manifest.update(archive=str(source.resolve()), archive_sha256=source_key,
                                source_files=[{"path": str(path.resolve()), "sha256": sha256_file(path)} for path in source_paths],
                                clean_before_training=clean)
                if clean:
                    manifest.update(filtered_source_sha256=sha256_file(training_source),
                                    speaker_filter={"method": selected["method"],
                                                    "selected_chunks": selected_chunks, "total_chunks": total_chunks},
                                    preparation="Target-speaker extraction before ASR and LoRA; rejected uncertain chunks; no held-out final test")
                (dataset / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            request = urllib.request.Request(f"http://127.0.0.1:8000/api/voices/{voice_id}/refresh-reference?clean={str(clean).lower()}",
                                             data=b"", method="POST")
            with urllib.request.urlopen(request, timeout=120) as response:
                selected = json.load(response)
            save(voice_id, reference_source_start=selected["reference_source_start"],
                 reference_duration=selected["reference_duration"])
        except OSError as exc:
            save(voice_id, reference_warning=f"Không cập nhật được mẫu giọng tức thì: {exc}")
        total = sum(manifest["counts"].values())
        if clean and (manifest["usable_seconds"] < 300 or total < 50):
            save(voice_id, status="needs_more_data", stage="review", dataset=str(dataset),
                 source_seconds=manifest["source_seconds"], usable_seconds=manifest["usable_seconds"],
                 train_clips=total, rejected=manifest["rejected"],
                 reason="Sau khi lọc đúng người còn dưới 5 phút hoặc 50 đoạn lời nói. Hãy thu/tải thêm giọng sạch và sửa transcript trước khi fine-tune lại; adapter cũ vẫn được giữ.")
            return 0
        # About 20 passes through accepted training clips.
        steps = max(100, math.ceil(manifest["counts"]["train"] * 20 / (14 * 2)))
        save(voice_id, status="waiting_for_gpu", stage="queued", dataset=str(dataset),
             source_seconds=manifest["source_seconds"], usable_seconds=manifest["usable_seconds"],
             asr_device=manifest.get("asr_device", "unknown"),
             train_minutes=manifest["minutes"]["train"], clips=total,
             train_clips=manifest["counts"]["train"], validation_clips=manifest["counts"]["validation"],
             test_clips=manifest["counts"]["test"], rejected=manifest["rejected"],
             rejected_speaker_segments=[], steps=steps, full_training=True,
             clean_before_training=clean)
        with lock_file.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            evaluated_run = prior_evaluation(manifest)
            if evaluated_run is None and total >= 22:
                eval_dataset = DATA / "prepared" / f"upload_{voice_id}_precheck_v2"
                eval_manifest = evaluation_dataset(dataset, eval_dataset)
                evaluated_run = f"upload_{voice_id}_precheck_r{attempt}"
                eval_run_dir = DATA / "training_runs" / evaluated_run
                save(voice_id, status="evaluating", stage="preliminary_training", evaluation_run=evaluated_run)
                command("preliminary_train", [str(PYTHON), "-m", "training.run", "--dataset", str(eval_dataset),
                                               "--run", evaluated_run, "--steps", "100", "--batch-size", "10",
                                               "--grad-accum", "2", "--rank", "32", "--target", "all", "--lr", "0.0001"], work)
                command("preliminary_evaluate", [str(PYTHON), "-m", "training.evaluate", "--run-dir", str(eval_run_dir),
                                                  "--train-python", str(TRAIN_PYTHON), "--count", str(eval_manifest["counts"]["test"])], work)
            if evaluated_run and clean:
                preliminary = json.loads((DATA / "training_runs" / evaluated_run / "status.json").read_text(encoding="utf-8"))
                gap = preliminary.get("generated_wer", 1) - preliminary.get("reference_wer", 0)
                if preliminary.get("test_items", 0) >= 5 and gap > .20:
                    save(voice_id, status="needs_review", stage="precheck", evaluation_run=evaluated_run,
                         pre_final_generated_wer=preliminary.get("generated_wer"),
                         pre_final_reference_wer=preliminary.get("reference_wer"),
                         reason="Bản train thử đọc sai hơn bản gốc trên tập câu tách riêng. Đã dừng trước khi tạo/thay adapter; hãy nghe lại mẫu, sửa transcript và bổ sung lời sạch.")
                    return 0
            successful = None
            for batch in (14, 12, 10, 8):
                run_name = f"upload_{voice_id}_full_b{batch}_r{attempt}"
                run_dir = DATA / "training_runs" / run_name
                save(voice_id, status="training", stage="training", batch_size=batch, run=run_name,
                     training_started_at=now())
                try:
                    command(f"train_b{batch}", [str(PYTHON), "-m", "training.run", "--dataset", str(dataset),
                            "--run", run_name, "--steps", str(steps), "--batch-size", str(batch),
                            "--grad-accum", "2", "--rank", "32", "--target", "all", "--lr", "0.0001"], work)
                    result = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
                    if not result.get("adapter_reload_verified"):
                        raise RuntimeError("Adapter reload verification missing")
                    successful = run_name
                    break
                except Exception:
                    log = run_dir / "training.log"
                    if log.is_file() and "out of memory" in log.read_text(errors="replace").lower():
                        continue
                    raise
            if successful is None:
                raise RuntimeError("GPU hết bộ nhớ ở mọi batch size 14, 12, 10, 8")
            run_name = successful
            run_dir = DATA / "training_runs" / run_name
            result = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            if result.get("status") != "trained_pending_evaluation" or result.get("enrollment_clips") != total:
                raise RuntimeError("Full-data training or enrollment incomplete")
            result.update(status="full_trained", evaluation_kind="pre_final_disjoint_split" if evaluated_run else "none",
                          evaluation_run=evaluated_run, trained_clips=total,
                          trained_seconds=manifest["usable_seconds"])
            if evaluated_run:
                prior = json.loads((DATA / "training_runs" / evaluated_run / "status.json").read_text())
                result["pre_final_generated_wer"] = prior.get("generated_wer")
                result["pre_final_reference_wer"] = prior.get("reference_wer")
            (run_dir / "status.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            trained_voice = register(run_name)
            save(voice_id, status="complete", stage="complete", finished_at=now(), trained_voice_id=trained_voice["id"],
                 optimizer_updates=result["optimizer_updates"], adapter_sha256=result["adapter_sha256"],
                 peak_gpu_mib=result.get("peak_gpu_mib"), evaluation_run=evaluated_run,
                 trained_clips=total, trained_seconds=manifest["usable_seconds"],
                 pre_final_generated_wer=result.get("pre_final_generated_wer"),
                 generated_wer=None, reference_wer=None, speaker_similarity=None,
                 model_selection=None, previous_run=None)
            return 0
    except Exception as exc:
        save(voice_id, status="failed", stage="failed", finished_at=now(), error=str(exc))
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice", required=True)
    args = parser.parse_args()
    return run(args.voice)


if __name__ == "__main__":
    raise SystemExit(main())
