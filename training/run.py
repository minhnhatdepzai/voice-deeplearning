"""Run the official VieNeu-TTS single-speaker LoRA recipe on a prepared dataset."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from training.pipeline import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def run(dataset: Path, upstream: Path, steps: int, run_name: str, batch_size: int = 2,
        grad_accum: int = 2, rank: int = 16, target: str = "backbone", lr: float = 2e-4) -> dict:
    dataset = dataset.resolve()
    upstream = upstream.resolve()
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    if manifest["counts"].get("train", 0) < (1 if manifest.get("full_training") else 20) or (not manifest.get("full_training") and manifest["counts"].get("test", 0) < 1):
        raise ValueError("Dataset has too few train clips or no disjoint test split")
    if not (upstream / "finetune/train_lora.py").is_file():
        raise FileNotFoundError("Official VieNeu fine-tune scripts missing; run training/setup.sh")
    python = upstream / ".venv/bin/python"
    if not python.is_file():
        raise FileNotFoundError("Training Python missing; run training/setup.sh")
    output = ROOT / "data" / "training_runs" / run_name
    if output.exists():
        raise FileExistsError(f"Run name already exists: {run_name}")
    output.mkdir(parents=True)
    state = {"run": run_name, "dataset": str(dataset), "source": manifest["source_id"], "speaker_id": manifest["speaker_id"],
             "started_at": datetime.now(timezone.utc).isoformat(), "status": "preparing", "max_steps": steps,
             "upstream_commit": subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()}
    state["config"] = {"batch_size": batch_size, "grad_accum": grad_accum, "rank": rank, "target": target, "lr": lr}
    status_path = output / "status.json"

    def write_status():
        status_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    def call(label: str, command: list[str]):
        state["status"] = label
        write_status()
        stop = threading.Event()
        peak = [0]
        def sample_gpu():
            while not stop.wait(0.25):
                try:
                    result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                            capture_output=True, text=True, timeout=3)
                    peak[0] = max(peak[0], int(result.stdout.splitlines()[0].strip()))
                except (ValueError, IndexError, OSError, subprocess.TimeoutExpired):
                    pass
        monitor = threading.Thread(target=sample_gpu, daemon=True) if label == "training" else None
        if monitor:
            monitor.start()
        try:
            with (output / f"{label}.log").open("w", encoding="utf-8") as log:
                # Upstream safe_path confines data/output to the current directory.
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        finally:
            stop.set()
            if monitor:
                monitor.join(timeout=2)
                state["peak_gpu_mib"] = peak[0]
                write_status()
        if result.returncode:
            raise RuntimeError(f"{label} failed (exit {result.returncode}); see {output / (label + '.log')}")

    write_status()
    try:
        parquet = output / "train.parquet"
        call("encoding", [str(python), str(upstream / "finetune/prepare_dataset.py"), "--dataset-dir", str(dataset),
                          "--metadata", str(dataset / "train_metadata.csv"), "--audio-dir", str(dataset / "raw_audio"),
                          "--out", str(parquet)])
        official_output = output / "official"
        command = [str(python), str(upstream / "finetune/train_lora.py"), "--data", str(parquet), "--run", run_name,
                   "--output-dir", str(official_output), "--max-steps", str(steps), "--batch-size", str(batch_size),
                   "--grad-accum", str(grad_accum), "--r", str(rank), "--alpha", str(rank * 2),
                   "--target", target, "--lr", str(lr), "--log-every", "1", "--eval-every", str(max(1, steps // 5)),
                   "--save-every", str(max(2, steps // 2)), "--merge"]
        call("training", command)
        adapter = official_output / run_name / "adapter" / "adapter_model.safetensors"
        merged = official_output / run_name / "merged"
        log = official_output / run_name / "train_log.jsonl"
        if not adapter.is_file() or not merged.is_dir() or not log.is_file():
            raise RuntimeError("Training exited without a complete adapter, log and merged model")
        steps_logged = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
        if not any(record.get("step", 0) > 0 and "loss" in record for record in steps_logged):
            raise RuntimeError("No optimizer update recorded")
        train_rows = [json.loads(line) for line in (dataset / "train.jsonl").read_text(encoding="utf-8").splitlines()]
        reference = next((row for row in train_rows if 3 <= row["duration"] <= 8), train_rows[0])
        if manifest.get("full_training"):
            call("enrolling", [str(python), "-m", "training.enroll_all", "--parquet", str(parquet),
                               "--merged", str(merged), "--name", manifest["speaker_id"],
                               "--description", f"{manifest['source_id']} all utterances"])
        else:
            call("enrolling", [str(python), str(upstream / "finetune/make_voice.py"),
                               "--audio", str(dataset / "raw_audio" / reference["file_name"]),
                               "--name", manifest["speaker_id"], "--description", f"{manifest['source_id']} research speaker",
                               "--out", str(merged)])
        state["preset_sha256"] = sha256_file(merged / "voices_v3_turbo.json")
        reload_dir = output / "reload_check"
        call("reload_check", [str(python), str(upstream / "finetune/merge_lora.py"),
                              "--adapter", str(adapter.parent), "--out", str(reload_dir)])
        merged_hash = sha256_file(merged / "update/model.safetensors")
        reloaded_hash = sha256_file(reload_dir / "update/model.safetensors")
        if merged_hash != reloaded_hash:
            raise RuntimeError("Cold-reloaded adapter does not match the training merge")
        shutil.rmtree(reload_dir)
        state.update({"status": "trained_pending_evaluation", "full_training": bool(manifest.get("full_training")),
                      "enrollment_clips": len(train_rows) if manifest.get("full_training") else 1,
                      "adapter": str(adapter), "adapter_sha256": sha256_file(adapter),
                      "merged": str(merged), "optimizer_updates": max(record.get("step", 0) for record in steps_logged),
                      "reference_wav": str(dataset / "raw_audio" / reference["file_name"]),
                      "adapter_reload_verified": True, "merged_sha256": merged_hash,
                      "finished_at": datetime.now(timezone.utc).isoformat()})
        write_status()
        return state
    except Exception as exc:
        state.update({"status": "failed", "error": str(exc), "finished_at": datetime.now(timezone.utc).isoformat()})
        write_status()
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--upstream", type=Path, default=Path(os.environ.get("VIENEU_UPSTREAM", ROOT / "data/upstream/vieneu")))
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--run", required=True)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--target", choices=["backbone", "all"], default="backbone")
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args()
    if not 1 <= args.steps <= 10000 or not __import__("re").fullmatch(r"[a-zA-Z0-9_-]{1,64}", args.run):
        ap.error("steps must be 1-10000 and run name alphanumeric")
    if not 1 <= args.batch_size <= 16 or not 1 <= args.grad_accum <= 32 or args.rank not in {8, 16, 32, 64}:
        ap.error("batch-size, grad-accum or rank out of range")
    print(json.dumps(run(args.dataset, args.upstream, args.steps, args.run, args.batch_size,
                         args.grad_accum, args.rank, args.target, args.lr), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
