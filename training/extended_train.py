"""Long unattended single-speaker LoRA queue; continue after the shell exits."""
from __future__ import annotations

import fcntl
import json
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
WORK = DATA / "extended_train"
WORK.mkdir(parents=True, exist_ok=True)
PYTHON = ROOT / ".venv/bin/python"
TRAIN_PYTHON = DATA / "upstream/vieneu/.venv/bin/python"
PLAN = [
    ("ljspeech_LJ", "ljspeech_en_lj_extended", 600),
    ("vivos_VIVOSSPK17", "vivos_vi_spk17_extended", 250),
    ("vivos_VIVOSSPK18", "vivos_vi_spk18_extended", 250),
]
STATE = WORK / "status.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save(state: dict) -> None:
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(STATE)


def command(label: str, args: list[str]) -> None:
    path = WORK / f"{label}.log"
    print(f"[{now()}] START {label}", flush=True)
    with path.open("w", encoding="utf-8") as log:
        process = subprocess.run(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if process.returncode:
        raise RuntimeError(f"{label} exit {process.returncode}; see {path}")
    print(f"[{now()}] DONE {label}", flush=True)


def existing_best(source: str, speaker: str) -> float | None:
    values = []
    for file in (DATA / "training_runs").glob("*/status.json"):
        try:
            status = json.loads(file.read_text())
            if status.get("source") == source and status.get("speaker_id") == speaker and status.get("generated_wer") is not None:
                values.append(float(status["generated_wer"]))
        except (ValueError, OSError):
            continue
    return min(values) if values else None


def register(run: str) -> None:
    request = urllib.request.Request("http://127.0.0.1:8000/api/training/register",
                                     data=json.dumps({"run": run}).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=30) as response:
        json.load(response)


def main() -> int:
    with (DATA / "auto_train/worker.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = {"started_at": now(), "status": "running", "plan": [name for _, name, _ in PLAN], "runs": {}}
        save(state)
        for dataset_id, base_name, steps in PLAN:
            dataset = DATA / "prepared" / dataset_id
            row = {"dataset": dataset_id, "status": "waiting_for_dataset", "started_at": now(), "steps": steps}
            state["runs"][base_name] = row
            save(state)
            try:
                deadline = time.monotonic() + 7200
                while not (dataset / "manifest.json").is_file():
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"Dataset not ready: {dataset}")
                    time.sleep(10)
                manifest = json.loads((dataset / "manifest.json").read_text())
                best = existing_best(manifest["source_id"], manifest["speaker_id"])
                row["prior_best_wer"] = best
                row["status"] = "training"
                save(state)
                successful = None
                for batch in (14, 12, 10, 8):
                    run = f"{base_name}_b{batch}"
                    run_dir = DATA / "training_runs" / run
                    try:
                        if not (run_dir / "status.json").is_file():
                            command(f"{run}-train", [str(PYTHON), "-m", "training.run", "--dataset", str(dataset),
                                     "--steps", str(steps), "--run", run, "--batch-size", str(batch),
                                     "--grad-accum", "2", "--rank", "32", "--target", "all", "--lr", "0.0001"])
                        result = json.loads((run_dir / "status.json").read_text())
                        if not result.get("adapter_reload_verified"):
                            raise RuntimeError("Adapter reload check missing")
                        successful = run
                        break
                    except Exception:
                        log = run_dir / "training.log"
                        if log.is_file() and "out of memory" in log.read_text(errors="replace").lower():
                            print(f"[{now()}] OOM at batch {batch}; retrying smaller batch", flush=True)
                            continue
                        raise
                if successful is None:
                    raise RuntimeError("Training ran out of memory at all batch sizes")
                run = successful
                row.update({"run": run, "status": "evaluating"})
                save(state)
                run_dir = DATA / "training_runs" / run
                command(f"{run}-evaluate", [str(PYTHON), "-m", "training.evaluate", "--run-dir", str(run_dir),
                                             "--train-python", str(TRAIN_PYTHON), "--count", "20"])
                command(f"{run}-similarity", [str(PYTHON), "-m", "training.speaker_similarity", "--run-dir", str(run_dir), "--count", "20"])
                result = json.loads((run_dir / "status.json").read_text())
                row.update({"status": "complete", "run": run, "finished_at": now(),
                            "peak_gpu_mib": result.get("peak_gpu_mib"), "optimizer_updates": result.get("optimizer_updates"),
                            "adapter_sha256": result.get("adapter_sha256"), "generated_wer": result.get("generated_wer"),
                            "reference_wer": result.get("reference_wer"), "speaker_similarity": result.get("speaker_similarity")})
                if best is None or result["generated_wer"] <= best:
                    try:
                        register(run)
                        row["registered"] = True
                    except Exception as exc:
                        row["registration_error"] = str(exc)
                else:
                    row["registered"] = False
                    row["registration_reason"] = "Held-out WER worse than earlier adapter"
                print(f"[{now()}] COMPLETE {run}: WER {row['generated_wer']:.4f}, peak {row['peak_gpu_mib']} MiB", flush=True)
            except Exception as exc:
                row.update({"status": "failed", "finished_at": now(), "error": str(exc)})
                print(f"[{now()}] FAILED {base_name}: {exc}", flush=True)
            save(state)
        state.update({"status": "complete" if all(r["status"] == "complete" for r in state["runs"].values()) else "completed_with_errors",
                      "finished_at": now()})
        save(state)
        return 0 if state["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
