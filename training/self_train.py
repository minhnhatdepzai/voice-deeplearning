"""Persistent worker for a guided browser recording; waits for other GPU jobs."""
from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from training.self_recording import ROOT, load, now, write_state

DATA = ROOT / "data"
PYTHON = ROOT / ".venv/bin/python"
TRAIN_PYTHON = DATA / "upstream/vieneu/.venv/bin/python"


def update(session_id: str, **fields) -> dict:
    state = load(session_id)
    state.update(fields)
    write_state(state)
    return state


def command(label: str, args: list[str], log_dir: Path) -> None:
    with (log_dir / f"{label}.log").open("w", encoding="utf-8") as log:
        process = subprocess.run(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if process.returncode:
        raise RuntimeError(f"{label} failed with exit {process.returncode}; see {log_dir / (label + '.log')}")


def register(run: str) -> dict:
    request = urllib.request.Request("http://127.0.0.1:8000/api/training/register",
                                     data=json.dumps({"run": run}).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    last_error = None
    for _ in range(30):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except (OSError, urllib.error.HTTPError) as exc:
            last_error = exc
            time.sleep(10)
    raise RuntimeError(f"Could not register trained voice: {last_error}")


def run(session_id: str) -> int:
    state = load(session_id)
    if state["status"] not in {"queued", "waiting_for_gpu"}:
        raise ValueError("Session is not queued for training")
    directory = DATA / "self_training" / session_id
    dataset = Path(state["dataset"])
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    run_name = f"self_{session_id}"
    run_dir = DATA / "training_runs" / run_name
    steps = min(100, max(45, round(manifest["counts"]["train"] * 1.4)))
    update(session_id, status="waiting_for_gpu", run=run_name, steps=steps)
    with (DATA / "auto_train/worker.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            update(session_id, status="training", training_started_at=now())
            command("train", [str(PYTHON), "-m", "training.run", "--dataset", str(dataset),
                              "--run", run_name, "--steps", str(steps), "--batch-size", "8",
                              "--grad-accum", "2", "--rank", "16", "--target", "all", "--lr", "0.0001"], directory)
            trained = json.loads((run_dir / "status.json").read_text())
            if not trained.get("adapter_reload_verified"):
                raise RuntimeError("Adapter reload verification missing")
            count = min(10, manifest["counts"]["test"])
            update(session_id, status="evaluating")
            command("evaluate", [str(PYTHON), "-m", "training.evaluate", "--run-dir", str(run_dir),
                                 "--train-python", str(TRAIN_PYTHON), "--count", str(count)], directory)
            command("similarity", [str(PYTHON), "-m", "training.speaker_similarity", "--run-dir", str(run_dir),
                                   "--count", str(count)], directory)
            evaluated = json.loads((run_dir / "status.json").read_text())
            if evaluated.get("status") != "evaluated":
                raise RuntimeError("Held-out evaluation missing")
            voice = register(run_name)
            update(session_id, status="complete", finished_at=now(), voice_id=voice["id"],
                   adapter_sha256=evaluated["adapter_sha256"], optimizer_updates=evaluated["optimizer_updates"],
                   generated_wer=evaluated["generated_wer"], reference_wer=evaluated["reference_wer"],
                   speaker_similarity=evaluated.get("speaker_similarity"), test_items=evaluated["test_items"])
            return 0
        except Exception as exc:
            update(session_id, status="failed", finished_at=now(), error=str(exc))
            raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    args = parser.parse_args()
    return run(args.session)


if __name__ == "__main__":
    raise SystemExit(main())
