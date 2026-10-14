"""Unattended, restartable training queue for public Vietnamese/English speakers."""
from __future__ import annotations

import fcntl
import json
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
WORK = DATA / "auto_train"
WORK.mkdir(parents=True, exist_ok=True)
PYTHON = ROOT / ".venv/bin/python"
TRAIN_PYTHON = DATA / "upstream/vieneu/.venv/bin/python"
PLAN = [
    ("vivos", "VIVOSSPK17", 300, "vivos_vi_spk17_v1"),
    ("vivos", "VIVOSSPK18", 300, "vivos_vi_spk18_v1"),
    ("vivos", "VIVOSSPK19", 300, "vivos_vi_spk19_v1"),
    ("libritts", "3752", 272, "libritts_en_3752_v1"),
    ("libritts", "8297", 265, "libritts_en_8297_v1"),
]
ARCHIVES = {
    "vivos": DATA / "datasets/vivos/vivos-vietnamese-speech-corpus-for-asr.zip",
    "libritts": DATA / "datasets/libritts/dev-clean.tar.gz",
}
STATE = WORK / "status.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save(state: dict) -> None:
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(STATE)


def command(label: str, argv: list[str], timeout: int = 1800) -> None:
    path = WORK / f"{label}.log"
    print(f"[{now()}] START {label}", flush=True)
    with path.open("w", encoding="utf-8") as log:
        result = subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"{label}: exit {result.returncode}; see {path}")
    print(f"[{now()}] DONE {label}", flush=True)


def register(run: str) -> None:
    request = urllib.request.Request("http://127.0.0.1:8000/api/training/register",
                                     data=json.dumps({"run": run}).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=30) as response:
        voice = json.load(response)
    print(f"[{now()}] REGISTERED {voice['name']} {voice['id']}", flush=True)


def main() -> int:
    with (WORK / "worker.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another automatic training worker is already running", flush=True)
            return 1
        if not PYTHON.is_file() or not TRAIN_PYTHON.is_file():
            print("Training Python environment missing; run training/setup.sh first", flush=True)
            return 1
        state = {"started_at": now(), "status": "running", "plan": [row[3] for row in PLAN], "runs": {}}
        save(state)
        for source, speaker, max_clips, run in PLAN:
            row = {"source": source, "speaker": speaker, "started_at": now(), "status": "preparing"}
            state["runs"][run] = row
            save(state)
            dataset = DATA / "prepared" / f"{source}_{speaker}"
            run_dir = DATA / "training_runs" / run
            try:
                if not (dataset / "manifest.json").is_file():
                    command(f"{run}-prepare", [str(PYTHON), "-m", "training.pipeline", "prepare", "--source", source,
                                                  "--archive", str(ARCHIVES[source]), "--speaker", speaker,
                                                  "--output", str(dataset), "--max-clips", str(max_clips)])
                row["status"] = "training"
                save(state)
                current = json.loads((run_dir / "status.json").read_text()) if (run_dir / "status.json").is_file() else {}
                if not current:
                    command(f"{run}-train", [str(PYTHON), "-m", "training.run", "--dataset", str(dataset),
                                               "--steps", "90", "--run", run])
                    current = json.loads((run_dir / "status.json").read_text())
                if not current.get("adapter_reload_verified"):
                    raise RuntimeError(f"{run}: missing verified adapter reload")
                row["status"] = "evaluating"
                save(state)
                if current.get("status") != "evaluated":
                    command(f"{run}-evaluate", [str(PYTHON), "-m", "training.evaluate", "--run-dir", str(run_dir),
                                                  "--train-python", str(TRAIN_PYTHON), "--count", "10"])
                current = json.loads((run_dir / "status.json").read_text())
                if not current.get("speaker_similarity"):
                    command(f"{run}-similarity", [str(PYTHON), "-m", "training.speaker_similarity",
                                                    "--run-dir", str(run_dir), "--count", "10"])
                current = json.loads((run_dir / "status.json").read_text())
                register(run)
                row.update({"status": "complete", "finished_at": now(), "adapter_sha256": current["adapter_sha256"],
                            "test_items": current["test_items"], "reference_wer": current["reference_wer"],
                            "generated_wer": current["generated_wer"], "speaker_similarity": current["speaker_similarity"]})
                print(f"[{now()}] COMPLETE {run} WER={current['generated_wer']:.4f}", flush=True)
            except Exception as exc:
                row.update({"status": "failed", "finished_at": now(), "error": str(exc)})
                print(f"[{now()}] FAILED {run}: {exc}", flush=True)
            save(state)
        state["finished_at"] = now()
        state["status"] = "complete" if all(row["status"] == "complete" for row in state["runs"].values()) else "completed_with_errors"
        save(state)
        print(f"[{now()}] QUEUE {state['status']}", flush=True)
        return 0 if state["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
