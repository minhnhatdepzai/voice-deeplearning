"""Train and evaluate an isolated candidate before replacing a library voice."""
from __future__ import annotations

import argparse
import fcntl
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from training.evaluate import evaluate
from training.run import run
from training.speaker_similarity import measure

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def stamp():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-f0-9]{32}", args.voice) or not re.fullmatch(r"[a-z0-9_]{1,20}", args.tag):
        parser.error("Invalid voice or tag")
    dataset = (DATA / "prepared" / args.dataset).resolve()
    if dataset.parent != (DATA / "prepared").resolve() or not (dataset / "manifest.json").is_file():
        parser.error("Dataset missing")
    status = DATA / "upload_training" / args.voice / f"{args.tag}_comparison.json"

    def save(**fields):
        state = json.loads(status.read_text(encoding="utf-8")) if status.is_file() else {}
        state.update(fields)
        temp = status.with_suffix(".tmp")
        temp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(status)

    lock_path = DATA / "auto_train/worker.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    save(status="waiting_for_gpu", dataset=str(dataset), error=None, finished_at=None)
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            save(status="training", started_at=stamp())
            selected = None
            for batch in (14, 12, 10, 8):
                name = f"upload_{args.voice}_{args.tag}_b{batch}"
                save(status="training", run=name, batch_size=batch)
                try:
                    run(dataset, DATA / "upstream/vieneu", 100, name, batch_size=batch,
                        grad_accum=2, rank=32, target="all", lr=1e-4)
                    selected = name
                    break
                except RuntimeError:
                    log = DATA / "training_runs" / name / "training.log"
                    if log.is_file() and "out of memory" in log.read_text(errors="replace").lower():
                        continue
                    raise
            if selected is None:
                raise RuntimeError("GPU hết bộ nhớ ở mọi batch size")
            directory = DATA / "training_runs" / selected
            save(status="evaluating")
            evaluate(directory, DATA / "upstream/vieneu/.venv/bin/python", count=1)
            similarity = measure(directory, count=1)
            result = json.loads((directory / "status.json").read_text(encoding="utf-8"))
            save(status="complete", finished_at=stamp(), generated_wer=result.get("generated_wer"),
                 speaker_similarity=similarity["metrics"], peak_gpu_mib=result.get("peak_gpu_mib"))
        except Exception as exc:
            save(status="failed", finished_at=stamp(), error=str(exc))
            raise


if __name__ == "__main__":
    main()
