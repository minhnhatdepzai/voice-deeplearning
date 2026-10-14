"""Score generated speech and original held-out audio with faster-whisper."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import unicodedata
from pathlib import Path

from training.pipeline import sha256_file
from training.gpu_whisper import make_whisper


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).casefold()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def distance(a: list[str], b: list[str]) -> int:
    previous = list(range(len(b) + 1))
    for i, item in enumerate(a, 1):
        current = [i]
        for j, other in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j-1] + (item != other)))
        previous = current
    return previous[-1]


def score(target: str, prediction: str) -> dict:
    t, p = normalize(target), normalize(prediction)
    words_t, words_p = t.split(), p.split()
    chars_t, chars_p = list(t.replace(" ", "")), list(p.replace(" ", ""))
    return {"word_errors": distance(words_t, words_p), "word_count": len(words_t),
            "char_errors": distance(chars_t, chars_p), "char_count": len(chars_t)}


def evaluate(run_dir: Path, train_python: Path, count: int) -> dict:
    run_dir = run_dir.resolve()
    state = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    if state["status"] not in {"trained_pending_evaluation", "evaluated"}:
        raise ValueError("Run has no completed trained adapter")
    if sha256_file(Path(state["adapter"])) != state["adapter_sha256"]:
        raise ValueError("Adapter hash changed after training")
    items_path = run_dir / "evaluation/items.json"
    if not items_path.is_file() or len(json.loads(items_path.read_text())) < count:
        command = [str(train_python), str(Path(__file__).with_name("generate_eval.py")), "--run-dir", str(run_dir), "--count", str(count)]
        subprocess.run(command, check=True)
    items = json.loads(items_path.read_text(encoding="utf-8"))[:count]
    model, asr_device = make_whisper("small")
    detailed = []
    for item in items:
        results = {}
        for key in ("reference_wav", "generated_wav"):
            segments, _ = model.transcribe(item[key], language=item["language"], vad_filter=True, beam_size=5)
            transcript = " ".join(segment.text.strip() for segment in segments)
            results[key] = {"transcript": transcript, **score(item["text"], transcript)}
        detailed.append({"test_file": item["test_file"], "target": item["text"], "language": item["language"], **results})
        print(f"scored {len(detailed)}/{len(items)}", flush=True)
    summary = {"run": state["run"], "source": state["source"], "speaker_id": state["speaker_id"],
               "adapter_sha256": state["adapter_sha256"], "test_items": len(detailed), "asr_model": f"faster-whisper small {asr_device}",
               "metrics": {}, "items": detailed}
    for label in ("reference_wav", "generated_wav"):
        word_errors = sum(row[label]["word_errors"] for row in detailed)
        word_count = sum(row[label]["word_count"] for row in detailed)
        char_errors = sum(row[label]["char_errors"] for row in detailed)
        char_count = sum(row[label]["char_count"] for row in detailed)
        summary["metrics"][label] = {"wer": round(word_errors / word_count, 4), "cer": round(char_errors / char_count, 4),
                                     "word_errors": word_errors, "word_count": word_count}
    (run_dir / "evaluation/results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    state["status"] = "evaluated"
    state["evaluation"] = str(run_dir / "evaluation/results.json")
    state["test_items"] = len(detailed)
    state["generated_wer"] = summary["metrics"]["generated_wav"]["wer"]
    state["reference_wer"] = summary["metrics"]["reference_wav"]["wer"]
    (run_dir / "status.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--train-python", type=Path, required=True)
    ap.add_argument("--count", type=int, default=5)
    args = ap.parse_args()
    print(json.dumps(evaluate(args.run_dir, args.train_python, args.count), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
