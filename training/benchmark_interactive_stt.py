"""Compare interactive Whisper profiles on held-out, labeled speech."""
from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from training.evaluate import normalize, score
from training.gpu_whisper import make_whisper

ROOT = Path(__file__).resolve().parents[1]


PROFILES = {"fast": ("small", 1), "balanced": ("turbo", 3), "accurate": ("turbo", 5)}


def benchmark(dataset: Path, count: int, profiles: list[str]) -> dict:
    items = [json.loads(line) for line in (dataset / "test.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()][:count]
    if not items:
        raise ValueError("Dataset has no held-out test items")
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "dataset": str(dataset.resolve()),
              "items": len(items), "profiles": []}
    for profile in profiles:
        model_name, beam = PROFILES[profile]
        began = time.perf_counter()
        model, device = make_whisper(model_name)
        loaded_at = time.perf_counter()
        detailed = []
        for item in items:
            path = dataset / "raw_audio" / item["file_name"]
            item_began = time.perf_counter()
            segments, _ = model.transcribe(str(path), language=item["language"], vad_filter=True,
                                           vad_parameters={"min_silence_duration_ms": 350}, beam_size=beam,
                                           condition_on_previous_text=False)
            transcript = " ".join(segment.text.strip() for segment in segments)
            metrics = score(item["text"], transcript)
            detailed.append({"file": item["file_name"], "target": normalize(item["text"]),
                             "transcript": normalize(transcript), "seconds": round(time.perf_counter() - item_began, 3),
                             **metrics})
        words = sum(row["word_count"] for row in detailed)
        errors = sum(row["word_errors"] for row in detailed)
        report["profiles"].append({"profile": profile, "model": model_name, "beam_size": beam, "device": device,
                                   "load_seconds": round(loaded_at - began, 3),
                                   "inference_seconds": round(time.perf_counter() - loaded_at, 3),
                                   "wer": round(errors / words, 4), "word_errors": errors,
                                   "word_count": words, "details": detailed})
        del model
        gc.collect()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/prepared/vivos_VIVOSSPK15")
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--profiles", nargs="+", choices=sorted(PROFILES), default=["fast", "balanced", "accurate"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = benchmark(args.dataset, args.count, args.profiles)
    output = args.output or ROOT / "data/benchmarks" / f"interactive_stt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "items": result["items"],
                      "profiles": [{k: row[k] for k in ("profile", "model", "beam_size", "device", "load_seconds", "inference_seconds", "wer")}
                                   for row in result["profiles"]]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
