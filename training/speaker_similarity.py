"""Measure held-out voice embedding similarity with VieNeu's frozen encoder."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from vieneu._v3_turbo_engine.speaker import OnnxSpeakerEncoder


def similarity(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def measure(run_dir: Path, count: int = 10) -> dict:
    run_dir = run_dir.resolve()
    status_path = run_dir / "status.json"
    state = json.loads(status_path.read_text(encoding="utf-8"))
    items = json.loads((run_dir / "evaluation/items.json").read_text(encoding="utf-8"))[:count]
    if not items:
        raise ValueError("No generated held-out audio")
    encoder = OnnxSpeakerEncoder.from_pretrained(state["merged"])
    enrollment = state.get("reference_wav")
    if not enrollment:
        train = [json.loads(line) for line in (Path(state["dataset"]) / "train.jsonl").read_text(encoding="utf-8").splitlines()]
        row = next((r for r in train if 3 <= r["duration"] <= 8), train[0])
        enrollment = str(Path(state["dataset"]) / "raw_audio" / row["file_name"])
    audio, sr = sf.read(enrollment, dtype="float32")
    anchor = encoder.embed(audio, sr)
    rows = []
    for item in items:
        original, original_sr = sf.read(item["reference_wav"], dtype="float32")
        generated, generated_sr = sf.read(item["generated_wav"], dtype="float32")
        ref_embedding = encoder.embed(original, original_sr)
        gen_embedding = encoder.embed(generated, generated_sr)
        rows.append({"test_file": item["test_file"],
                     "original_to_enrollment": round(similarity(ref_embedding, anchor), 4),
                     "generated_to_enrollment": round(similarity(gen_embedding, anchor), 4),
                     "generated_to_original": round(similarity(gen_embedding, ref_embedding), 4)})
    metrics = {key: round(sum(r[key] for r in rows) / len(rows), 4) for key in
               ("original_to_enrollment", "generated_to_enrollment", "generated_to_original")}
    report = {"run": state["run"], "encoder": "VieNeu v3 Turbo frozen 192d ONNX speaker encoder",
              "items": len(rows), "metrics": metrics, "samples": rows,
              "note": "Cosine similarity is a proxy for voice identity, not a human listening score."}
    (run_dir / "evaluation/speaker_similarity.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    state["speaker_similarity"] = metrics
    status_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--count", type=int, default=10)
    args = ap.parse_args()
    print(json.dumps(measure(args.run_dir, args.count), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
