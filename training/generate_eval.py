"""Generate held-out sentences with a merged VieNeu model (training environment)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vieneu import Vieneu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--count", type=int, default=5)
    args = ap.parse_args()
    run_dir = args.run_dir.resolve()
    state = json.loads((run_dir / "status.json").read_text())
    dataset = Path(state["dataset"])
    rows = [json.loads(line) for line in (dataset / "test.jsonl").read_text(encoding="utf-8").splitlines()][:args.count]
    if not rows:
        raise SystemExit("No test rows")
    output = run_dir / "evaluation"
    output.mkdir(exist_ok=True)
    model = Vieneu(mode="v3turbo", backend="pytorch", device="cuda", backbone_repo=state["merged"])
    generated = []
    for i, row in enumerate(rows):
        wav = output / f"test_{i:02}.wav"
        audio = model.infer(row["text"], voice=state["speaker_id"], use_ref_codes=False, apply_watermark=False)
        model.save(audio, str(wav))
        generated.append({"test_file": row["file_name"], "text": row["text"], "generated_wav": str(wav),
                          "reference_wav": str(dataset / "raw_audio" / row["file_name"]), "language": row["language"]})
        print(f"generated {i+1}/{len(rows)}: {wav}", flush=True)
    (output / "items.json").write_text(json.dumps(generated, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
