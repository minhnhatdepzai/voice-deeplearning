"""Create one voice preset from every encoded training utterance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def enroll(parquet: Path, merged: Path, name: str, description: str) -> int:
    vectors = np.asarray(pq.read_table(parquet, columns=["speaker_embedding"])
                         .column("speaker_embedding").to_pylist(), dtype=np.float32)
    if vectors.ndim != 2 or len(vectors) == 0:
        raise ValueError("No speaker embeddings in encoded training data")
    if np.any(np.linalg.norm(vectors, axis=1) == 0):
        raise ValueError("Empty speaker embedding")
    # VieNeu conditions on the original embedding scale (roughly norm 12–15).
    # Unit-normalizing it causes unstable, repetitive speech at inference.
    mean = vectors.mean(axis=0)
    path = merged / "voices_v3_turbo.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"presets": {}}
    data.setdefault("presets", {})[name] = {
        "description": description, "gender": "",
        "speaker_emb": [round(float(value), 6) for value in mean], "codes": None,
    }
    data["default_voice"] = name
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return len(vectors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--merged", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--description", default="")
    args = parser.parse_args()
    print(f"Enrolled {enroll(args.parquet, args.merged, args.name, args.description)} utterances")
