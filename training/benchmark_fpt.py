"""Reproducible Vietnamese STT benchmark on the FPT Kaggle mirror.

The Kaggle ZIP repeats many MP3s under suffixed filenames. Only canonical
*.mp3 entries paired with transcriptAll.txt are used. This does not teach a
voice or claim speaker identity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from training.gpu_whisper import make_whisper

from training.evaluate import score
from training.pipeline import sha256_file


def benchmark(archive_path: Path, output: Path, per_set: int = 30) -> dict:
    with zipfile.ZipFile(archive_path) as archive:
        available = set(archive.namelist())
        grouped = defaultdict(list)
        for line in archive.read("transcriptAll.txt").decode("utf-8-sig").splitlines():
            fields = line.split("|", 2)
            if len(fields) != 3 or not fields[1].strip():
                continue
            filename, transcript, timing = fields
            member = "mp3/" + filename
            if not filename.endswith(".mp3") or member not in available:
                continue
            subset = "Set001" if "_Set001_" in filename else "Set002" if "_Set002_" in filename else None
            if subset:
                grouped[subset].append({"member": member, "text": transcript, "timing": timing})
        selected = []
        for subset in sorted(grouped):
            chosen = sorted(grouped[subset], key=lambda r: hashlib.sha256(r["member"].encode()).hexdigest())[:per_set]
            selected.extend([{**r, "subset": subset} for r in chosen])
        model, _ = make_whisper("small")
        results = []
        with tempfile.TemporaryDirectory(prefix="vocalab-fpt-") as temporary:
            temp = Path(temporary) / "audio.mp3"
            for index, row in enumerate(selected, 1):
                temp.write_bytes(archive.read(row["member"]))
                segments, _ = model.transcribe(str(temp), language="vi", vad_filter=True, beam_size=5)
                prediction = " ".join(segment.text.strip() for segment in segments)
                results.append({"member": row["member"], "subset": row["subset"], "target": row["text"],
                                "prediction": prediction, **score(row["text"], prediction)})
                if index % 10 == 0:
                    print(f"scored {index}/{len(selected)}", flush=True)
    metrics = {}
    for subset in ("Set001", "Set002", "all"):
        rows = results if subset == "all" else [r for r in results if r["subset"] == subset]
        word_errors = sum(r["word_errors"] for r in rows)
        word_count = sum(r["word_count"] for r in rows)
        char_errors = sum(r["char_errors"] for r in rows)
        char_count = sum(r["char_count"] for r in rows)
        metrics[subset] = {"items": len(rows), "wer": round(word_errors / word_count, 4),
                           "cer": round(char_errors / char_count, 4), "word_errors": word_errors, "word_count": word_count}
    report = {"source": "FPT Open Speech Dataset (FOSD), Mendeley DOI 10.17632/k9sxg2twv4.4",
              "original_url": "https://data.mendeley.com/datasets/k9sxg2twv4/4",
              "kaggle_mirror": "https://www.kaggle.com/datasets/thinh127/fpt-open-speech-dataset-fosd-vietnamese",
              "license": "FPT Public License and CC BY 4.0 on original Mendeley record; Kaggle metadata says unknown",
              "archive_sha256": sha256_file(archive_path), "created_at": datetime.now(timezone.utc).isoformat(),
              "asr_model": "faster-whisper small int8 CPU", "sampling": f"first {per_set} per subset by SHA-256 filename order",
              "metrics": metrics, "items": results}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--per-set", type=int, default=30)
    args = ap.parse_args()
    if not 1 <= args.per_set <= 1000:
        ap.error("per-set must be 1-1000")
    result = benchmark(args.archive, args.output, args.per_set)
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
