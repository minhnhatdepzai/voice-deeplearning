"""CUDA inference from a verified speech reference for an uploaded voice."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vieneu import Vieneu


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--script-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.reference.is_file():
        parser.error("Reference WAV missing")
    segments = json.loads(args.script_file.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    engine = Vieneu(mode="v3turbo", backend="pytorch", device="cuda")
    for index, segment in enumerate(segments):
        audio = engine.infer(segment["text"], ref_audio=str(args.reference), denoise=True, use_ref_codes=True)
        engine.save(audio, str(args.output_dir / f"{index:02}.wav"))


if __name__ == "__main__":
    main()
