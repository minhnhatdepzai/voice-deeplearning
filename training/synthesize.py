"""GPU inference for a trained single-speaker VieNeu model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vieneu import Vieneu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--voice", required=True)
    ap.add_argument("--text-file", type=Path)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--script-file", type=Path)
    ap.add_argument("--output-dir", type=Path)
    args = ap.parse_args()
    engine = Vieneu(mode="v3turbo", backend="pytorch", device="cuda", backbone_repo=str(args.model.resolve()))
    if args.script_file and args.output_dir:
        segments = json.loads(args.script_file.read_text(encoding="utf-8"))
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for index, segment in enumerate(segments):
            audio = engine.infer(segment["text"], voice=args.voice, use_ref_codes=False)
            engine.save(audio, str(args.output_dir / f"{index:02}.wav"))
    elif args.text_file and args.output:
        audio = engine.infer(args.text_file.read_text(encoding="utf-8"), voice=args.voice, use_ref_codes=False)
        engine.save(audio, str(args.output))
    else:
        ap.error("Provide either --script-file/--output-dir or --text-file/--output")


if __name__ == "__main__":
    main()
