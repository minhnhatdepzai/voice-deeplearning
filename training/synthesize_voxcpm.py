"""Render script segments with VoxCPM2 in its isolated Python environment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf
from voxcpm import VoxCPM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--locale", choices=["vi-VN", "en-US"], default="vi-VN")
    parser.add_argument("--script-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    segments = json.loads(args.script_file.read_text(encoding="utf-8"))
    model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
    for index, segment in enumerate(segments):
        instruction = "(Speak natural Vietnamese with clear native tones.)" if args.locale == "vi-VN" else "(Speak natural American English with a General American accent.)"
        audio = model.generate(text=instruction + segment["text"], reference_wav_path=str(args.reference),
                               cfg_value=2.0, inference_timesteps=10)
        sf.write(args.output_dir / f"{index:02}.wav", audio, model.tts_model.sample_rate)


if __name__ == "__main__":
    main()
