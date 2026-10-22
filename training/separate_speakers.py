"""Run ClearVoice's two-speaker model in its isolated CUDA environment."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from clearvoice import ClearVoice


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--enhance", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    audio, rate = sf.read(args.input, dtype="float32")
    if rate != 16000 or audio.ndim != 1:
        parser.error("Expected mono 16 kHz WAV")
    separator = ClearVoice(task="speech_separation", model_names=["MossFormer2_SS_16K"])
    enhancer = ClearVoice(task="speech_enhancement", model_names=["FRCRN_SE_16K"]) if args.enhance else None
    chunk_samples = 12 * rate
    overlap = rate
    stride = chunk_samples - overlap
    for index, start in enumerate(range(0, len(audio), stride)):
        piece = audio[start:start + chunk_samples]
        if len(piece) < rate:
            # The previous chunk already covers a very short tail.
            break
        output = np.asarray(separator(piece[None, :], False), dtype=np.float32)
        if output.ndim != 3 or output.shape[0] != 2 or output.shape[1] != 1:
            raise RuntimeError(f"Unexpected separation shape: {output.shape}")
        for speaker in range(2):
            stem = output[speaker, 0, :len(piece)]
            sf.write(args.output / f"{index:05d}_{speaker}.wav", stem, rate)
            if enhancer is not None:
                cleaned = np.asarray(enhancer(stem[None, :], False), dtype=np.float32).reshape(-1)
                sf.write(args.output / f"{index:05d}_{speaker}_clean.wav", cleaned[:len(piece)], rate)
        if enhancer is not None:
            cleaned_mix = np.asarray(enhancer(piece[None, :], False), dtype=np.float32).reshape(-1)
            sf.write(args.output / f"{index:05d}_input_clean.wav", cleaned_mix[:len(piece)], rate)
        print(f"chunk {index + 1}", flush=True)


if __name__ == "__main__":
    main()
