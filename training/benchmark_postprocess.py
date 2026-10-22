"""Measure whether lightweight output denoising harms speech or voice identity."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
from vieneu._v3_turbo_engine.speaker import OnnxSpeakerEncoder

from training.evaluate import score
from training.gpu_whisper import make_whisper
from training.speaker_similarity import similarity

ROOT = Path(__file__).resolve().parents[1]
FILTERS = {
    "original": None,
    "highpass": "highpass=f=55",
    "light_denoise": "highpass=f=55,afftdn=nr=6:nf=-35:tn=1",
    "strong_denoise": "highpass=f=70,afftdn=nr=12:nf=-30:tn=1",
}


def noise_floor(audio: np.ndarray, sample_rate: int) -> float:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    frame = max(1, round(sample_rate * .02))
    rms = [float(np.sqrt(np.mean(audio[i:i + frame] ** 2) + 1e-12)) for i in range(0, len(audio) - frame + 1, frame)]
    return round(20 * np.log10(max(1e-6, np.percentile(rms, 10))), 2) if rms else -120.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--voice", default="aaa22c2ba6ec450ca984819738fd5f02")
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    voices = json.loads((ROOT / "data/voices.json").read_text(encoding="utf-8"))
    voice = next(row for row in voices if row["id"] == args.voice)
    state = json.loads((ROOT / "data/training_runs" / voice["run"] / "status.json").read_text(encoding="utf-8"))
    encoder = OnnxSpeakerEncoder.from_pretrained(state["merged"])
    reference, reference_rate = sf.read(ROOT / "data/voices" / f"{voice['id']}.wav", dtype="float32")
    anchor = encoder.embed(reference, reference_rate)
    whisper, device = make_whisper("turbo")
    results = []
    output_dir = ROOT / "data/benchmarks/postprocess"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, audio_filter in FILTERS.items():
        rows = []
        for index, item in enumerate(report["rows"]):
            source = Path(item["audio"])
            target = source if audio_filter is None else output_dir / f"{args.report.stem}_{name}_{index:02}.wav"
            if audio_filter:
                subprocess.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(source),
                                "-af", audio_filter, str(target)], check=True)
            segments, _ = whisper.transcribe(str(target), language="vi", vad_filter=True, beam_size=3,
                                             condition_on_previous_text=False)
            transcript = " ".join(segment.text.strip() for segment in segments)
            audio, rate = sf.read(target, dtype="float32")
            rows.append({"score": score(item["target"], transcript),
                         "speaker_cosine": similarity(encoder.embed(audio, rate), anchor),
                         "noise_floor_dbfs": noise_floor(audio, rate)})
        errors = sum(row["score"]["word_errors"] for row in rows)
        words = sum(row["score"]["word_count"] for row in rows)
        results.append({"filter": name, "wer": round(errors / words, 4),
                        "speaker_cosine": round(float(np.mean([row["speaker_cosine"] for row in rows])), 4),
                        "noise_floor_dbfs": round(float(np.mean([row["noise_floor_dbfs"] for row in rows])), 2)})
    output = args.report.with_name(args.report.stem + "_postprocess.json")
    output.write_text(json.dumps({"source_report": str(args.report), "asr_device": device, "results": results,
                                  "limits": "Noise floor is a low-energy-frame proxy; listening tests are still required for naturalness."},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
