"""Run the real HTTP speech-to-speech path on held-out Vietnamese sentences."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf
from fastapi.testclient import TestClient
from vieneu._v3_turbo_engine.speaker import OnnxSpeakerEncoder

from server import app as server
from training.evaluate import score
from training.speaker_similarity import similarity

ROOT = Path(__file__).resolve().parents[1]


def benchmark(voice_id: str, dataset: Path, count: int, quality: str, tts_profile: str) -> dict:
    voice = server._get_voice(voice_id)
    items = [json.loads(line) for line in (dataset / "test.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()][:count]
    client = TestClient(server.app)
    warmup = client.post("/api/speech-to-speech/warmup", json={"voice_id": voice_id, "quality": quality, "language": "vi"})
    warmup.raise_for_status()
    warmup_data = warmup.json()

    config = server._warm_voice_config(voice)
    encoder = OnnxSpeakerEncoder.from_pretrained(str(config.get("model") or "pnnbao-ump/VieNeu-TTS-v3-Turbo"))
    anchor_audio, anchor_sr = sf.read(config["reference"], dtype="float32")
    anchor = encoder.embed(anchor_audio, anchor_sr)
    rows = []
    for item in items:
        source = dataset / "raw_audio" / item["file_name"]
        with source.open("rb") as handle:
            response = client.post("/api/speech-to-speech", data={"voice_id": voice_id, "language": "vi", "quality": quality,
                                                                    "tts_profile": tts_profile},
                                   files={"file": (source.name, handle, "audio/wav")})
        response.raise_for_status()
        result = response.json()
        generated = ROOT / "data/outputs" / Path(result["audio_url"]).name
        generated_rows, _, _ = server._transcribe(generated, "vi", model_name="turbo", beam_size=3)
        generated_text = " ".join(row["text"] for row in generated_rows)
        input_score = score(item["text"], result["text"])
        output_score = score(item["text"], generated_text)
        audio, sample_rate = sf.read(generated, dtype="float32")
        identity = similarity(encoder.embed(audio, sample_rate), anchor)
        rows.append({"file": item["file_name"], "target": item["text"], "recognized_input": result["text"],
                     "recognized_output": generated_text, "input_score": input_score, "output_score": output_score,
                     "speaker_cosine_to_voice_reference": round(identity, 4), "latency_ms": result["latency_ms"],
                     "input_seconds": result["input_seconds"], "output_seconds": round(len(audio) / sample_rate, 3),
                     "audio": str(generated)})
    client.post("/api/speech-to-speech/release")
    server._release_stt_model()

    def aggregate(label: str) -> float:
        errors = sum(row[label]["word_errors"] for row in rows)
        words = sum(row[label]["word_count"] for row in rows)
        return round(errors / words, 4)

    return {"created_at": datetime.now(timezone.utc).isoformat(), "voice_id": voice_id, "voice_name": voice["name"],
            "quality": quality, "tts_profile": tts_profile, "items": len(rows), "warmup": warmup_data,
            "input_wer": aggregate("input_score"), "generated_wer": aggregate("output_score"),
            "mean_speaker_cosine": round(sum(row["speaker_cosine_to_voice_reference"] for row in rows) / len(rows), 4),
            "mean_warm_latency_ms": round(sum(row["latency_ms"]["total"] for row in rows) / len(rows)),
            "rows": rows,
            "limits": "WER measures intelligibility and cosine is only a speaker-identity proxy; neither is a human naturalness score."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice", default="e9d48c38456f4132ac0cfaa46cf5afc2")
    parser.add_argument("--run", help="Benchmark an unregistered training run using --voice as its reference WAV")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/prepared/vivos_VIVOSSPK15")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--quality", choices=["fast", "balanced", "accurate"], default="balanced")
    parser.add_argument("--tts-profile", choices=["natural", "clear", "stable"], default="stable")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    original_get_voice = server._get_voice
    if args.run:
        candidate = {**original_get_voice(args.voice), "kind": "finetuned", "run": args.run,
                     "name": f"Candidate {args.run}"}
        server._get_voice = lambda voice_id: candidate if voice_id == args.voice else original_get_voice(voice_id)
    try:
        report = benchmark(args.voice, args.dataset, args.count, args.quality, args.tts_profile)
    finally:
        server._get_voice = original_get_voice
    report["run"] = args.run
    output = args.output or ROOT / "data/benchmarks" / f"speech_to_speech_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), **{key: report[key] for key in
          ("voice_name", "quality", "tts_profile", "items", "input_wer", "generated_wer", "mean_speaker_cosine", "mean_warm_latency_ms")}},
          ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
