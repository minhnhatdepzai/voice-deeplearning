"""Build licensed two-speaker interview benchmarks and fine-tune datasets."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import lfilter, resample_poly

ROOT = Path(__file__).resolve().parents[1]
RATE = 24000


def _rows(dataset: Path, split: str) -> list[dict]:
    return [json.loads(line) for line in (dataset / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _eligible(dataset: Path, split: str = "train") -> list[dict]:
    return [row for row in _rows(dataset, split) if 3 <= row["duration"] <= 15]


def _audio(dataset: Path, row: dict) -> np.ndarray:
    audio, rate = sf.read(dataset / "raw_audio" / row["file_name"], dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if rate != RATE:
        audio = resample_poly(audio, RATE, rate).astype(np.float32)
    return audio


def _mix_at(track: np.ndarray, audio: np.ndarray, start: int) -> None:
    end = min(len(track), start + len(audio))
    track[start:end] += audio[:end - start]


def _colored_noise(length: int, rng: np.random.Generator) -> np.ndarray:
    white = rng.normal(0, 1, length).astype(np.float32)
    rumble = lfilter([1], [1, -.985], white).astype(np.float32)
    rumble /= max(float(np.std(rumble)), 1e-6)
    traffic = .65 * rumble + .35 * white
    time = np.arange(length, dtype=np.float32) / RATE
    hum = np.sin(2 * np.pi * 60 * time) + .35 * np.sin(2 * np.pi * 120 * time)
    noise = traffic + .28 * hum
    for start_seconds in np.arange(17, length / RATE, 41):
        start = int(start_seconds * RATE)
        width = min(int(.8 * RATE), length - start)
        if width > 0:
            envelope = np.hanning(width).astype(np.float32)
            noise[start:start + width] += 1.8 * envelope * np.sin(2 * np.pi * 430 * np.arange(width) / RATE)
    return noise.astype(np.float32)


def _with_noise(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    speech_rms = float(np.sqrt(np.mean(clean * clean)))
    noise_rms = float(np.sqrt(np.mean(noise * noise)))
    scaled = noise * (speech_rms / max(noise_rms, 1e-6)) / (10 ** (snr_db / 20))
    result = clean + scaled
    peak = float(np.max(np.abs(result)))
    return result * (.96 / peak) if peak > .96 else result


def _trim_codec_tail(audio: np.ndarray, rate: int) -> np.ndarray:
    """Remove systematic silent tails and end on a whole 80 ms codec frame."""
    window = max(1, round(rate * .01))
    usable = len(audio) // window * window
    frames = audio[:usable].reshape(-1, window) if usable else np.empty((0, window))
    rms = np.sqrt(np.mean(frames * frames, axis=1)) if len(frames) else np.array([])
    threshold = max(.0012, float(np.percentile(rms, 75)) * .10) if len(rms) else .0012
    active = np.flatnonzero(rms >= threshold)
    if not len(active):
        return audio
    last_active = min(len(audio), (int(active[-1]) + 1) * window)
    codec_frame = round(rate * .08)  # 3840 samples after resampling to 48 kHz.
    end = max(round(3 * rate), int(np.ceil(last_active / codec_frame)) * codec_frame)
    end = min(end, int(np.ceil(len(audio) / codec_frame)) * codec_frame)
    result = np.zeros(end, dtype=np.float32)
    result[:min(end, len(audio))] = audio[:end]
    return result


def generate(output: Path, main_dataset: Path, other_dataset: Path) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    main_rows = _eligible(main_dataset)[:80]
    other_rows = _eligible(other_dataset)[:20]
    events, cursor = [], 0.5
    main_index = other_index = 0
    cycle = 0
    while main_index < len(main_rows):
        if other_index < len(other_rows):
            row = other_rows[other_index]
            start = cursor
            end = start + row["duration"]
            events.append({"role": "interviewer", "speaker_id": row["speaker_id"], "start": start,
                           "end": end, "row": row})
            other_index += 1
            cursor = end + .28
        for _ in range(4):
            if main_index >= len(main_rows):
                break
            row = main_rows[main_index]
            overlap = .22 if cycle % 5 == 0 and _ == 0 else 0
            start = max(0, cursor - overlap)
            end = start + row["duration"]
            events.append({"role": "guest", "speaker_id": row["speaker_id"], "start": start,
                           "end": end, "row": row})
            main_index += 1
            cursor = end + .22
        cycle += 1
    total_samples = int((cursor + .5) * RATE)
    main_track = np.zeros(total_samples, dtype=np.float32)
    other_track = np.zeros(total_samples, dtype=np.float32)
    for event in events:
        dataset = main_dataset if event["role"] == "guest" else other_dataset
        audio = _audio(dataset, event["row"])
        event["end"] = event["start"] + len(audio) / RATE
        _mix_at(main_track if event["role"] == "guest" else other_track, audio, round(event["start"] * RATE))
    clean = main_track + other_track
    peak = float(np.max(np.abs(clean)))
    if peak > .96:
        scale = .96 / peak
        clean *= scale; main_track *= scale; other_track *= scale
    rng = np.random.default_rng(20261022)
    noise = _colored_noise(len(clean), rng)
    variants = {"clean": clean, "moderate_snr18": _with_noise(clean, noise, 18),
                "street_snr10": _with_noise(clean, noise, 10)}
    sf.write(output / "guest_truth.wav", main_track, RATE, subtype="PCM_16")
    sf.write(output / "interviewer_truth.wav", other_track, RATE, subtype="PCM_16")
    for name, audio in variants.items():
        sf.write(output / f"interview_{name}.wav", audio, RATE, subtype="PCM_16")
    serializable = []
    for event in events:
        serializable.append({"role": event["role"], "speaker_id": event["speaker_id"],
                             "start": round(event["start"], 3), "end": round(event["end"], 3),
                             "duration": round(event["end"] - event["start"], 3),
                             "file_name": event["row"]["file_name"], "text": event["row"]["text"]})
    guest_seconds = sum(row["duration"] for row in serializable if row["role"] == "guest")
    interviewer_seconds = sum(row["duration"] for row in serializable if row["role"] == "interviewer")
    result = {"license": "LibriTTS CC BY 4.0", "main_dataset": str(main_dataset.resolve()),
              "other_dataset": str(other_dataset.resolve()), "duration": round(len(clean) / RATE, 2),
              "guest_seconds": round(guest_seconds, 2), "interviewer_seconds": round(interviewer_seconds, 2),
              "expected_guest_share": round(guest_seconds / (guest_seconds + interviewer_seconds), 4),
              "events": serializable}
    (output / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _link_or_copy(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        import shutil
        shutil.copyfile(source, target)


def build_dataset(interview: Path, extracted: Path, output: Path, include_remaining_clean: bool) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Dataset already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    audio_dir = output / "raw_audio"
    audio_dir.mkdir()
    source_manifest = json.loads((interview / "manifest.json").read_text(encoding="utf-8"))
    main_dataset = Path(source_manifest["main_dataset"])
    target, target_rate = sf.read(extracted, dtype="float32")
    if target.ndim > 1:
        target = np.mean(target, axis=1)
    encoder = None
    from vieneu._v3_turbo_engine.speaker import OnnxSpeakerEncoder
    encoder = OnnxSpeakerEncoder.from_pretrained("pnnbao-ump/VieNeu-TTS-v3-Turbo")
    train, similarities, used_files = [], [], set()
    main_lookup = {row["file_name"]: row for row in _eligible(main_dataset)}
    for index, event in enumerate(source_manifest["events"]):
        if event["role"] != "guest":
            continue
        start = max(0, round((event["start"] - .04) * target_rate))
        end = min(len(target), round((event["end"] + .04) * target_rate))
        audio = target[start:end]
        audio = _trim_codec_tail(audio, target_rate)
        if len(audio) / target_rate < 3 or float(np.sqrt(np.mean(audio * audio))) < .002:
            continue
        original_row = main_lookup[event["file_name"]]
        original, original_rate = sf.read(main_dataset / "raw_audio" / event["file_name"], dtype="float32")
        a = np.asarray(encoder.embed(audio, target_rate), dtype=np.float32)
        b = np.asarray(encoder.embed(original, original_rate), dtype=np.float32)
        cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        if cosine < .62:
            continue
        if target_rate != RATE:
            audio = resample_poly(audio, RATE, target_rate).astype(np.float32)
        filename = f"interview_{index:04d}.wav"
        sf.write(audio_dir / filename, audio, RATE, subtype="PCM_16")
        train.append({"file_name": filename, "text": event["text"], "speaker_id": "8297",
                      "language": "en", "source": "libritts_interview", "duration": round(len(audio) / RATE, 3),
                      "split": "train", "extraction_cosine": round(cosine, 4),
                      "reference_score": round(cosine, 4)})
        similarities.append(cosine); used_files.add(event["file_name"])
    if include_remaining_clean:
        for row in _eligible(main_dataset):
            if row["file_name"] in used_files:
                continue
            filename = "clean_" + row["file_name"]
            clean_audio = _trim_codec_tail(_audio(main_dataset, row), RATE)
            sf.write(audio_dir / filename, clean_audio, RATE, subtype="PCM_16")
            train.append({**row, "file_name": filename, "duration": round(len(clean_audio) / RATE, 3),
                          "split": "train", "source": "libritts_clean_complement", "reference_score": 1.0})
    splits = {"train": train}
    for split in ("validation", "test"):
        rows = [row for row in _rows(main_dataset, split) if 3 <= row["duration"] <= 15]
        copied = []
        for row in rows:
            filename = f"{split}_{row['file_name']}"
            _link_or_copy(main_dataset / "raw_audio" / row["file_name"], audio_dir / filename)
            copied.append({**row, "file_name": filename, "split": split})
        splits[split] = copied
    for split, rows in splits.items():
        with (output / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output / "train_metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="|")
        writer.writerow(["file_name", "text"])
        writer.writerows((row["file_name"], row["text"]) for row in train)
    counts = {split: len(rows) for split, rows in splits.items()}
    minutes = {split: round(sum(row["duration"] for row in rows) / 60, 2) for split, rows in splits.items()}
    reference = max((row for row in train if 3 <= row["duration"] <= 8),
                    key=lambda row: (row.get("reference_score", 0), -abs(row["duration"] - 5.5)))
    manifest = {"source": {"name": "LibriTTS two-speaker interview benchmark", "language": "en",
                           "license": "CC BY 4.0", "original": "https://www.openslr.org/60/"},
                "source_id": "libritts_interview", "speaker_id": "8297", "counts": counts, "minutes": minutes,
                "reference_file": reference["file_name"],
                "extraction": {"source": str(extracted.resolve()), "accepted": len(similarities),
                               "cosine_mean": round(float(np.mean(similarities)), 4) if similarities else None,
                               "cosine_min": round(float(np.min(similarities)), 4) if similarities else None,
                               "includes_clean_complement": include_remaining_clean},
                "preparation": "Known LibriTTS transcript; target-speaker separation; 3-15 s clips; disjoint held-out test"}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("generate")
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--main", type=Path, default=ROOT / "data/prepared/libritts_8297")
    create.add_argument("--other", type=Path, default=ROOT / "data/prepared/libritts_3752")
    dataset = sub.add_parser("dataset")
    dataset.add_argument("--interview", type=Path, required=True)
    dataset.add_argument("--extracted", type=Path, required=True)
    dataset.add_argument("--output", type=Path, required=True)
    dataset.add_argument("--include-clean", action="store_true")
    args = parser.parse_args()
    result = (generate(args.output, args.main, args.other) if args.command == "generate" else
              build_dataset(args.interview, args.extracted, args.output, args.include_clean))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
