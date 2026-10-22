"""Prepare every usable utterance from an uploaded, authorized single-speaker file."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import wave
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from training.pipeline import sha256_file, text_key
from training.gpu_whisper import make_whisper

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data/upload_training"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def state_path(voice_id: str) -> Path:
    return WORK / voice_id / "status.json"


def load(voice_id: str) -> dict | None:
    path = state_path(voice_id)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def save(voice_id: str, **fields) -> dict:
    path = state_path(voice_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = load(voice_id) or {"voice_id": voice_id, "created_at": now()}
    state.update(fields)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return state


def word_groups(segments) -> list[tuple[float, float, str]]:
    """Use Whisper word times so long source segments do not truncate at 20 s."""
    groups = []
    for segment in segments:
        words = [word for word in (segment.words or []) if word.word.strip() and word.end > word.start]
        if not words:
            if segment.end > segment.start and segment.text.strip():
                groups.append((segment.start, segment.end, segment.text.strip()))
            continue
        current = []
        for word in words:
            if current and (word.end - current[0].start > 12 or word.start - current[-1].end > 1.0):
                groups.append((current[0].start, current[-1].end, "".join(w.word for w in current).strip()))
                current = []
            current.append(word)
        if current:
            groups.append((current[0].start, current[-1].end, "".join(w.word for w in current).strip()))
    # Merge short neighboring phrases within VieNeu's current 3-15 second recipe.
    merged = []
    for start, end, text in groups:
        if merged and end - merged[-1][0] <= 15 and start - merged[-1][1] <= .8 and merged[-1][1] - merged[-1][0] < 5:
            before = merged.pop()
            merged.append((before[0], end, before[2] + " " + text))
        else:
            merged.append((start, end, text))
    return merged


def representative_speech(rows: list[dict], audio_dir: Path) -> tuple[dict, float]:
    """Choose a clear clip near the dominant speaker, without removing any training clips."""
    import soundfile as sf
    from vieneu._v3_turbo_engine.speaker import OnnxSpeakerEncoder

    eligible = [row for row in rows if 3 <= row["duration"] <= 8]
    if not eligible:
        raise ValueError("Không tìm thấy câu nói rõ 3–8 giây để làm mẫu tức thì.")
    count = min(len(rows), 64)
    sample_indices = np.unique(np.linspace(0, len(rows) - 1, count, dtype=int))
    sampled = [rows[index] for index in sample_indices]
    encoder = OnnxSpeakerEncoder.from_pretrained("pnnbao-ump/VieNeu-TTS-v3-Turbo")
    embeddings = {}

    def embedding(row):
        key = row["file_name"]
        if key not in embeddings:
            audio, sr = sf.read(audio_dir / key, dtype="float32")
            vector = np.asarray(encoder.embed(audio, sr), dtype=np.float32).reshape(-1)
            embeddings[key] = vector / np.linalg.norm(vector)
        return embeddings[key]

    centroid = np.mean([embedding(row) for row in sampled], axis=0)
    centroid /= np.linalg.norm(centroid)
    candidates = [row for row in sampled if 3 <= row["duration"] <= 8] or eligible[:64]
    best = max(candidates, key=lambda row: float(embedding(row) @ centroid))
    return best, round(float(embedding(best) @ centroid), 4)


def transcribe_full(model, wav_path: Path, language: str, chunk_seconds: int = 300):
    """Decode bounded chunks, retaining words across every chunk boundary."""
    result = []
    with wave.open(str(wav_path), "rb") as source:
        rate = source.getframerate()
        duration = source.getnframes() / rate
        chunks = max(1, int(np.ceil(duration / chunk_seconds)))
        for index in range(chunks):
            core_start = index * chunk_seconds
            core_end = min(duration, (index + 1) * chunk_seconds)
            read_start = max(0, core_start - 3)
            read_end = min(duration, core_end + 3)
            source.setpos(round(read_start * rate))
            frames = source.readframes(round((read_end - read_start) * rate))
            chunk_path = wav_path.parent / f"asr_chunk_{index:06d}.wav"
            try:
                with wave.open(str(chunk_path), "wb") as chunk:
                    chunk.setnchannels(source.getnchannels())
                    chunk.setsampwidth(source.getsampwidth())
                    chunk.setframerate(rate)
                    chunk.writeframes(frames)
                segments, _ = model.transcribe(str(chunk_path), language=language, vad_filter=True,
                                               word_timestamps=True, beam_size=5,
                                               condition_on_previous_text=False)
                for segment in segments:
                    words = []
                    for word in segment.words or []:
                        midpoint = read_start + (word.start + word.end) / 2
                        if core_start <= midpoint and (midpoint < core_end or
                                                       (index == chunks - 1 and midpoint <= core_end)):
                            words.append(SimpleNamespace(start=read_start + word.start,
                                                         end=read_start + word.end, word=word.word))
                    if words:
                        result.append(SimpleNamespace(start=words[0].start, end=words[-1].end,
                                                      text="".join(w.word for w in words), words=words))
                    elif not segment.words:
                        midpoint = read_start + (segment.start + segment.end) / 2
                        if core_start <= midpoint < core_end:
                            result.append(SimpleNamespace(start=read_start + segment.start,
                                                          end=read_start + segment.end,
                                                          text=segment.text, words=[]))
            finally:
                chunk_path.unlink(missing_ok=True)
    return result


def build(voice: dict, source: Path, output: Path, progress=None) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Dataset already exists: {output}")
    output.mkdir(parents=True)
    normalized = output / "source_24k.wav"
    try:
        subprocess.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(source), "-vn", "-ac", "1",
                        "-ar", "24000", "-c:a", "pcm_s16le", str(normalized)], check=True, timeout=None)
        with wave.open(str(normalized), "rb") as wav:
            source_seconds = wav.getnframes() / wav.getframerate()
        if progress:
            progress(stage="transcribing", source_seconds=round(source_seconds, 1))
        model, asr_device = make_whisper("small")
        language = "vi" if voice["locale"] == "vi-VN" else "en"
        groups = word_groups(transcribe_full(model, normalized, language))
        if progress:
            progress(stage="segmenting", detected_groups=len(groups))
        (output / "raw_audio").mkdir()
        rows = []
        rejected = Counter()
        with wave.open(str(normalized), "rb") as wav:
            for index, (start, end, text) in enumerate(groups):
                start_frame = max(0, round((start - .08) * 24000))
                end_frame = min(wav.getnframes(), round((end + .08) * 24000))
                duration = (end_frame - start_frame) / 24000
                key = text_key(text)
                if not 3 <= duration <= 15:
                    rejected["duration"] += 1
                    continue
                if not key:
                    rejected["empty_transcript"] += 1
                    continue
                wav.setpos(start_frame)
                raw = wav.readframes(end_frame - start_frame)
                samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768
                rms = float(np.sqrt(np.mean(samples * samples)))
                clipping = float(np.mean(np.abs(samples) >= .98))
                if rms < .001:
                    rejected["silence"] += 1
                    continue
                filename = f"{index:06d}.wav"
                with wave.open(str(output / "raw_audio" / filename), "wb") as clip:
                    clip.setnchannels(1); clip.setsampwidth(2); clip.setframerate(24000)
                    clip.writeframes(raw)
                rows.append({"file_name": filename, "text": text.replace("|", ","),
                             "speaker_id": f"UPLOAD{voice['id'][:8].upper()}", "language": language,
                             "source": "upload", "source_start": round(start_frame / 24000, 3),
                             "source_end": round(end_frame / 24000, 3), "duration": round(duration, 3),
                             "source_sha256": hashlib.sha256(raw).hexdigest()})
                if progress and len(rows) % 25 == 0:
                    progress(stage="segmenting", clips=len(rows), usable_seconds=round(sum(row["duration"] for row in rows), 1))
        if len(rows) < 1:
            raise ValueError("Không tìm được lời nói có phiên âm trong file gốc.")
        for row in rows:
            row["split"] = "train"
        counts = {"train": len(rows), "validation": 0, "test": 0}
        for split in ("train", "validation", "test"):
            with (output / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
                for row in rows:
                    if row["split"] == split:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        with (output / "train_metadata.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="|")
            writer.writerow(["file_name", "text"])
            writer.writerows((row["file_name"], row["text"]) for row in rows if row["split"] == "train")
        minutes = {split: round(sum(row["duration"] for row in rows if row["split"] == split) / 60, 2)
                   for split in ("train", "validation", "test")}
        manifest = {"source": {"name": voice["source"], "language": language,
                               "license": "User-owned or authorized voice; consent confirmed at upload"},
                    "source_id": "upload", "voice_id": voice["id"], "archive": str(source.resolve()),
                    "archive_sha256": sha256_file(source), "display_name": voice["name"] + " · Fine-tune toàn file",
                    "speaker_id": f"UPLOAD{voice['id'][:8].upper()}", "locale": voice["locale"],
                    "source_seconds": round(source_seconds, 1), "usable_seconds": round(sum(row["duration"] for row in rows), 1),
                    "full_training": True, "speaker_filter": None,
                    "counts": dict(counts), "minutes": minutes, "rejected": dict(rejected),
                    "asr_device": asr_device,
                    "transcript": f"Automatic faster-whisper small {asr_device}; review transcript before relying on voice quality",
                    "preparation": "Entire source transcribed in bounded chunks; 3-15 second clips follow the current VieNeu recipe; no speaker filtering or holdout"}
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest
    finally:
        normalized.unlink(missing_ok=True)
