"""Mix generated speech into an audio/video timeline at explicit timestamps."""
from __future__ import annotations

import subprocess
import wave
from pathlib import Path

import numpy as np

RATE = 24000
MODES = {"mute", "keep", "duck", "remove_vocals"}


def decode(source: Path, output: Path) -> None:
    result = subprocess.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(source), "-vn", "-ac", "1",
                             "-ar", str(RATE), "-c:a", "pcm_s16le", str(output)], capture_output=True, timeout=600)
    if result.returncode:
        raise ValueError(result.stderr.decode(errors="replace")[-400:])


def read(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getframerate() != RATE:
            raise ValueError("Audio phải là WAV mono 24 kHz, PCM 16-bit")
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


def write(path: Path, audio: np.ndarray) -> None:
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(RATE)
        wav.writeframes(pcm.tobytes())


def mix(speech: Path, markers: list[dict], positions: list[float | None], source: Path | None,
        output: Path, mode: str = "mute", background: Path | None = None) -> dict:
    if mode not in MODES or len(markers) != len(positions):
        raise ValueError("Chế độ âm thanh hoặc mốc thời gian không hợp lệ")
    generated = read(speech)
    original = read(background or source) if source is not None else np.zeros(0, dtype=np.float32)
    locations = []
    for marker, position in zip(markers, positions):
        if position is not None and (not np.isfinite(position) or position < 0 or position > 7200):
            raise ValueError("Mốc chèn phải từ 0 đến 7200 giây")
        start = int(round((marker["start"] if position is None else position) * RATE))
        a = max(0, min(len(generated), int(marker.get("start_sample", round(marker["start"] * RATE)))))
        b = max(a, min(len(generated), int(marker.get("end_sample", round(marker["end"] * RATE)))))
        if b <= a:
            raise ValueError("Một đoạn lời thoại không có mẫu âm thanh để phối.")
        locations.append((a, b, start))
    length = max([len(original), *(start + b - a for a, b, start in locations)], default=0)
    mixed = np.zeros(length, dtype=np.float32)
    if mode != "mute" and source is not None:
        mixed[:len(original)] = original
    if mode == "duck":
        for a, b, start in locations:
            end = min(start + b - a, len(original))
            if end > start:
                mixed[start:end] *= 0.18
    for a, b, start in locations:
        mixed[start:start + b - a] += generated[a:b]
    # Soft limit overlapping voices and source audio without harsh clipping.
    peak = float(np.max(np.abs(mixed))) if len(mixed) else 0.0
    if peak > 0.98:
        mixed *= 0.98 / peak
    write(output, mixed)
    return {"duration": round(length / RATE, 3), "placed_segments": [
        {"text": m["text"], "start": round(start / RATE, 3), "end": round((start + b - a) / RATE, 3)}
        for m, (a, b, start) in zip(markers, locations)], "source_mode": mode}
