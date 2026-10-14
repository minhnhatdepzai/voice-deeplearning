"""Sentence-level delivery planning and local audio shaping for TTS output."""
from __future__ import annotations

import array
import json
import math
import re
import subprocess
import tempfile
import wave
from pathlib import Path

PRESETS = {
    "natural": {"pitch": 0.0, "pace": 1.0, "gain": 0.0, "pause": 180},
    "mc": {"pitch": 0.35, "pace": 1.06, "gain": 1.0, "pause": 300},
    "presenter": {"pitch": 0.0, "pace": 0.97, "gain": 0.5, "pause": 420},
    "storyteller": {"pitch": -0.25, "pace": 0.91, "gain": -0.3, "pause": 540},
}


def plan(text: str, delivery: str | None) -> list[dict]:
    config = json.loads(delivery) if delivery else {}
    if not isinstance(config, dict) or config.get("mode", "natural") not in PRESETS:
        raise ValueError("Chế độ đọc không hợp lệ.")
    mode = config.get("mode", "natural")
    parts = [part.strip() for part in text.splitlines() if part.strip()]
    if not parts or len(parts) > 24:
        raise ValueError("Lời thoại cần từ 1 đến 24 dòng; mỗi dòng là một đoạn điều khiển.")
    overrides = config.get("segments", [])
    if not isinstance(overrides, list) or len(overrides) > 24:
        raise ValueError("Điều khiển từng đoạn không hợp lệ.")
    result = []
    for index, part in enumerate(parts):
        if len(part) > 500:
            raise ValueError("Mỗi đoạn lời thoại tối đa 500 ký tự.")
        item = overrides[index] if index < len(overrides) else {}
        if not isinstance(item, dict):
            raise ValueError("Điều khiển từng đoạn không hợp lệ.")
        defaults = PRESETS[mode]
        pitch = float(item.get("pitch", 0)) + defaults["pitch"]
        pace = float(item.get("pace", 1)) * defaults["pace"]
        gain = float(item.get("gain", 0)) + defaults["gain"]
        pause = int(item.get("pause", defaults["pause"]))
        contour = item.get("contour", "level")
        at = item.get("at")
        if at is not None:
            if isinstance(at, bool) or not isinstance(at, (int, float)) or not math.isfinite(at) or not 0 <= at <= 7200:
                raise ValueError("Mốc chèn phải từ 0 đến 7200 giây.")
        if not (-3 <= pitch <= 3 and .7 <= pace <= 1.3 and -6 <= gain <= 6 and 0 <= pause <= 1500 and contour in {"level", "rise", "fall"}):
            raise ValueError("Cao độ, tốc độ, âm lượng hoặc khoảng nghỉ nằm ngoài giới hạn.")
        result.append({"text": part, "pitch": pitch, "pace": pace, "gain": gain, "pause": pause, "contour": contour, "at": at})
    return result


def _samples(path: Path) -> tuple[int, array.array]:
    with wave.open(str(path), "rb") as stream:
        if stream.getnchannels() != 1 or stream.getsampwidth() != 2:
            raise ValueError("Audio tổng hợp cần mono PCM 16-bit.")
        rate = stream.getframerate()
        data = array.array("h")
        data.frombytes(stream.readframes(stream.getnframes()))
        return rate, data


def _write(path: Path, rate: int, data: array.array) -> None:
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(data.tobytes())


def _shape(source: Path, target: Path, pace: float, pitch: float, gain: float) -> None:
    factor = 2 ** (pitch / 12)
    filters = f"rubberband=tempo={pace:.5f}:pitch={factor:.6f}:formant=preserved,volume={gain:.3f}dB,aresample=24000"
    proc = subprocess.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(source), "-af", filters,
                           "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(target)], capture_output=True, timeout=120)
    if proc.returncode:
        raise ValueError(proc.stderr.decode(errors="replace")[-400:])


def combine(raw_files: list[Path], segments: list[dict], output: Path) -> dict:
    if len(raw_files) != len(segments):
        raise ValueError("Số đoạn âm thanh không khớp lời thoại.")
    result = array.array("h")
    markers = []
    rate = 24000
    with tempfile.TemporaryDirectory(prefix="vocalab-prosody-") as temp:
        directory = Path(temp)
        for index, (raw, settings) in enumerate(zip(raw_files, segments)):
            source_rate, samples = _samples(raw)
            if source_rate != rate:
                normalized = directory / f"normalized-{index}.wav"
                proc = subprocess.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(raw), "-ac", "1", "-ar", str(rate), str(normalized)], capture_output=True, timeout=120)
                if proc.returncode:
                    raise ValueError(proc.stderr.decode(errors="replace")[-400:])
                _, samples = _samples(normalized)
            start_sample = len(result)
            start = start_sample / rate
            if settings["contour"] == "level" or len(samples) < rate:
                shaped = directory / f"shaped-{index}.wav"
                _write(directory / f"input-{index}.wav", rate, samples)
                _shape(directory / f"input-{index}.wav", shaped, settings["pace"], settings["pitch"], settings["gain"])
                _, samples = _samples(shaped)
            else:
                cut = int(len(samples) * .72)
                head_in = directory / f"head-{index}.wav"
                tail_in = directory / f"tail-{index}.wav"
                head_out = directory / f"head-out-{index}.wav"
                tail_out = directory / f"tail-out-{index}.wav"
                _write(head_in, rate, samples[:cut])
                _write(tail_in, rate, samples[cut:])
                _shape(head_in, head_out, settings["pace"], settings["pitch"], settings["gain"])
                tail_shift = .8 if settings["contour"] == "rise" else -.8
                _shape(tail_in, tail_out, settings["pace"], settings["pitch"] + tail_shift, settings["gain"])
                _, head = _samples(head_out)
                _, tail = _samples(tail_out)
                overlap = min(int(rate * .025), len(head) // 4, len(tail) // 4)
                if overlap:
                    blend = array.array("h", (round(head[-overlap + k] * (1 - k / overlap) + tail[k] * (k / overlap)) for k in range(overlap)))
                    samples = head[:-overlap] + blend + tail[overlap:]
                else:
                    samples = head + tail
            result.extend(samples)
            markers.append({"text": settings["text"], "start": round(start, 3), "end": round(len(result) / rate, 3),
                            "start_sample": start_sample, "end_sample": len(result), "contour": settings["contour"]})
            if index < len(segments) - 1:
                result.extend(array.array("h", [0]) * round(settings["pause"] * rate / 1000))
    _write(output, rate, result)
    return {"segments": markers, "duration": round(len(result) / rate, 3)}
