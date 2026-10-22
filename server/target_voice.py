"""Select a speaker from separated speech using an enrolled voice sample."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SEPARATOR_PYTHON = ROOT / "data/upstream/clearvoice/.venv/bin/python"
RATE = 16000
CHUNK = 12 * RATE
OVERLAP = RATE
STRIDE = CHUNK - OVERLAP
MAX_SECONDS = 60 * 60


def _decode(source: Path, target: Path, *, max_seconds: int | None = None) -> None:
    command = ["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(source)]
    if max_seconds is not None:
        command.extend(["-t", str(max_seconds)])
    command.extend(["-vn", "-ac", "1", "-ar", str(RATE), "-c:a", "pcm_f32le", str(target)])
    completed = subprocess.run(command, capture_output=True, timeout=300)
    if completed.returncode:
        raise ValueError("Không đọc được tiếng nói từ audio/video: " + completed.stderr.decode(errors="replace")[-300:])


def _embedding(encoder, audio: np.ndarray) -> np.ndarray | None:
    if len(audio) < RATE or float(np.sqrt(np.mean(audio * audio))) < .002:
        return None
    vector = np.asarray(encoder.embed(audio, RATE), dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else None


def _score(encoder, reference: np.ndarray, audio: np.ndarray) -> float:
    vector = _embedding(encoder, audio)
    return float(reference @ vector) if vector is not None else -1.0


def _read_chunk(path: Path, length: int) -> np.ndarray:
    audio, rate = sf.read(path, dtype="float32")
    if rate != RATE or audio.ndim != 1 or len(audio) < length:
        raise ValueError(f"Mô hình tách âm trả về đoạn không hợp lệ: {path.name}")
    return audio[:length]


def choose_stem(scores: list[float], source_similarity: float, *, allow_fallback: bool) -> int | None:
    """Return 0/1 for separated voice, -1 for denoised single speaker, or None."""
    winner = int(np.argmax(scores))
    best, other = scores[winner], scores[1 - winner]
    if best >= .58 and best - other >= .06:
        return winner
    if best >= .70 and other < .55:
        return winner
    if best >= .48 and source_similarity >= .60 and best - other >= .22:
        return winner
    if allow_fallback and source_similarity >= .78 and best < .48 and other < .30:
        # Two-stem models can damage a lone speaker; use denoised source.
        return -1
    return None


def process(source: Path, reference: Path, workspace: Path, *, enhance: bool = True,
            allow_single_speaker_fallback: bool = True) -> dict:
    """Write target/residual WAV. Identity decisions are conservative per 11-second chunk.

    A two-speaker model cannot guarantee complete removal with 3+ overlapping
    speakers; low-confidence chunks are left intact in the residual and silent
    in the target, so they cannot contaminate a later training set.
    """
    if not SEPARATOR_PYTHON.is_file():
        raise ValueError("Chưa cài ClearVoice. Chạy: bash training/setup_voice_filter.sh")
    source, reference, workspace = source.resolve(), reference.resolve(), workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    mixture_path = workspace / "mixture.wav"
    reference_path = workspace / "reference.wav"
    _decode(source, mixture_path)
    _decode(reference, reference_path, max_seconds=8)
    mixture, rate = sf.read(mixture_path, dtype="float32")
    sample, ref_rate = sf.read(reference_path, dtype="float32")
    if rate != RATE or ref_rate != RATE or mixture.ndim != 1 or sample.ndim != 1:
        raise ValueError("Audio đầu vào phải giải mã được về mono 16 kHz.")
    if len(sample) < 3 * RATE or len(mixture) < RATE:
        raise ValueError("Cần mẫu giọng ít nhất 3 giây và đoạn cần tách ít nhất 1 giây.")
    if len(mixture) > (MAX_SECONDS + 1) * RATE:
        raise ValueError("Mỗi lượt xử lý tối đa 60 phút audio/video.")
    from vieneu._v3_turbo_engine.speaker import OnnxSpeakerEncoder

    encoder = OnnxSpeakerEncoder.from_pretrained("pnnbao-ump/VieNeu-TTS-v3-Turbo")
    ref_vector = _embedding(encoder, sample[:8 * RATE])
    if ref_vector is None:
        raise ValueError("Mẫu tham chiếu quá yên lặng hoặc không có giọng nói rõ.")
    stem_dir = workspace / "stems"
    command = [str(SEPARATOR_PYTHON), str(ROOT / "training/separate_speakers.py"),
               "--input", str(mixture_path), "--output", str(stem_dir)]
    if enhance:
        command.append("--enhance")
    completed = subprocess.run(command, cwd=ROOT / "data/upstream/clearvoice", capture_output=True, timeout=14400,
                               env={**os.environ, "PYTHONUNBUFFERED": "1"})
    if completed.returncode:
        raise RuntimeError("Tách người nói thất bại: " + completed.stderr.decode(errors="replace")[-700:])
    target = np.zeros(len(mixture), dtype=np.float32)
    raw_target = np.zeros(len(mixture), dtype=np.float32)
    weight = np.zeros(len(mixture), dtype=np.float32)
    decisions = []
    for index, start in enumerate(range(0, len(mixture), STRIDE)):
        length = min(CHUNK, len(mixture) - start)
        if length < RATE:
            break
        stems = [_read_chunk(stem_dir / f"{index:05d}_{speaker}.wav", length) for speaker in range(2)]
        scores = [_score(encoder, ref_vector, stem) for stem in stems]
        raw_similarity = _score(encoder, ref_vector, mixture[start:start + length])
        winner = int(np.argmax(scores))
        best, other = scores[winner], scores[1 - winner]
        selected = choose_stem(scores, raw_similarity,
                               allow_fallback=allow_single_speaker_fallback and enhance)
        direct = selected == -1
        raw = stems[selected] if selected is not None and not direct else np.zeros(length, dtype=np.float32)
        cleaned = raw
        if direct:
            candidate = _read_chunk(stem_dir / f"{index:05d}_input_clean.wav", length)
            if _score(encoder, ref_vector, candidate) >= raw_similarity - .10:
                cleaned = raw = candidate
            else:
                selected = None
        elif enhance and selected is not None:
            candidate = _read_chunk(stem_dir / f"{index:05d}_{selected}_clean.wav", length)
            if _score(encoder, ref_vector, candidate) >= best - .08:
                cleaned = candidate
        fade = np.ones(length, dtype=np.float32)
        if start > 0:
            fade[:min(OVERLAP, length)] = np.linspace(0, 1, min(OVERLAP, length), endpoint=False)
        if start + length < len(mixture):
            fade[-min(OVERLAP, length):] = np.linspace(1, 0, min(OVERLAP, length), endpoint=False)
        target[start:start + length] += cleaned * fade
        raw_target[start:start + length] += raw * fade
        weight[start:start + length] += fade
        decisions.append({"start": round(start / RATE, 2), "end": round((start + length) / RATE, 2),
                          "selected_stem": selected, "similarity": round(float(best), 3),
                          "source_similarity": round(float(raw_similarity), 3),
                          "margin": round(float(best - other), 3)})
    valid = weight > 0
    target[valid] /= weight[valid]
    raw_target[valid] /= weight[valid]
    residual = mixture - raw_target
    target_path, residual_path = workspace / "target.wav", workspace / "without_target.wav"
    for audio in (target, residual):
        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        if peak > .98:
            audio *= .98 / peak
    sf.write(target_path, target, RATE)
    sf.write(residual_path, residual, RATE)
    result = {"target": str(target_path), "without_target": str(residual_path),
              "duration": round(len(mixture) / RATE, 2),
              "selected_chunks": sum(row["selected_stem"] is not None for row in decisions),
              "total_chunks": len(decisions), "decisions": decisions,
              "method": "MossFormer2_SS_16K + VieNeu speaker encoder" + (" + FRCRN_SE_16K" if enhance else "")}
    (workspace / "selection.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
