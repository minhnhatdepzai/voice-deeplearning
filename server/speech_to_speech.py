"""Shared validation and measurement helpers for interactive speech conversion."""
from __future__ import annotations

import re
from dataclasses import dataclass


MAX_SPEECH_TO_SPEECH_BYTES = 500 * 1024 * 1024


@dataclass(frozen=True)
class RecognitionProfile:
    model: str
    beam_size: int


@dataclass(frozen=True)
class SynthesisProfile:
    temperature: float
    top_k: int
    top_p: float
    repetition_penalty: float = 1.2


PROFILES = {
    "fast": RecognitionProfile("small", 1),
    "balanced": RecognitionProfile("turbo", 3),
    "accurate": RecognitionProfile("turbo", 5),
}

SYNTHESIS_PROFILES = {
    "natural": SynthesisProfile(0.8, 25, 0.95),
    "clear": SynthesisProfile(0.55, 20, 0.9),
    "stable": SynthesisProfile(0.35, 15, 0.85),
}

POSTPROCESS_FILTERS = {
    "off": None,
    "light": "highpass=f=55,afftdn=nr=6:nf=-35:tn=1",
    "strong": "highpass=f=70,afftdn=nr=12:nf=-30:tn=1",
}


def recognition_profile(name: str) -> RecognitionProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise ValueError("Chất lượng phải là fast, balanced hoặc accurate.") from exc


def synthesis_profile(name: str) -> dict:
    try:
        profile = SYNTHESIS_PROFILES[name]
    except KeyError as exc:
        raise ValueError("Cách tạo giọng phải là natural, clear hoặc stable.") from exc
    return {"temperature": profile.temperature, "top_k": profile.top_k, "top_p": profile.top_p,
            "repetition_penalty": profile.repetition_penalty}


def postprocess_filter(name: str) -> str | None:
    try:
        return POSTPROCESS_FILTERS[name]
    except KeyError as exc:
        raise ValueError("Mức lọc đầu ra phải là off, light hoặc strong.") from exc


def validate_utterance(duration: float, size: int) -> None:
    if duration < 0.25:
        raise ValueError("Bản ghi quá ngắn; hãy nói ít nhất khoảng nửa giây.")
    if size > MAX_SPEECH_TO_SPEECH_BYTES:
        raise ValueError("Bản ghi vượt quá giới hạn tải lên 500 MB.")


def split_synthesis_text(text: str, max_chars: int = 240) -> list[str]:
    """Split long transcripts into TTS-safe chunks without dropping words."""
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    sentences = re.split(r"(?<=[.!?…])\s+", normalized)
    pieces: list[str] = []
    for sentence in sentences:
        if len(sentence) <= max_chars:
            pieces.append(sentence)
            continue
        words = sentence.split()
        current: list[str] = []
        for word in words:
            if current and len(" ".join((*current, word))) > max_chars:
                pieces.append(" ".join(current))
                current = []
            current.append(word)
        if current:
            pieces.append(" ".join(current))
    chunks: list[str] = []
    for piece in pieces:
        if chunks and len(chunks[-1]) + 1 + len(piece) <= max_chars:
            chunks[-1] += " " + piece
        else:
            chunks.append(piece)
    return chunks
