"""Persistent guided recording sessions and one-speaker LoRA dataset export."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import unicodedata
import uuid
import wave
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from training.pipeline import sha256_file
from training.reading_prompts import PROMPTS

ROOT = Path(__file__).resolve().parents[1]
SESSIONS = ROOT / "data/self_training"
PREPARED = ROOT / "data/prepared"
MIN_CLIPS = 50
MIN_SECONDS = 300
SESSION_PATTERN = re.compile(r"^[a-f0-9]{32}$")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def session_dir(session_id: str) -> Path:
    if not SESSION_PATTERN.fullmatch(session_id):
        raise ValueError("Mã phiên thu âm không hợp lệ")
    return SESSIONS / session_id


def write_state(state: dict) -> None:
    directory = session_dir(state["id"])
    temporary = directory / "session.tmp"
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(directory / "session.json")


def load(session_id: str) -> dict:
    path = session_dir(session_id) / "session.json"
    if not path.is_file():
        raise FileNotFoundError("Không tìm thấy phiên thu âm")
    return json.loads(path.read_text(encoding="utf-8"))


def create(name: str, locale: str) -> dict:
    name = name.strip()
    if not 1 <= len(name) <= 60 or locale not in PROMPTS:
        raise ValueError("Tên giọng hoặc ngôn ngữ không hợp lệ")
    session_id = uuid.uuid4().hex
    directory = session_dir(session_id)
    (directory / "clips").mkdir(parents=True)
    state = {"id": session_id, "name": name, "locale": locale, "status": "recording",
             "created_at": now(), "prompts": PROMPTS[locale], "clips": {},
             "min_clips": MIN_CLIPS, "min_seconds": MIN_SECONDS}
    write_state(state)
    return public_state(state)


def public_state(state: dict) -> dict:
    accepted = [item for item in state["clips"].values() if item.get("accepted")]
    return {**state, "accepted_count": len(accepted),
            "accepted_seconds": round(sum(item["duration"] for item in accepted), 1),
            "ready_to_train": len(accepted) >= MIN_CLIPS and sum(item["duration"] for item in accepted) >= MIN_SECONDS}


def match_score(prompt: str, recognized: str) -> float:
    def simple(text: str) -> str:
        text = unicodedata.normalize("NFD", text.casefold())
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
        return " ".join("".join(ch if ch.isalnum() else " " for ch in text).split())
    return round(SequenceMatcher(None, simple(prompt), simple(recognized)).ratio(), 3)


def inspect_wav(path: Path) -> dict:
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getframerate() != 24000:
            raise ValueError("Bản ghi cần WAV mono 24 kHz, PCM 16-bit")
        duration = wav.getnframes() / wav.getframerate()
        if not 2.0 <= duration <= 20.0:
            raise ValueError("Mỗi câu cần 2–20 giây; hãy đọc lại ở tốc độ tự nhiên")
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768
    rms = float(np.sqrt(np.mean(samples * samples)))
    clipping = float(np.mean(np.abs(samples) >= .98))
    if rms < .006:
        raise ValueError("Bản ghi quá nhỏ hoặc im lặng; hãy kiểm tra micro")
    if clipping > .02:
        raise ValueError("Âm thanh bị vỡ tiếng; hãy giảm âm lượng micro và đọc lại")
    return {"duration": round(duration, 2), "rms": round(rms, 4), "clipping_ratio": round(clipping, 4)}


def add_clip(session_id: str, index: int, normalized_wav: Path, recognized: str) -> dict:
    state = load(session_id)
    if state["status"] != "recording" or not 0 <= index < len(state["prompts"]):
        raise ValueError("Phiên thu âm hoặc số câu không hợp lệ")
    quality = inspect_wav(normalized_wav)
    score = match_score(state["prompts"][index], recognized)
    accepted = bool(recognized.strip()) and score >= .65
    target = session_dir(session_id) / "clips" / f"{index:03}.wav"
    shutil.copyfile(normalized_wav, target)
    item = {"index": index, "text": state["prompts"][index], "file_name": target.name,
            "recognized": recognized.strip(), "match_score": score, "accepted": accepted,
            "needs_review": not accepted, "recorded_at": now(), "sha256": sha256_file(target), **quality}
    state["clips"][str(index)] = item
    write_state(state)
    return {"clip": item, "session": public_state(state)}


def accept_clip(session_id: str, index: int) -> dict:
    state = load(session_id)
    if state["status"] != "recording" or str(index) not in state["clips"]:
        raise ValueError("Câu chưa được thu hoặc phiên đã đóng")
    state["clips"][str(index)]["accepted"] = True
    state["clips"][str(index)]["needs_review"] = False
    state["clips"][str(index)]["reviewed_by_user"] = True
    write_state(state)
    return public_state(state)


def prepare(session_id: str) -> dict:
    state = load(session_id)
    if state["status"] != "recording":
        raise ValueError("Phiên thu âm đã được gửi huấn luyện")
    accepted = sorted((item for item in state["clips"].values() if item["accepted"]), key=lambda item: item["index"])
    seconds = sum(item["duration"] for item in accepted)
    if len(accepted) < MIN_CLIPS or seconds < MIN_SECONDS:
        raise ValueError(f"Cần ít nhất {MIN_CLIPS} câu và {MIN_SECONDS // 60} phút giọng đọc đã kiểm tra")
    digests = [item["sha256"] for item in accepted]
    if len(set(digests)) != len(digests):
        raise ValueError("Có bản ghi âm trùng nhau; hãy đọc lại các câu bị trùng")
    target = PREPARED / f"self_{session_id}"
    if target.exists():
        raise FileExistsError("Bộ dữ liệu của phiên này đã tồn tại")
    (target / "raw_audio").mkdir(parents=True)
    try:
        rows = []
        speaker = f"SELF{session_id[:8].upper()}"
        for item in accepted:
            source = session_dir(session_id) / "clips" / item["file_name"]
            if sha256_file(source) != item["sha256"]:
                raise ValueError("Bản ghi đã thay đổi sau khi kiểm tra")
            shutil.copyfile(source, target / "raw_audio" / item["file_name"])
            rows.append({"file_name": item["file_name"], "text": item["text"].replace("|", ","), "speaker_id": speaker,
                         "language": "vi" if state["locale"] == "vi-VN" else "en", "source": "self",
                         "source_sha256": item["sha256"], "duration": item["duration"], "split": "train"})
        holdout = max(5, round(len(rows) * .1))
        shuffled = sorted(rows, key=lambda row: hashlib.sha256(f"{session_id}:{row['file_name']}:split-v1".encode()).hexdigest())
        for row in shuffled[:holdout]:
            row["split"] = "test"
        for row in shuffled[holdout:holdout * 2]:
            row["split"] = "validation"
        counts = Counter(row["split"] for row in rows)
        if any(counts[split] < 5 for split in ("validation", "test")):
            raise ValueError("Chưa có đủ câu validation/test; hãy ghi thêm vài câu")
        for split in ("train", "validation", "test"):
            with (target / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
                for row in rows:
                    if row["split"] == split:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        with (target / "train_metadata.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="|")
            writer.writerow(["file_name", "text"])
            writer.writerows((row["file_name"], row["text"]) for row in rows if row["split"] == "train")
        manifest = {"source": {"name": "User guided recording", "language": state["locale"],
                               "license": "User-owned or authorized voice; consent confirmed in browser"},
                    "source_id": "self", "session_id": session_id, "display_name": state["name"],
                    "speaker_id": speaker, "locale": state["locale"], "counts": dict(counts),
                    "minutes": {split: round(sum(row["duration"] for row in rows if row["split"] == split) / 60, 2)
                                for split in ("train", "validation", "test")},
                    "preparation": "24 kHz mono PCM; one user-confirmed speaker; transcript prompts; disjoint deterministic splits"}
        (target / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        state.update({"status": "queued", "dataset": str(target), "prepared_at": now()})
        write_state(state)
        return public_state(state)
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise
