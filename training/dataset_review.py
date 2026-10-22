"""Human review of ASR labels before an uploaded voice is fine-tuned."""
from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from training.pipeline import sha256_file


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _base_rows(dataset: Path) -> list[dict]:
    source = dataset / "review_candidates.jsonl"
    if not source.is_file():
        source.write_text((dataset / "train.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    return [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]


def load(dataset: Path, offset: int = 0, limit: int = 25) -> dict:
    rows = _base_rows(dataset)
    review_path = dataset / "review.json"
    state = json.loads(review_path.read_text(encoding="utf-8")) if review_path.is_file() else {"approved": False, "changes": {}}
    window = []
    for row in rows[offset:offset + limit]:
        change = state["changes"].get(row["file_name"], {})
        window.append({**row, "text": change.get("text", row["text"]), "include": change.get("include", True)})
    return {"approved": state["approved"], "total": len(rows), "offset": offset, "limit": limit,
            "changed": len(state["changes"]), "clips": window}


def edit(dataset: Path, filename: str, text: str, include: bool) -> dict:
    if not re.fullmatch(r"\d{6}\.wav", filename):
        raise ValueError("Tên clip không hợp lệ.")
    if not isinstance(text, str) or len(text) > 1000 or (include and not text.strip()):
        raise ValueError("Transcript phải có chữ và dưới 1000 ký tự.")
    if "\n" in text or "\r" in text:
        raise ValueError("Transcript phải ở một dòng.")
    rows = _base_rows(dataset)
    if filename not in {row["file_name"] for row in rows}:
        raise ValueError("Không tìm thấy clip.")
    path = dataset / "review.json"
    state = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"approved": False, "changes": {}}
    if state["approved"]:
        raise ValueError("Bộ dữ liệu đã được duyệt; hãy tạo lượt train mới trước khi sửa.")
    state["changes"][filename] = {"text": text.strip().replace("|", ","), "include": bool(include)}
    _atomic_json(path, state)
    return state


def approve(dataset: Path) -> dict:
    rows = _base_rows(dataset)
    path = dataset / "review.json"
    state = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"approved": False, "changes": {}}
    if state["approved"]:
        return state
    selected = []
    for row in rows:
        change = state["changes"].get(row["file_name"], {})
        if change.get("include", True):
            selected.append({**row, "text": change.get("text", row["text"])})
    if not selected:
        raise ValueError("Cần giữ ít nhất một clip để huấn luyện.")
    temporary = dataset / "train.jsonl.tmp"
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")
    temporary.replace(dataset / "train.jsonl")
    temporary = dataset / "train_metadata.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="|")
        writer.writerow(["file_name", "text"])
        writer.writerows((row["file_name"], row["text"]) for row in selected)
    temporary.replace(dataset / "train_metadata.csv")
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["counts"] = {"train": len(selected), "validation": 0, "test": 0}
    manifest["minutes"] = {"train": round(sum(row["duration"] for row in selected) / 60, 2), "validation": 0, "test": 0}
    manifest["usable_seconds"] = round(sum(row["duration"] for row in selected), 1)
    manifest["review"] = {"approved_at": datetime.now(timezone.utc).isoformat(),
                          "candidate_clips": len(rows), "accepted_clips": len(selected),
                          "edited_clips": len(state["changes"])}
    manifest["review_sha256"] = sha256_file(dataset / "train.jsonl")
    _atomic_json(manifest_path, manifest)
    state["approved"] = True
    state["approved_at"] = manifest["review"]["approved_at"]
    _atomic_json(path, state)
    return state
