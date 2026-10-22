"""Disk-backed state for one-process speech conversion jobs."""
from __future__ import annotations

import json
import re
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path


class SpeechJobs:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def directory(self, job_id: str) -> Path:
        if not re.fullmatch(r"[a-f0-9]{32}", job_id):
            raise ValueError("Invalid job ID")
        return self.root / job_id

    def get(self, job_id: str) -> dict | None:
        path = self.directory(job_id) / "status.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    def put(self, job_id: str, **fields) -> dict:
        with self.lock:
            path = self.directory(job_id) / "status.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            state = self.get(job_id) or {"id": job_id, "created_at": datetime.now(timezone.utc).isoformat()}
            state.update(fields, updated_at=datetime.now(timezone.utc).isoformat())
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(path)
            return state

    def unfinished(self) -> list[dict]:
        states = []
        for path in self.root.glob("*/status.json"):
            try:
                state = self.get(path.parent.name)
                if state and state.get("status") in {"queued", "running"}:
                    states.append(state)
            except (ValueError, OSError, json.JSONDecodeError):
                continue
        return states

    def delete(self, job_id: str) -> None:
        shutil.rmtree(self.directory(job_id))
