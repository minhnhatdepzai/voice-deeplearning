"""Manage the single warm VieNeu process used for interactive conversion."""
from __future__ import annotations

import atexit
import json
import os
import select
import subprocess
import threading
import time
import uuid
from pathlib import Path


class WarmVoiceWorker:
    def __init__(self, root: Path, data: Path):
        self.root = root
        self.data = data
        self._lock = threading.RLock()
        self._process: subprocess.Popen | None = None
        self._config: tuple | None = None
        self._log = None
        self._load_ms: int | None = None
        self._last_used: float | None = None
        atexit.register(self.stop)

    def _read(self, timeout: float) -> dict:
        process = self._process
        if process is None or process.stdout is None:
            raise RuntimeError("VieNeu worker is not running")
        ready, _, _ = select.select([process.stdout], [], [], timeout)
        if not ready:
            raise TimeoutError("VieNeu worker did not respond in time")
        line = process.stdout.readline()
        if not line:
            code = process.poll()
            raise RuntimeError(f"VieNeu worker stopped unexpectedly (exit {code})")
        return json.loads(line)

    def prepare(self, mode: str, model: Path | None = None, voice: str | None = None,
                timeout: float = 180) -> dict:
        config = (mode, str(model.resolve()) if model else None, voice)
        with self._lock:
            if self._process is not None and self._process.poll() is None and self._config == config:
                return {"warm": True, "load_ms": self._load_ms or 0}
            self.stop()
            python = self.data / "upstream/vieneu/.venv/bin/python"
            if not python.is_file():
                raise FileNotFoundError("VieNeu CUDA environment is missing; run training/setup.sh")
            command = [str(python), "-u", str(self.root / "training/realtime_tts_worker.py"), "--mode", mode]
            if model:
                command.extend(["--model", str(model)])
            if voice:
                command.extend(["--voice", voice])
            log_path = self.data / "speech_to_speech_worker.log"
            self._log = log_path.open("a", encoding="utf-8")
            self._process = subprocess.Popen(
                command, cwd=self.root, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self._log, text=True, bufsize=1, env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            self._config = config
            response = self._read(timeout)
            if response.get("event") != "ready":
                self.stop()
                raise RuntimeError(f"Unexpected VieNeu worker response: {response}")
            self._load_ms = int(response.get("load_ms", 0))
            self._last_used = time.time()
            return {"warm": False, "load_ms": self._load_ms}

    def synthesize(self, text: str, output: Path, *, mode: str,
                   model: Path | None = None, voice: str | None = None,
                   reference: Path | None = None, sampling: dict | None = None,
                   timeout: float = 120) -> dict:
        with self._lock:
            warmup = self.prepare(mode, model, voice)
            process = self._process
            if process is None or process.stdin is None:
                raise RuntimeError("VieNeu worker is unavailable")
            request_id = uuid.uuid4().hex
            request = {"id": request_id, "text": text, "output": str(output)}
            if sampling:
                request["sampling"] = sampling
            if reference:
                request["reference"] = str(reference)
            process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            process.stdin.flush()
            response = self._read(timeout)
            if response.get("id") != request_id:
                raise RuntimeError("VieNeu worker returned a mismatched response")
            if not response.get("ok"):
                raise RuntimeError(response.get("error") or "VieNeu inference failed")
            self._last_used = time.time()
            return {**response, **warmup}

    def stop(self) -> None:
        with self._lock:
            process, self._process = self._process, None
            self._config = None
            self._load_ms = None
            self._last_used = None
            if process is not None:
                try:
                    if process.stdin:
                        process.stdin.close()
                    process.terminate()
                    process.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    process.kill()
                    process.wait(timeout=3)
                if process.stdout:
                    process.stdout.close()
            if self._log:
                self._log.close()
                self._log = None

    def status(self) -> dict:
        with self._lock:
            running = self._process is not None and self._process.poll() is None
            return {"running": running, "mode": self._config[0] if running and self._config else None,
                    "load_ms": self._load_ms if running else None, "last_used": self._last_used if running else None}
