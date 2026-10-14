"""Local voice studio API. Models are loaded on first inference, not at startup."""

from __future__ import annotations

import asyncio
import fcntl
import gc
import json
import os
import re
import shutil
import subprocess
import threading
import tempfile
import time
import uuid
import tarfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from training.pipeline import SOURCES, build as build_training_dataset, read_archive
from training.run import run as run_training
from training.evaluate import evaluate as evaluate_training
from training.benchmark_fpt import benchmark as benchmark_fpt
from training.speaker_similarity import measure as measure_speaker_similarity
from server.prosody import combine as combine_delivery, plan as plan_delivery
from server.media_edit import decode as decode_audio, mix as mix_audio
from server.timeline import render as render_timeline
from server.text_normalization import normalize_for_speech
from training import self_recording
from training import uploaded_audio

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("VOICE_STUDIO_DATA", ROOT / "data"))
VOICES = DATA / "voices"
VOICE_SOURCES = VOICES / "sources"
OUTPUTS = DATA / "outputs"
UPLOADS = DATA / "uploads"
for directory in (VOICES, VOICE_SOURCES, OUTPUTS, UPLOADS):
    directory.mkdir(parents=True, exist_ok=True)
INDEX = DATA / "voices.json"
ALLOWED = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".webm", ".mp4", ".mov", ".mkv"}
VIDEO = {".mp4", ".mov", ".mkv", ".webm"}
MAX_BYTES = 500 * 1024 * 1024
_voice_lock = threading.Lock()
_tts_lock = threading.Lock()
_stt_lock = threading.Lock()
_stt_device = None
_training_pool = ThreadPoolExecutor(max_workers=1)
_training_jobs: dict[str, dict] = {}
_training_lock = threading.Lock()


@contextmanager
def _gpu_slot(wait_seconds: float | None = 30, min_free_mib: int = 0):
    """Share the same cross-process GPU lock as the LoRA workers."""
    path = DATA / "auto_train/worker.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = None if wait_seconds is None else time.monotonic() + wait_seconds
    with path.open("a+") as handle:
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if deadline is not None and time.monotonic() >= deadline:
                    raise HTTPException(503, "GPU đang huấn luyện hoặc xử lý tác vụ khác. Đợi tác vụ hoàn tất rồi thử tạo giọng lại.",
                                        headers={"Retry-After": "15"})
                time.sleep(.2)
        try:
            if min_free_mib:
                query = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                       capture_output=True, text=True, timeout=5)
                if query.returncode == 0:
                    try:
                        free = int(query.stdout.splitlines()[0].strip())
                    except (IndexError, ValueError):
                        free = 0
                    if free < min_free_mib:
                        raise HTTPException(503, f"GPU chỉ còn {free} MiB VRAM trống; cần ít nhất {min_free_mib} MiB để tạo giọng. Hãy chờ tác vụ GPU khác hoàn tất.",
                                            headers={"Retry-After": "15"})
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
ARCHIVES = {
    "vivos": DATA / "datasets/vivos/vivos-vietnamese-speech-corpus-for-asr.zip",
    "libritts": DATA / "datasets/libritts/dev-clean.tar.gz",
    "emov": DATA / "datasets/emov/jenie_Amused.tar.gz",
    "fpt": DATA / "datasets/fpt/fpt-open-speech-dataset-fosd-vietnamese.zip",
    "ljspeech": DATA / "datasets/ljspeech/LJSpeech-1.1.tar.bz2",
}
FPT_SOURCE = {"name": "FPT Open Speech Dataset", "language": "vi", "original": "https://data.mendeley.com/datasets/k9sxg2twv4/4",
              "mirror": "https://www.kaggle.com/datasets/thinh127/fpt-open-speech-dataset-fosd-vietnamese",
              "license": "FPT Public License / CC BY 4.0 (original); Kaggle metadata unknown", "trainable": False,
              "role": "STT benchmark; speaker IDs absent in mirror"}
PREPARED = DATA / "prepared"
RUNS = DATA / "training_runs"

app = FastAPI(title="Vocalab Local API", version="1.0")


def _read_voices() -> list[dict]:
    if not INDEX.exists():
        return []
    return json.loads(INDEX.read_text(encoding="utf-8"))


def _write_voices(voices: list[dict]) -> None:
    temporary = INDEX.with_suffix(".tmp")
    temporary.write_text(json.dumps(voices, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(INDEX)


def _run(*args: str) -> None:
    try:
        subprocess.run(args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
    except FileNotFoundError as exc:
        raise HTTPException(500, "Cần cài ffmpeg và ffprobe trên máy.") from exc
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(408, "Xử lý media quá thời gian cho phép.") from exc
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.decode(errors="replace")[-500:]
        if "out of memory" in message.lower() or "tried to allocate" in message.lower():
            raise HTTPException(503, "GPU hết VRAM khi xử lý. Hãy đợi tác vụ train hoặc tác vụ GPU khác hoàn tất rồi thử lại.",
                                headers={"Retry-After": "15"}) from exc
        raise HTTPException(422, f"Không xử lý được file media: {message}") from exc


async def _save_upload(upload: UploadFile, max_bytes: int | None = MAX_BYTES) -> Path:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(415, "Định dạng file chưa hỗ trợ.")
    path = UPLOADS / f"{uuid.uuid4().hex}{suffix}"
    size = 0
    try:
        with path.open("wb") as out:
            while chunk := await upload.read(1024 * 1024):
                size += len(chunk)
                if max_bytes is not None and size > max_bytes:
                    raise HTTPException(413, f"File vượt quá giới hạn {max_bytes // 1048576} MB.")
                if shutil.disk_usage(UPLOADS).free < max(2 * 1024**3, len(chunk) * 2):
                    raise HTTPException(507, "Ổ đĩa không đủ chỗ trống để giữ file và chuẩn bị dữ liệu train.")
                out.write(chunk)
        if not size:
            raise HTTPException(422, "File rỗng.")
        return path
    except Exception:
        path.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()


def _duration(path: Path) -> float:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            check=True, capture_output=True, text=True, timeout=30,
        )
        return float(result.stdout.strip())
    except (ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise HTTPException(422, "Không đọc được thời lượng audio/video.") from exc


def _source_media_type(path: Path) -> str:
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=format_name",
                             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                            capture_output=True, text=True, timeout=30)
    detected = result.stdout.strip().split(",") if result.returncode == 0 else []
    if "mp3" in detected:
        return "audio/mpeg"
    if "wav" in detected:
        return "audio/wav"
    if "flac" in detected:
        return "audio/flac"
    if path.suffix == ".webm":
        return "video/webm"
    if path.suffix == ".mp4":
        return "video/mp4"
    return "application/octet-stream"


def _get_voice(voice_id: str) -> dict:
    with _voice_lock:
        voice = next((item for item in _read_voices() if item["id"] == voice_id), None)
    if voice is None:
        raise HTTPException(404, "Không tìm thấy giọng nói.")
    return voice


def _srt_time(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    secs, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"


@app.get("/api/health")
def health():
    return {"ok": True, "tts_backend": "VieNeu CUDA subprocess", "stt_device_last_used": _stt_device,
            "ffmpeg": shutil.which("ffmpeg") is not None}


class PrepareRequest(BaseModel):
    source: str
    speaker: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_-]+$")
    max_clips: int = Field(default=300, ge=30, le=10000)


class TrainRequest(BaseModel):
    dataset: str = Field(pattern=r"^[a-z0-9]+_[A-Za-z0-9_-]+$")
    steps: int = Field(default=30, ge=1, le=10000)


class EvaluateRequest(BaseModel):
    run: str = Field(pattern=r"^[A-Za-z0-9_-]{1,128}$")
    count: int = Field(default=5, ge=1, le=30)


class RegisterRequest(BaseModel):
    run: str = Field(pattern=r"^[A-Za-z0-9_-]{1,128}$")


class SelfSessionRequest(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    locale: str = Field(pattern=r"^(vi-VN|en-US)$")
    consent: bool


def _session_or_404(session_id: str) -> dict:
    try:
        return self_recording.load(session_id)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/api/self-training/sessions")
def self_sessions():
    result = []
    for path in self_recording.SESSIONS.glob("*/session.json"):
        try:
            result.append(self_recording.public_state(json.loads(path.read_text(encoding="utf-8"))))
        except (KeyError, json.JSONDecodeError):
            continue
    return sorted(result, key=lambda row: row["created_at"], reverse=True)


@app.post("/api/self-training/sessions")
def create_self_session(request: SelfSessionRequest):
    if not request.consent:
        raise HTTPException(400, "Cần xác nhận quyền sử dụng giọng nói.")
    try:
        return self_recording.create(request.name, request.locale)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/api/self-training/sessions/{session_id}")
def get_self_session(session_id: str):
    return self_recording.public_state(_session_or_404(session_id))


@app.post("/api/self-training/sessions/{session_id}/clips/{index}")
async def upload_self_clip(session_id: str, index: int, file: UploadFile = File(...)):
    state = _session_or_404(session_id)
    if state["status"] != "recording" or not 0 <= index < len(state["prompts"]):
        raise HTTPException(422, "Phiên thu âm hoặc số câu không hợp lệ.")
    source = await _save_upload(file)
    try:
        if source.stat().st_size > 25 * 1024 * 1024:
            raise HTTPException(413, "Mỗi bản ghi tối đa 25 MB.")
        with tempfile.TemporaryDirectory(prefix="vocalab-self-", dir=UPLOADS) as temp:
            audio = Path(temp) / "clip.wav"
            await asyncio.to_thread(decode_audio, source, audio)
            quality = await asyncio.to_thread(self_recording.inspect_wav, audio)
            segments, _ = await asyncio.to_thread(_transcribe, audio, "vi" if state["locale"] == "vi-VN" else "en")
            recognized = " ".join(item["text"] for item in segments)
            result = await asyncio.to_thread(self_recording.add_clip, session_id, index, audio, recognized)
            result["clip"]["quality"] = quality
            result["session"] = _auto_start_completed_session(result["session"])
            return result
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        source.unlink(missing_ok=True)


@app.post("/api/self-training/sessions/{session_id}/clips/{index}/accept")
def accept_self_clip(session_id: str, index: int):
    try:
        return _auto_start_completed_session(self_recording.accept_clip(session_id, index))
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/api/self-training/sessions/{session_id}/clips/{index}/audio")
def self_clip_audio(session_id: str, index: int):
    state = _session_or_404(session_id)
    if str(index) not in state["clips"]:
        raise HTTPException(404, "Chưa thu câu này.")
    path = self_recording.session_dir(session_id) / "clips" / f"{index:03}.wav"
    if not path.is_file():
        raise HTTPException(404, "Bản ghi đã mất.")
    return FileResponse(path, media_type="audio/wav")


def _auto_start_completed_session(state: dict) -> dict:
    """Queue training after the final approved reading, including manual reviews."""
    if state["status"] == "recording" and state["ready_to_train"] and state["accepted_count"] == len(state["prompts"]):
        try:
            return start_self_training(state["id"])["session"]
        except Exception as exc:
            # Keep the approved recordings available; the user can retry the train button.
            state = self_recording.public_state(self_recording.load(state["id"]))
            state["auto_train_error"] = str(exc)
    return state


@app.post("/api/self-training/sessions/{session_id}/train")
def start_self_training(session_id: str):
    state = _session_or_404(session_id)
    if state["status"] == "recording":
        try:
            state = self_recording.prepare(session_id)
        except (ValueError, FileExistsError) as exc:
            raise HTTPException(422, str(exc)) from exc
    elif state["status"] != "queued":
        raise HTTPException(409, "Phiên này đã được đưa vào hàng đợi hoặc đã huấn luyện.")
    unit = f"vocalab-self-{session_id[:16]}"
    command = ["systemd-run", "--user", "--unit", unit, "--description", f"Vocalab guided voice {session_id[:8]}",
               f"--working-directory={ROOT}", str(ROOT / ".venv/bin/python"), "-m", "training.self_train", "--session", session_id]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=15)
    if completed.returncode:
        state = _session_or_404(session_id)
        state["dispatch_error"] = f"Không khởi động được worker: {completed.stderr.strip()[-300:]}"
        self_recording.write_state(state)
        raise HTTPException(500, state["dispatch_error"])
    state = _session_or_404(session_id)
    if state.pop("dispatch_error", None):
        self_recording.write_state(state)
    return {"session": self_recording.public_state(_session_or_404(session_id)), "unit": unit, "worker": completed.stdout.strip()}


def _archive_ready(source: str) -> bool:
    path = ARCHIVES[source]
    if not path.is_file():
        return False
    if source in {"vivos", "fpt"}:
        return zipfile.is_zipfile(path)
    if source == "libritts":
        return path.stat().st_size == 1_291_469_655 and tarfile.is_tarfile(path)
    if source == "emov":
        return tarfile.is_tarfile(path) and (path.parent / "cmuarctic.data").is_file()
    if source == "ljspeech":
        return tarfile.is_tarfile(path)
    return False


@app.get("/api/training/sources")
def training_sources():
    rows = [{"id": key, **value, "trainable": True, "downloaded": _archive_ready(key)} for key, value in SOURCES.items()]
    rows.append({"id": "fpt", **FPT_SOURCE, "downloaded": _archive_ready("fpt")})
    return rows


@app.get("/api/training/benchmarks")
def training_benchmarks():
    path = DATA / "benchmarks/fpt_stt.json"
    if not path.is_file():
        return []
    result = json.loads(path.read_text(encoding="utf-8"))
    return [{"source": "fpt", "items": result["metrics"]["all"]["items"],
             "wer": result["metrics"]["all"]["wer"], "cer": result["metrics"]["all"]["cer"],
             "subsets": result["metrics"]}]


@app.get("/api/training/speakers/{source}")
def training_speakers(source: str):
    if source not in SOURCES or not _archive_ready(source):
        raise HTTPException(404, "Bộ dữ liệu chưa tải xong hoặc chưa có trên máy.")
    from collections import Counter
    archive, rows = read_archive(source, ARCHIVES[source])
    archive.close()
    return [{"id": speaker, "clips": count} for speaker, count in Counter(r["speaker"] for r in rows).most_common()]


@app.get("/api/training/datasets")
def training_datasets():
    rows = []
    for manifest in PREPARED.glob("*/manifest.json"):
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            rows.append({"id": manifest.parent.name, "source": data["source_id"], "speaker": data["speaker_id"],
                         "counts": data["counts"], "minutes": data["minutes"], "license": data["source"]["license"]})
        except (KeyError, json.JSONDecodeError):
            continue
    return sorted(rows, key=lambda r: r["id"])


def _submit_job(kind: str, work):
    job_id = uuid.uuid4().hex
    with _training_lock:
        _training_jobs[job_id] = {"id": job_id, "kind": kind, "status": "queued", "created_at": datetime.now(timezone.utc).isoformat()}

    def execute():
        with _training_lock:
            _training_jobs[job_id]["status"] = "running"
        try:
            result = work()
            with _training_lock:
                _training_jobs[job_id].update({"status": "complete", "result": result})
        except Exception as exc:
            with _training_lock:
                _training_jobs[job_id].update({"status": "failed", "error": str(exc)})
    _training_pool.submit(execute)
    return _training_jobs[job_id]


@app.post("/api/training/prepare")
def prepare_training_dataset(request: PrepareRequest):
    if request.source not in SOURCES or not _archive_ready(request.source):
        raise HTTPException(422, "Bộ dữ liệu chưa sẵn sàng.")
    dataset_id = f"{request.source}_{request.speaker}"
    target = PREPARED / dataset_id
    if target.exists():
        raise HTTPException(409, "Bộ dữ liệu đã được chuẩn bị.")
    return _submit_job("prepare", lambda: {"dataset": dataset_id, "manifest": build_training_dataset(request.source, ARCHIVES[request.source], request.speaker, target, request.max_clips)})


@app.post("/api/training/train")
def start_training(request: TrainRequest):
    dataset = PREPARED / request.dataset
    if not (dataset / "manifest.json").is_file():
        raise HTTPException(404, "Bộ dữ liệu chưa được chuẩn bị.")
    upstream = Path(os.environ.get("VIENEU_UPSTREAM", DATA / "upstream/vieneu"))
    if not (upstream / ".venv/bin/python").is_file():
        raise HTTPException(422, "Chưa cài môi trường training. Chạy training/setup.sh.")
    run_name = f"{request.dataset}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:4]}"
    def work():
        with _gpu_slot(wait_seconds=None):
            return run_training(dataset, upstream, request.steps, run_name)
    return _submit_job("train", work)


@app.get("/api/training/jobs")
def training_jobs():
    with _training_lock:
        return list(reversed(list(_training_jobs.values())))


@app.post("/api/training/evaluate")
def start_training_evaluation(request: EvaluateRequest):
    run_dir = RUNS / request.run
    if not (run_dir / "status.json").is_file():
        raise HTTPException(404, "Không tìm thấy phiên fine-tune.")
    upstream = Path(os.environ.get("VIENEU_UPSTREAM", DATA / "upstream/vieneu"))
    def work():
        with _gpu_slot(wait_seconds=None):
            metrics = evaluate_training(run_dir, upstream / ".venv/bin/python", request.count)["metrics"]
            speaker = measure_speaker_similarity(run_dir, request.count)["metrics"]
        return {"run": request.run, "metrics": metrics, "speaker_similarity": speaker}
    return _submit_job("evaluate", work)


@app.post("/api/training/benchmark/fpt")
def start_fpt_benchmark():
    if not _archive_ready("fpt"):
        raise HTTPException(422, "Chưa tải bộ dữ liệu FPT từ Kaggle.")
    def work():
        with _gpu_slot(wait_seconds=None):
            return {"source": "fpt", "metrics": benchmark_fpt(ARCHIVES["fpt"], DATA / "benchmarks/fpt_stt.json")["metrics"]}
    return _submit_job("benchmark", work)


@app.post("/api/training/register")
def register_trained_voice(request: RegisterRequest):
    run_dir = RUNS / request.run
    status_path = run_dir / "status.json"
    if not status_path.is_file():
        raise HTTPException(404, "Không tìm thấy phiên fine-tune.")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") not in {"evaluated", "full_trained"}:
        raise HTTPException(422, "Model chưa hoàn tất huấn luyện.")
    with _voice_lock:
        existing = next((voice for voice in _read_voices() if voice.get("run") == request.run), None)
    if existing:
        return existing
    merged = run_dir / "official" / request.run / "merged"
    if Path(status["merged"]).resolve() != merged.resolve() or not (merged / "voices_v3_turbo.json").is_file():
        raise HTTPException(422, "Mô hình đã merge hoặc giọng preset không hợp lệ.")
    if status.get("release_import"):
        metadata_path = run_dir / "release_metadata.json"
        preview_source = run_dir / "preview.wav"
        if not metadata_path.is_file() or not preview_source.is_file():
            raise HTTPException(422, "Thiếu thông tin hoặc mẫu nghe của model phát hành.")
        release = json.loads(metadata_path.read_text(encoding="utf-8"))
        if release.get("run") != request.run or release.get("speaker_id") != status["speaker_id"]:
            raise HTTPException(422, "Thông tin model phát hành không khớp.")
        manifest = {"source": {"license": release["license"]},
                    "display_name": f"{status['speaker_id']} · Fine-tune công khai",
                    "locale": release["locale"]}
        reference = {"duration": _duration(preview_source)}
    else:
        dataset = PREPARED / Path(status["dataset"]).name
        rows = [json.loads(line) for line in (dataset / "train.jsonl").read_text(encoding="utf-8").splitlines()]
        reference = next((r for r in rows if 3 <= r["duration"] <= 8), rows[0])
        preview_source = dataset / "raw_audio" / reference["file_name"]
        manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    voice_id = uuid.uuid4().hex
    preview = VOICES / f"{voice_id}.wav"
    shutil.copyfile(preview_source, preview)
    label = manifest.get("display_name") or ("Jenie · Vui tươi (EN)" if status["source"] == "emov" else f"{status['speaker_id']} · Fine-tune")
    locale = manifest.get("locale") or ("vi-VN" if status["source"] == "vivos" else "en-US" if status["source"] == "ljspeech" else "en")
    item = {"id": voice_id, "name": label, "source": status["source"],
            "kind": "finetuned", "run": request.run, "license": manifest["source"]["license"], "locale": locale,
            "created_at": datetime.now(timezone.utc).isoformat(), "duration": reference["duration"],
            "preview_url": f"/api/media/voices/{voice_id}.wav"}
    if manifest.get("voice_id"):
        original = next((v for v in _read_voices() if v["id"] == manifest["voice_id"]), None)
        if original and original.get("source_url"):
            item.update({"origin_voice_id": original["id"], "source_url": original["source_url"],
                         "duration": original["duration"]})
    with _voice_lock:
        voices = _read_voices()
        voices.append(item)
        _write_voices(voices)
    return item


@app.get("/api/training/runs")
def training_runs():
    result = []
    with _voice_lock:
        registered = {voice.get("run") for voice in _read_voices() if voice.get("run")}
    for path in RUNS.glob("*/status.json"):
        try:
            status = json.loads(path.read_text(encoding="utf-8"))
            result.append({"run": status["run"], "dataset": Path(status["dataset"]).name,
                           "status": status["status"], "updates": status.get("optimizer_updates", 0),
                           "adapter_sha256": status.get("adapter_sha256"), "generated_wer": status.get("generated_wer"),
                           "reference_wer": status.get("reference_wer"),
                           "speaker_similarity": status.get("speaker_similarity"),
                           "test_items": status.get("test_items"), "registered": status["run"] in registered,
                           "error": status.get("error")})
        except (KeyError, json.JSONDecodeError):
            continue
    return sorted(result, key=lambda r: r["run"], reverse=True)


@app.get("/api/training/queue")
def training_queue():
    result = {}
    for name in ("auto_train", "extended_train"):
        path = DATA / name / "status.json"
        if path.is_file():
            try:
                result[name] = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
    return result


@app.get("/api/voices")
def list_voices():
    with _voice_lock:
        voices = list(reversed(_read_voices()))
    for voice in voices:
        if "locale" not in voice:
            voice["locale"] = "vi-VN" if voice.get("source") == "vivos" else "en" if voice.get("source") in {"libritts", "emov"} else None
    return voices


@app.post("/api/voices")
async def create_voice(file: UploadFile = File(...), name: str = Form(...), start: float = Form(0),
                       consent: bool = Form(False), locale: str = Form("vi-VN")):
    name = name.strip()
    if not consent:
        raise HTTPException(400, "Bạn cần xác nhận quyền sử dụng giọng nói này.")
    if not 1 <= len(name) <= 60:
        raise HTTPException(422, "Tên giọng phải dài 1–60 ký tự.")
    if start < 0:
        raise HTTPException(422, "Thời điểm bắt đầu không hợp lệ.")
    if locale not in {"vi-VN", "en-US"}:
        raise HTTPException(422, "Ngôn ngữ giọng nói không hợp lệ.")
    source_name = Path(file.filename or "media").name[:150]
    source = await _save_upload(file, max_bytes=None)
    voice_id = uuid.uuid4().hex
    clip = VOICES / f"{voice_id}.wav"
    original = VOICE_SOURCES / f"{voice_id}{source.suffix}"
    try:
        duration = await asyncio.to_thread(_duration, source)
        if duration - start < 3:
            raise HTTPException(422, "Cần ít nhất 3 giây tiếng nói sau điểm bắt đầu.")
        await asyncio.to_thread(_run, "ffmpeg", "-y", "-v", "error", "-ss", str(start), "-i", str(source), "-t", "8", "-vn", "-ac", "1", "-ar", "44100", str(clip))
        shutil.move(source, original)
        item = {"id": voice_id, "name": name, "source": source_name, "locale": locale,
                "created_at": datetime.now(timezone.utc).isoformat(), "duration": round(duration, 1),
                "reference_duration": round(min(8, duration - start), 1),
                "source_file": original.name, "source_url": f"/api/voices/{voice_id}/source",
                "preview_url": f"/api/media/voices/{voice_id}.wav"}
        with _voice_lock:
            voices = _read_voices()
            voices.append(item)
            _write_voices(voices)
        return item
    except Exception:
        clip.unlink(missing_ok=True)
        original.unlink(missing_ok=True)
        raise
    finally:
        source.unlink(missing_ok=True)


@app.delete("/api/voices/{voice_id}")
def delete_voice(voice_id: str):
    training = uploaded_audio.load(voice_id)
    if training and training.get("status") in {"queued", "preparing", "waiting_for_gpu", "training", "evaluating"}:
        raise HTTPException(409, "Đang huấn luyện từ file này. Chờ hoàn tất trước khi xóa.")
    with _voice_lock:
        voices = _read_voices()
        updated = [item for item in voices if item["id"] != voice_id]
        if len(updated) == len(voices):
            raise HTTPException(404, "Không tìm thấy giọng nói.")
        for item in updated:
            if item.get("origin_voice_id") == voice_id:
                item.pop("origin_voice_id", None)
                item.pop("source_url", None)
                preview = VOICES / f"{item['id']}.wav"
                if preview.is_file():
                    item["duration"] = round(_duration(preview), 1)
        _write_voices(updated)
    (VOICES / f"{voice_id}.wav").unlink(missing_ok=True)
    if item := next((voice for voice in voices if voice["id"] == voice_id), None):
        if item.get("source_file") and re.fullmatch(re.escape(voice_id) + r"\.[a-z0-9]+", item["source_file"]):
            (VOICE_SOURCES / item["source_file"]).unlink(missing_ok=True)
    return {"ok": True}


@app.get("/api/voices/{voice_id}/training")
def uploaded_voice_training(voice_id: str):
    _get_voice(voice_id)
    state = uploaded_audio.load(voice_id) or {"voice_id": voice_id, "status": "not_started"}
    if state.get("status") == "training" and state.get("run"):
        log = RUNS / state["run"] / "official" / state["run"] / "train_log.jsonl"
        if log.is_file():
            try:
                steps = [json.loads(line).get("step", 0) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
                state["optimizer_updates"] = max(steps, default=0)
            except (OSError, ValueError):
                pass
        run_status = RUNS / state["run"] / "status.json"
        if run_status.is_file():
            try:
                state["training_stage"] = json.loads(run_status.read_text(encoding="utf-8")).get("status")
            except (OSError, ValueError):
                pass
    return state


@app.post("/api/voices/{voice_id}/refresh-reference")
def refresh_uploaded_reference(voice_id: str):
    voice = _get_voice(voice_id)
    full_dataset = PREPARED / f"upload_{voice_id}_full_v2"
    dataset = full_dataset if (full_dataset / "manifest.json").is_file() else PREPARED / f"upload_{voice_id}"
    manifest_path = dataset / "manifest.json"
    train_path = dataset / "train.jsonl"
    if not voice.get("source_file") or not manifest_path.is_file() or not train_path.is_file():
        raise HTTPException(422, "Cần chuẩn bị dữ liệu từ file gốc trước khi chọn mẫu giọng.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("voice_id") != voice_id:
        raise HTTPException(422, "Bộ dữ liệu không thuộc giọng này.")
    rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    try:
        reference, confidence = uploaded_audio.representative_speech(rows, dataset / "raw_audio")
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    source = dataset / "raw_audio" / reference["file_name"]
    if not source.is_file():
        raise HTTPException(422, "Mẫu giọng trong bộ dữ liệu đã mất.")
    preview = VOICES / f"{voice_id}.wav"
    temporary = VOICES / f"{voice_id}.tmp.wav"
    shutil.copyfile(source, temporary)
    with _voice_lock:
        voices = _read_voices()
        item = next((row for row in voices if row["id"] == voice_id), None)
        if item is None:
            temporary.unlink(missing_ok=True)
            raise HTTPException(404, "Giọng đã bị xóa.")
        temporary.replace(preview)
        item.update({"reference_duration": reference["duration"],
                     "reference_source_start": reference["source_start"],
                     "reference_similarity_to_dominant_voice": confidence,
                     "reference_text": reference["text"], "reference_verified_speech": True,
                     "reference_updated_at": datetime.now(timezone.utc).isoformat(),
                     "preview_url": f"/api/media/voices/{voice_id}.wav?v={uuid.uuid4().hex[:8]}"})
        _write_voices(voices)
    return item


@app.post("/api/voices/{voice_id}/train-full")
def train_full_voice(voice_id: str):
    voice = _get_voice(voice_id)
    if not voice.get("source_file") or not (VOICE_SOURCES / voice["source_file"]).is_file():
        raise HTTPException(422, "Giọng này không còn file gốc đầy đủ; hãy tải lên lại để fine-tune.")
    previous = uploaded_audio.load(voice_id)
    if previous and previous.get("status") not in {"failed", "not_started", "complete"}:
        raise HTTPException(409, "File này đã được gửi huấn luyện. Xem tiến độ ở thư viện giọng.")
    attempt = (previous or {}).get("attempt", 0) + 1
    uploaded_audio.save(voice_id, status="queued", stage="queued", source_seconds=voice["duration"],
                        source_name=voice["source"], name=voice["name"], locale=voice["locale"], attempt=attempt,
                        error=None)
    unit = f"vocalab-upload-{voice_id[:16]}-r{attempt}"
    command = ["systemd-run", "--user", "--unit", unit, "--description", f"Vocalab full file voice {voice_id[:8]}",
               f"--working-directory={ROOT}", str(ROOT / ".venv/bin/python"), "-m", "training.upload_train", "--voice", voice_id]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=15)
    if completed.returncode:
        uploaded_audio.save(voice_id, status="failed", error=f"Không khởi động được worker: {completed.stderr.strip()[-300:]}")
        raise HTTPException(500, completed.stderr.strip()[-300:])
    return uploaded_audio.load(voice_id)


@app.get("/api/voices/{voice_id}/source")
def voice_source(voice_id: str):
    voice = _get_voice(voice_id)
    name = voice.get("source_file")
    if not name or not re.fullmatch(re.escape(voice_id) + r"\.[a-z0-9]+", name):
        raise HTTPException(404, "Giọng này không có tệp nguồn đầy đủ.")
    path = VOICE_SOURCES / name
    if not path.is_file():
        raise HTTPException(404, "Tệp nguồn đã mất.")
    return FileResponse(path, filename=voice["source"], media_type=_source_media_type(path), content_disposition_type="inline")


def _synthesize(voice_id: str, segments: list[dict], output: Path, engine_name: str = "vieneu", locale: str = "vi-VN") -> dict:
    voice = _get_voice(voice_id)
    reference = VOICES / f"{voice_id}.wav"
    if not reference.exists():
        raise HTTPException(404, "File giọng nói đã mất.")
    synthesis_segments = [{**segment, "text": normalize_for_speech(segment["text"], locale)} for segment in segments]
    with _tts_lock, _gpu_slot(wait_seconds=300, min_free_mib=4000):
        try:
            with tempfile.TemporaryDirectory(prefix="vocalab-tts-", dir=UPLOADS) as temp:
                temp_path = Path(temp)
                raw_files = [temp_path / f"{index:02}.wav" for index in range(len(segments))]
                if engine_name == "voxcpm2":
                    python = DATA / "upstream/voxcpm/.venv/bin/python"
                    if not python.is_file():
                        raise HTTPException(422, "Chưa cài VoxCPM2 trên máy.")
                    script_file = temp_path / "script.json"
                    script_file.write_text(json.dumps(synthesis_segments, ensure_ascii=False), encoding="utf-8")
                    _run(str(python), str(ROOT / "training/synthesize_voxcpm.py"), "--reference", str(reference),
                         "--locale", locale, "--script-file", str(script_file), "--output-dir", str(temp_path))
                elif voice.get("kind") == "finetuned":
                    run_dir = RUNS / voice["run"]
                    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
                    if status.get("status") not in {"evaluated", "full_trained"}:
                        raise HTTPException(422, "Giọng fine-tune chưa hoàn tất.")
                    upstream = Path(os.environ.get("VIENEU_UPSTREAM", DATA / "upstream/vieneu"))
                    script_file = temp_path / "script.json"
                    script_file.write_text(json.dumps(synthesis_segments, ensure_ascii=False), encoding="utf-8")
                    _run(str(upstream / ".venv/bin/python"), str(ROOT / "training/synthesize.py"), "--model", status["merged"],
                         "--voice", status["speaker_id"], "--script-file", str(script_file), "--output-dir", str(temp_path))
                else:
                    upstream = Path(os.environ.get("VIENEU_UPSTREAM", DATA / "upstream/vieneu"))
                    script_file = temp_path / "script.json"
                    script_file.write_text(json.dumps(synthesis_segments, ensure_ascii=False), encoding="utf-8")
                    _run(str(upstream / ".venv/bin/python"), str(ROOT / "training/synthesize_reference.py"),
                         "--reference", str(reference), "--script-file", str(script_file), "--output-dir", str(temp_path))
                return combine_delivery(raw_files, segments, output)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(500, f"Tạo giọng thất bại: {exc}") from exc


def _separate(source: Path, directory: Path) -> tuple[Path, Path]:
    python = DATA / "upstream/demucs/.venv/bin/python"
    if not python.is_file():
        raise HTTPException(422, "Demucs chưa được cài trên máy.")
    command = [str(python), "-m", "demucs", "--two-stems", "vocals", "-d", "cuda", "--segment", "7",
               "--out", str(directory), str(source)]
    try:
        with _gpu_slot(min_free_mib=3000):
            completed = subprocess.run(command, capture_output=True, timeout=1800)
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(408, "Tách vocal quá thời gian cho phép.") from exc
    if completed.returncode:
        raise HTTPException(422, "Không tách được vocal: " + completed.stderr.decode(errors="replace")[-450:])
    folder = directory / "htdemucs" / source.stem
    vocals, background = folder / "vocals.wav", folder / "no_vocals.wav"
    if not vocals.is_file() or not background.is_file():
        raise HTTPException(500, "Demucs không trả về đủ hai stem.")
    return vocals, background


@app.post("/api/separate")
async def separate_source(file: UploadFile = File(...)):
    source = await _save_upload(file)
    output_id = uuid.uuid4().hex
    try:
        with tempfile.TemporaryDirectory(prefix="vocalab-separate-", dir=UPLOADS) as temp:
            directory = Path(temp)
            audio = directory / "input.wav"
            await asyncio.to_thread(decode_audio, source, audio)
            vocals, background = await asyncio.to_thread(_separate, audio, directory)
            voice_out = OUTPUTS / f"{output_id}.wav"
            music_out = OUTPUTS / f"{uuid.uuid4().hex}.wav"
            shutil.copyfile(vocals, voice_out)
            shutil.copyfile(background, music_out)
            return {"vocals_url": f"/api/media/outputs/{voice_out.name}",
                    "background_url": f"/api/media/outputs/{music_out.name}"}
    finally:
        source.unlink(missing_ok=True)


@app.post("/api/tts")
async def synthesize(voice_id: str = Form(...), text: str = Form(...), delivery: str | None = Form(None),
                     video: UploadFile | None = File(None), source: UploadFile | None = File(None),
                     source_mode: str = Form("mute"), engine: str = Form("vieneu"), locale: str = Form("vi-VN")):
    voice = _get_voice(voice_id)
    text = text.strip()
    if not text or len(text) > 5000:
        raise HTTPException(422, "Văn bản cần dài 1–5000 ký tự.")
    try:
        if engine not in {"vieneu", "voxcpm2"}:
            raise HTTPException(422, "Mô hình giọng nói không hợp lệ.")
        if locale not in {"vi-VN", "en-US"}:
            raise HTTPException(422, "Ngôn ngữ đọc không hợp lệ.")
        voice_locale = voice.get("locale")
        if voice_locale is None:
            voice_locale = "vi-VN" if voice.get("source") == "vivos" else "en" if voice.get("source") in {"libritts", "emov"} else None
        if voice_locale == "vi-VN" and locale == "en-US":
            raise HTTPException(422, "Giọng mẫu này được ghi bằng tiếng Việt. Hãy chọn hoặc thu giọng tiếng Anh Mỹ.")
        if voice_locale in {"en", "en-US"} and locale == "vi-VN":
            raise HTTPException(422, "Giọng mẫu này được ghi bằng tiếng Anh. Hãy chọn hoặc thu giọng tiếng Việt.")
        if locale == "en-US" and engine == "vieneu" and voice.get("kind") != "finetuned":
            raise HTTPException(422, "VieNeu cho Anh Mỹ cần giọng đã fine-tune từ bản ghi tiếng Anh. Với giọng tham chiếu, hãy chọn VoxCPM2.")
        segments = plan_delivery(text, delivery)
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise HTTPException(422, str(exc)) from exc
    output_id = uuid.uuid4().hex
    wav = OUTPUTS / f"{output_id}.wav"
    video_path = None
    source_path = None
    try:
        if source_mode not in {"mute", "keep", "duck", "remove_vocals"}:
            raise HTTPException(422, "Chế độ âm thanh gốc không hợp lệ.")
        if source is not None and video is not None:
            raise HTTPException(422, "Chọn một tệp nguồn audio hoặc video.")
        if video is not None:
            if Path(video.filename or "").suffix.lower() not in VIDEO:
                raise HTTPException(415, "Video cần là MP4, MOV, MKV hoặc WebM.")
            video_path = await _save_upload(video)
        if source is not None:
            source_path = await _save_upload(source)
            if source_path.suffix.lower() in VIDEO:
                video_path = source_path
        generated = UPLOADS / f"{uuid.uuid4().hex}.wav"
        try:
            delivery_result = await asyncio.to_thread(_synthesize, voice_id, segments, generated, engine, locale)
            positions = [item["at"] for item in segments]
            if source_path or video_path:
                with tempfile.TemporaryDirectory(prefix="vocalab-mix-", dir=UPLOADS) as temp:
                    directory = Path(temp)
                    original = directory / "original.wav"
                    media_source = source_path or video_path
                    try:
                        await asyncio.to_thread(decode_audio, media_source, original)
                    except ValueError:
                        if source_mode != "mute":
                            raise HTTPException(422, "Tệp nguồn không có track âm thanh để giữ hoặc tách.")
                        original = None
                    background = None
                    if source_mode == "remove_vocals" and original:
                        _, separated_background = await asyncio.to_thread(_separate, original, directory)
                        background = directory / "background_24k.wav"
                        await asyncio.to_thread(decode_audio, separated_background, background)
                    delivery_result = await asyncio.to_thread(mix_audio, generated, delivery_result["segments"], positions,
                                                               original, wav, source_mode, background)
            else:
                delivery_result = await asyncio.to_thread(mix_audio, generated, delivery_result["segments"], positions,
                                                           None, wav)
        finally:
            generated.unlink(missing_ok=True)
        result = {"audio_url": f"/api/media/outputs/{wav.name}", "video_url": None,
                  "delivery": delivery_result, "locale": locale, "engine": engine,
                  "compute_device": "cuda" if engine == "vieneu" else "model_default",
                  "pronunciation_note": "Giọng Anh chưa xác minh vùng miền; nên dùng mẫu en-US để có âm Mỹ rõ hơn." if locale == "en-US" and voice_locale != "en-US" else None}
        if video_path:
            video_duration = await asyncio.to_thread(_duration, video_path)
            if any(marker["end"] > video_duration + 0.05 for marker in delivery_result["placed_segments"]):
                wav.unlink(missing_ok=True)
                raise HTTPException(422, f"Một đoạn giọng AI vượt quá thời lượng video ({video_duration:.1f} giây).")
            mp4 = OUTPUTS / f"{output_id}.mp4"
            await asyncio.to_thread(_run, "ffmpeg", "-y", "-v", "error", "-i", str(video_path), "-i", str(wav), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-af", "apad", "-shortest", "-movflags", "+faststart", str(mp4))
            result["video_url"] = f"/api/media/outputs/{mp4.name}"
        return result
    except ValueError as exc:
        wav.unlink(missing_ok=True)
        raise HTTPException(422, str(exc)) from exc
    finally:
        for path in {video_path, source_path} - {None}:
            path.unlink(missing_ok=True)


def _transcribe(path: Path, language: str):
    global _stt_device
    with _stt_lock, _gpu_slot(min_free_mib=2500):
        from training.gpu_whisper import make_whisper
        model, _stt_device = make_whisper(os.environ.get("WHISPER_MODEL", "small"))
        try:
            segments, info = model.transcribe(str(path), language=None if language == "auto" else language, vad_filter=True, beam_size=5)
            rows = [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()} for s in segments]
            return rows, info.language
        except Exception as exc:
            raise HTTPException(500, f"STT thất bại: {exc}") from exc
        finally:
            del model
            gc.collect()


@app.post("/api/stt")
async def transcribe(file: UploadFile = File(...), language: str = Form("auto")):
    if language not in {"auto", "vi", "en"}:
        raise HTTPException(422, "Ngôn ngữ không hợp lệ.")
    source = await _save_upload(file)
    output_id = uuid.uuid4().hex
    audio = UPLOADS / f"{output_id}.wav"
    try:
        await asyncio.to_thread(_run, "ffmpeg", "-y", "-v", "error", "-i", str(source), "-vn", "-ac", "1", "-ar", "16000", str(audio))
        rows, detected = await asyncio.to_thread(_transcribe, audio, language)
        transcript = "\n".join(row["text"] for row in rows)
        (OUTPUTS / f"{output_id}.txt").write_text(transcript, encoding="utf-8")
        srt = "\n\n".join(f"{i}\n{_srt_time(row['start'])} --> {_srt_time(row['end'])}\n{row['text']}" for i, row in enumerate(rows, 1))
        (OUTPUTS / f"{output_id}.srt").write_text(srt + ("\n" if srt else ""), encoding="utf-8")
        return {"text": transcript, "language": detected, "segments": rows, "txt_url": f"/api/media/outputs/{output_id}.txt", "srt_url": f"/api/media/outputs/{output_id}.srt"}
    finally:
        source.unlink(missing_ok=True)
        audio.unlink(missing_ok=True)


@app.get("/api/media/{kind}/{filename}")
def media(kind: str, filename: str):
    if kind not in {"voices", "outputs"} or not re.fullmatch(r"[a-f0-9]{32}\.(wav|mp4|txt|srt)", filename):
        raise HTTPException(404)
    base = VOICES if kind == "voices" else OUTPUTS
    path = base / filename
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, filename=filename if kind == "outputs" else None)


@app.post("/api/editor/export")
async def export_timeline(files: list[UploadFile] = File(...), timeline: str = Form(...),
                          overlay: UploadFile | None = File(None)):
    try:
        recipe = json.loads(timeline)
    except json.JSONDecodeError as exc:
        raise HTTPException(422, "Timeline JSON không hợp lệ.") from exc
    sources = []
    overlay_path = None
    output_id = uuid.uuid4().hex
    try:
        if len(files) > 12:
            raise HTTPException(422, "Một lần xuất tối đa 12 tệp nguồn.")
        for file in files:
            sources.append(await _save_upload(file))
        if overlay is not None:
            overlay_path = await _save_upload(overlay)
        with tempfile.TemporaryDirectory(prefix="vocalab-editor-", dir=UPLOADS) as temp:
            try:
                result = await asyncio.to_thread(render_timeline, sources, recipe, OUTPUTS / output_id,
                                                 Path(temp), overlay_path)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
        return {"url": f"/api/media/outputs/{Path(result['output']).name}", **{k: v for k, v in result.items() if k != "output"}}
    except Exception:
        for suffix in (".mp4", ".wav"):
            (OUTPUTS / f"{output_id}{suffix}").unlink(missing_ok=True)
        raise
    finally:
        for path in sources:
            path.unlink(missing_ok=True)
        if overlay_path:
            overlay_path.unlink(missing_ok=True)


DIST = ROOT / "dist"
if DIST.is_dir():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="frontend")
