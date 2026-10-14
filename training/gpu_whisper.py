"""Prefer CUDA for faster-whisper using the CUDA libraries bundled with VieNeu."""
from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path

from faster_whisper import WhisperModel

ROOT = Path(__file__).resolve().parents[1]


def _free_gpu_mib() -> int:
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, check=True, timeout=5)
        return int(result.stdout.splitlines()[0].strip())
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError, IndexError):
        return 0


def _preload_cuda() -> None:
    site_packages = next((ROOT / "data/upstream/vieneu/.venv/lib").glob("python*/site-packages"))
    base = site_packages / "nvidia"
    for name in ("cuda_runtime/lib/libcudart.so.12", "cublas/lib/libcublasLt.so.12",
                 "cublas/lib/libcublas.so.12", "cudnn/lib/libcudnn.so.9"):
        ctypes.CDLL(str(base / name), mode=ctypes.RTLD_GLOBAL)


def make_whisper(model_name: str = "small", min_free_mib: int = 2500) -> tuple[WhisperModel, str]:
    """Use GPU when it has room; return device label for status and reports."""
    if _free_gpu_mib() >= min_free_mib:
        try:
            _preload_cuda()
            return WhisperModel(model_name, device="cuda", compute_type="float16"), "cuda/float16"
        except (OSError, RuntimeError, StopIteration):
            pass
    return WhisperModel(model_name, device="cpu", compute_type="int8"), "cpu/int8"
