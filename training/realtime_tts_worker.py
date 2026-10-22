"""Long-lived VieNeu CUDA worker used by the speech-to-speech page.

The worker speaks a tiny JSON-lines protocol over stdin/stdout.  Keeping the
model alive avoids paying the model-load cost for every short utterance.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from contextlib import redirect_stdout
from pathlib import Path


def emit(payload: dict) -> None:
    sys.__stdout__.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.__stdout__.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["reference", "trained"], required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--voice")
    args = parser.parse_args()
    if args.mode == "trained" and (not args.model or not args.voice):
        parser.error("trained mode requires --model and --voice")

    started = time.perf_counter()
    # VieNeu prints model progress to stdout. Keep stdout exclusively for the
    # protocol so the parent process can parse every response deterministically.
    with redirect_stdout(sys.stderr):
        from vieneu import Vieneu

        options = {"mode": "v3turbo", "backend": "pytorch", "device": "cuda"}
        if args.mode == "trained":
            options["backbone_repo"] = str(args.model.resolve())
        engine = Vieneu(**options)
    emit({"event": "ready", "load_ms": round((time.perf_counter() - started) * 1000), "mode": args.mode})

    for raw in sys.stdin:
        request_id = None
        try:
            request = json.loads(raw)
            request_id = request["id"]
            text = str(request["text"]).strip()
            output = Path(request["output"]).resolve()
            if not text or len(text) > 800:
                raise ValueError("text must contain 1-800 characters")
            output.parent.mkdir(parents=True, exist_ok=True)
            sampling = request.get("sampling") or {}
            began = time.perf_counter()
            with redirect_stdout(sys.stderr):
                if args.mode == "trained":
                    audio = engine.infer(text, voice=args.voice, use_ref_codes=False, **sampling)
                else:
                    reference = Path(request["reference"]).resolve()
                    if not reference.is_file():
                        raise FileNotFoundError("reference WAV is missing")
                    audio = engine.infer(text, ref_audio=str(reference), denoise=True, use_ref_codes=True, **sampling)
                engine.save(audio, str(output))
            emit({"id": request_id, "ok": True,
                  "inference_ms": round((time.perf_counter() - began) * 1000)})
        except Exception as exc:  # keep the worker alive after a bad utterance
            traceback.print_exc(file=sys.stderr)
            emit({"id": request_id, "ok": False, "error": str(exc)})


if __name__ == "__main__":
    main()
