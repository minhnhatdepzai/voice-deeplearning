"""Package evaluated public-corpus LoRA adapters for a GitHub Release.

Uploads, raw recordings, merged base weights and incomplete runs are excluded.
The output is ignored by Git; models/catalog.json is the publishable index.
"""
from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "data/training_runs"
OUT = ROOT / "release_assets"
CATALOG = ROOT / "models/catalog.json"
ALLOWED = {"vivos", "libritts", "ljspeech", "emov"}
SOURCES = json.loads((ROOT / "training/sources.json").read_text(encoding="utf-8"))
TAG = "v0.1.0-models"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_file(archive: tarfile.TarFile, path: Path, name: str) -> None:
    info = archive.gettarinfo(str(path), arcname=name)
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    with path.open("rb") as handle:
        archive.addfile(info, handle)


def add_json(archive: tarfile.TarFile, obj: dict, name: str) -> None:
    payload = (json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode()
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mode = 0o644
    info.mtime = 0
    archive.addfile(info, io.BytesIO(payload))


def main() -> None:
    OUT.mkdir(exist_ok=True)
    CATALOG.parent.mkdir(exist_ok=True)
    entries = []
    for path in sorted(RUNS.glob("*/status.json")):
        status = json.loads(path.read_text(encoding="utf-8"))
        run = path.parent.name
        source = status.get("source")
        if source not in ALLOWED or status.get("status") != "evaluated":
            continue
        if not status.get("adapter_reload_verified"):
            raise RuntimeError(f"Adapter was not cold-reload verified: {run}")
        adapter = path.parent / "official" / run / "adapter"
        weights = adapter / "adapter_model.safetensors"
        preset = path.parent / "official" / run / "merged/voices_v3_turbo.json"
        preview = path.parent / "evaluation/test_01.wav"
        if not all(p.is_file() for p in (weights, preset, preview)):
            raise FileNotFoundError(f"Missing complete release files for {run}")
        if sha256(weights) != status["adapter_sha256"]:
            raise ValueError(f"Adapter SHA-256 mismatch: {run}")
        results = json.loads((path.parent / "evaluation/results.json").read_text(encoding="utf-8"))
        speaker = json.loads((path.parent / "evaluation/speaker_similarity.json").read_text(encoding="utf-8"))
        manifest = json.loads((Path(status["dataset"]) / "manifest.json").read_text(encoding="utf-8"))
        train_args = json.loads((path.parent / "official" / run / "train_args.json").read_text(encoding="utf-8"))
        config = {key: train_args[key] for key in ("batch_size", "grad_accum", "lr", "target")}
        config["rank"] = train_args["r"]
        entry = {
            "run": run,
            "source": source,
            "speaker_id": status["speaker_id"],
            "locale": "vi-VN" if source == "vivos" else "en-US",
            "license": SOURCES[source]["license"],
            "source_url": SOURCES[source]["original"],
            "upstream_commit": status["upstream_commit"],
            "optimizer_updates": status["optimizer_updates"],
            "config": config,
            "counts": manifest["counts"],
            "train_minutes": manifest["minutes"]["train"],
            "test_items": results["test_items"],
            "wer_reference": results["metrics"]["reference_wav"]["wer"],
            "wer_generated": results["metrics"]["generated_wav"]["wer"],
            "speaker_cosine_reference": speaker["metrics"]["original_to_enrollment"],
            "speaker_cosine_generated": speaker["metrics"]["generated_to_enrollment"],
            "adapter_sha256": status["adapter_sha256"],
            "preset_sha256": sha256(preset),
            "preview_sha256": sha256(preview),
            "asset": f"{run}.tar.gz",
        }
        asset = OUT / entry["asset"]
        with tarfile.open(asset, "w:gz") as tar:
            for item in sorted(adapter.iterdir()):
                if item.is_file():
                    add_file(tar, item, f"adapter/{item.name}")
            add_file(tar, preset, "voices_v3_turbo.json")
            add_file(tar, preview, "preview.wav")
            add_json(tar, entry, "metadata.json")
        entry["asset_sha256"] = sha256(asset)
        entry["asset_bytes"] = asset.stat().st_size
        entries.append(entry)
        print(f"{run}: {entry['asset_bytes']} bytes")
    if not entries:
        raise RuntimeError("No evaluated public-corpus adapters found")
    CATALOG.write_text(json.dumps({"release_tag": TAG, "models": entries}, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
    print(f"Wrote {CATALOG.relative_to(ROOT)} with {len(entries)} models")


if __name__ == "__main__":
    main()
