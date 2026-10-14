"""Verify, merge and register a public-corpus adapter from the GitHub Release."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = json.loads((ROOT / "models/catalog.json").read_text(encoding="utf-8"))
REPO = "minhnhatdepzai/voice-deeplearning"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_files(asset: Path, destination: Path) -> None:
    with tarfile.open(asset, "r:gz") as archive:
        for member in archive:
            target = destination / member.name
            if not member.isfile() or target.resolve().is_relative_to(destination.resolve()) is False:
                raise ValueError(f"Unsafe archive entry: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"Cannot read archive entry: {member.name}")
            with target.open("wb") as output:
                shutil.copyfileobj(source, output)


def install(run: str, local_asset: Path | None) -> None:
    entry = next((item for item in CATALOG["models"] if item["run"] == run), None)
    if entry is None:
        raise ValueError(f"Unknown run: {run}")
    final = ROOT / "data/training_runs" / run
    if final.exists():
        raise FileExistsError(f"Run already exists: {final}")
    upstream = ROOT / "data/upstream/vieneu"
    python = upstream / ".venv/bin/python"
    merge_script = upstream / "finetune/merge_lora.py"
    if not python.is_file() or not merge_script.is_file():
        raise FileNotFoundError("Run bash training/setup.sh first")
    final.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".install-{run}-", dir=final.parent) as temporary:
        stage = Path(temporary)
        asset = local_asset or stage / entry["asset"]
        if local_asset is None:
            url = f"https://github.com/{REPO}/releases/download/{CATALOG['release_tag']}/{entry['asset']}"
            urllib.request.urlretrieve(url, asset)
        if sha256(asset) != entry["asset_sha256"]:
            raise ValueError("Release asset SHA-256 mismatch")
        extracted = stage / "files"
        extracted.mkdir()
        extract_files(asset, extracted)
        adapter = extracted / "adapter"
        preset = extracted / "voices_v3_turbo.json"
        preview = extracted / "preview.wav"
        metadata = json.loads((extracted / "metadata.json").read_text(encoding="utf-8"))
        if metadata["run"] != run or metadata["source"] != entry["source"]:
            raise ValueError("Release metadata does not match catalog")
        for path, expected in ((adapter / "adapter_model.safetensors", entry["adapter_sha256"]),
                               (preset, entry["preset_sha256"]), (preview, entry["preview_sha256"])):
            if sha256(path) != expected:
                raise ValueError(f"SHA-256 mismatch: {path.name}")
        merged = extracted / "official" / run / "merged"
        subprocess.run([str(python), str(merge_script), "--adapter", str(adapter), "--out", str(merged)],
                       cwd=ROOT, check=True)
        shutil.copyfile(preset, merged / "voices_v3_turbo.json")
        official_adapter = extracted / "official" / run / "adapter"
        official_adapter.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(adapter), official_adapter)
        (extracted / "release_metadata.json").write_text(json.dumps(entry, ensure_ascii=False, indent=2) + "\n",
                                                           encoding="utf-8")
        (extracted / "status.json").write_text(json.dumps({
            "run": run, "dataset": str(ROOT / "data/prepared" / f"release_{run}"),
            "source": entry["source"], "speaker_id": entry["speaker_id"], "status": "evaluated",
            "release_import": True, "adapter": str(final / "official" / run / "adapter/adapter_model.safetensors"),
            "adapter_sha256": entry["adapter_sha256"], "adapter_reload_verified": True,
            "merged": str(final / "official" / run / "merged"),
            "optimizer_updates": entry["optimizer_updates"], "test_items": entry["test_items"],
            "generated_wer": entry["wer_generated"], "reference_wer": entry["wer_reference"],
            "speaker_similarity": entry["speaker_cosine_generated"],
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        extracted.rename(final)
    print(f"Installed {run}: {final}")
    request = urllib.request.Request("http://127.0.0.1:8000/api/training/register",
                                     data=json.dumps({"run": run}).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            voice = json.load(response)
        print(f"Registered in Studio: {voice['name']}")
    except urllib.error.URLError:
        print("Start ./run.sh, then register this run in the Training page or call POST /api/training/register.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", choices=[item["run"] for item in CATALOG["models"]])
    parser.add_argument("--asset", type=Path, help="Use a previously downloaded tar.gz")
    args = parser.parse_args()
    install(args.run, args.asset)
