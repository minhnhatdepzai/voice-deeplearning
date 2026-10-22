"""Build one-speaker VieNeu datasets from verified public archives.

Archives stay local. Only selected speaker clips are decoded. No utterance or
normalized transcript is shared between train, validation and test.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import subprocess
import tarfile
import unicodedata
import wave
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = json.loads((Path(__file__).parent / "sources.json").read_text(encoding="utf-8"))


def text_key(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def vivos_rows(archive: zipfile.ZipFile) -> list[dict]:
    names = archive.namelist()
    prompt_name = next((n for n in names if n.endswith("vivos/train/prompts.txt")), None)
    if prompt_name is None:
        raise ValueError("VIVOS train/prompts.txt missing")
    prompts = {}
    for line in archive.read(prompt_name).decode("utf-8-sig").splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            prompts[parts[0]] = parts[1]
    rows = []
    for name in names:
        match = re.search(r"vivos/train/waves/([^/]+)/([^/]+)\.wav$", name)
        if not match:
            continue
        speaker, stem = match.groups()
        text = prompts.get(stem) or prompts.get(f"{speaker}/{stem}")
        if text:
            rows.append({"speaker": speaker, "member": name, "text": text})
    return rows


def libri_rows(archive: tarfile.TarFile) -> list[dict]:
    members = {m.name: m for m in archive.getmembers() if m.isfile()}
    rows = []
    for name in members:
        if not name.endswith(".normalized.txt"):
            continue
        audio_name = name.removesuffix(".normalized.txt") + ".wav"
        if audio_name not in members:
            continue
        match = re.search(r"LibriTTS/dev-clean/([^/]+)/", name)
        if not match:
            continue
        text = archive.extractfile(members[name]).read().decode("utf-8-sig").strip()
        if text:
            rows.append({"speaker": match.group(1), "member": audio_name, "text": text})
    return rows


def emov_rows(archive: tarfile.TarFile, archive_path: Path) -> list[dict]:
    transcript_file = archive_path.parent / "cmuarctic.data"
    if not transcript_file.is_file():
        raise FileNotFoundError(f"EmoV-DB needs the CMU Arctic transcript: {transcript_file}")
    prompts = {number: text for number, text in re.findall(r'\(\s*arctic_a(\d{4})\s+"([^"]+)"\s*\)',
                                                              transcript_file.read_text(encoding="utf-8"))}
    match = re.fullmatch(r"([a-z]+)_([A-Za-z]+)\.tar\.gz", archive_path.name)
    if not match:
        raise ValueError("EmoV archive must be named like jenie_Amused.tar.gz")
    speaker = f"{match.group(1)}_{match.group(2).lower()}"
    rows = []
    for member in archive.getmembers():
        if not member.isfile():
            continue
        found = re.fullmatch(r"[a-z]+_[0-9]+-[0-9]+_(\d{4})\.wav", Path(member.name).name)
        if found and found.group(1) in prompts:
            rows.append({"speaker": speaker, "member": member.name, "text": prompts[found.group(1)]})
    return rows


def ljspeech_rows(archive: tarfile.TarFile) -> list[dict]:
    members = {member.name: member for member in archive.getmembers() if member.isfile()}
    metadata = next((name for name in members if name.endswith(("/metadata.csv", "/transcripts.csv"))), None)
    if metadata is None:
        raise ValueError("LJ Speech metadata.csv/transcripts.csv missing")
    rows = []
    lines = archive.extractfile(members[metadata]).read().decode("utf-8-sig").splitlines()
    base = metadata.rsplit("/", 1)[0]
    for fields in csv.reader(lines, delimiter="|"):
        if len(fields) < 2 or not re.fullmatch(r"LJ\d{3}-\d{4}", fields[0]):
            continue
        audio_name = f"{base}/wavs/{fields[0]}.wav"
        if audio_name in members:
            rows.append({"speaker": "LJ", "member": audio_name, "text": fields[2] if len(fields) > 2 and fields[2] else fields[1]})
    return rows


def read_archive(source: str, path: Path):
    if source == "vivos":
        archive = zipfile.ZipFile(path)
        return archive, vivos_rows(archive)
    archive = tarfile.open(path, "r:*")
    if source == "emov":
        rows = emov_rows(archive, path)
    elif source == "ljspeech":
        rows = ljspeech_rows(archive)
    else:
        rows = libri_rows(archive)
    return archive, rows


def extract_bytes(archive, member: str) -> bytes:
    if isinstance(archive, zipfile.ZipFile):
        return archive.read(member)
    return archive.extractfile(member).read()


def ffmpeg_wav(data: bytes, target: Path) -> None:
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", "pipe:0", "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(target)],
        input=data, capture_output=True, timeout=120,
    )
    if proc.returncode:
        raise ValueError(proc.stderr.decode(errors="replace")[-300:])


def wav_stats(path: Path) -> tuple[float, float]:
    with wave.open(str(path), "rb") as stream:
        samples = stream.readframes(stream.getnframes())
        duration = stream.getnframes() / stream.getframerate()
    import array
    signal = array.array("h")
    signal.frombytes(samples)
    if not signal:
        return duration, 0.0
    rms = (sum(x * x for x in signal) / len(signal)) ** 0.5 / 32768
    return duration, rms


def build(source: str, archive_path: Path, speaker: str, output: Path, max_clips: int = 300) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output exists and is not empty: {output}")
    archive, rows = read_archive(source, archive_path)
    try:
        speakers = Counter(row["speaker"] for row in rows)
        if speaker not in speakers:
            raise ValueError(f"Speaker {speaker!r} missing. Available: {speakers.most_common(12)}")
        selected = sorted((row for row in rows if row["speaker"] == speaker), key=lambda r: hashlib.sha256(r["member"].encode()).hexdigest())[:max_clips]
        if isinstance(archive, tarfile.TarFile):
            # gzip is a stream: read selected members in archive order to avoid
            # decompressing from the beginning for every random access.
            selected.sort(key=lambda row: archive.getmember(row["member"]).offset_data)
        (output / "raw_audio").mkdir(parents=True, exist_ok=True)
        prepared = []
        seen_text = set()
        seen_audio = set()
        rejected = Counter()
        for row in selected:
            key = text_key(row["text"])
            if key in seen_text:
                rejected["duplicate_text"] += 1
                continue
            raw = extract_bytes(archive, row["member"])
            digest = hashlib.sha256(raw).hexdigest()
            if digest in seen_audio:
                rejected["duplicate_audio"] += 1
                continue
            filename = hashlib.sha256(row["member"].encode()).hexdigest()[:20] + ".wav"
            target = output / "raw_audio" / filename
            try:
                ffmpeg_wav(raw, target)
                duration, rms = wav_stats(target)
            except (ValueError, subprocess.TimeoutExpired) as exc:
                rejected["decode_error"] += 1
                target.unlink(missing_ok=True)
                continue
            if not 3 <= duration <= 15 or rms < 0.004:
                rejected["duration_or_silence"] += 1
                target.unlink(missing_ok=True)
                continue
            seen_text.add(key)
            seen_audio.add(digest)
            split_bucket = int(hashlib.sha256((row["member"] + "|split-v1").encode()).hexdigest()[:8], 16) % 10
            split = "train" if split_bucket < 8 else "validation" if split_bucket == 8 else "test"
            prepared.append({"file_name": filename, "text": row["text"].replace("|", ","), "speaker_id": speaker,
                             "language": SOURCES[source]["language"], "source": source, "source_member": row["member"],
                             "source_sha256": digest, "duration": round(duration, 3), "split": split})
        if not prepared:
            raise ValueError("No usable speech clips")
        counts = Counter(row["split"] for row in prepared)
        if min(counts.get(s, 0) for s in ("train", "validation", "test")) == 0:
            raise ValueError(f"Need train/validation/test samples; found {dict(counts)}. Increase --max-clips.")
        for split in ("train", "validation", "test"):
            with (output / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
                for row in prepared:
                    if row["split"] == split:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        with (output / "train_metadata.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="|")
            writer.writerow(["file_name", "text"])
            writer.writerows((r["file_name"], r["text"]) for r in prepared if r["split"] == "train")
        manifest = {"source": SOURCES[source], "source_id": source, "archive": str(archive_path.resolve()),
                    "archive_sha256": sha256_file(archive_path), "speaker_id": speaker, "counts": dict(counts),
                    "minutes": {s: round(sum(r["duration"] for r in prepared if r["split"] == s) / 60, 2) for s in counts},
                    "rejected": dict(rejected), "preparation": "24 kHz mono PCM; one verified archive speaker; deterministic utterance splits"}
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest
    finally:
        archive.close()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["speakers", "prepare"])
    ap.add_argument("--source", required=True, choices=SOURCES)
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--speaker")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--max-clips", type=int, default=300)
    args = ap.parse_args()
    if args.command == "speakers":
        archive, rows = read_archive(args.source, args.archive)
        archive.close()
        print(json.dumps(Counter(row["speaker"] for row in rows).most_common(), ensure_ascii=False))
    else:
        if not args.speaker or not args.output or args.max_clips < 30:
            ap.error("prepare requires --speaker, --output and --max-clips >= 30")
        print(json.dumps(build(args.source, args.archive, args.speaker, args.output, args.max_clips), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
