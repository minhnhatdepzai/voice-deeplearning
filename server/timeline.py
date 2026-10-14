"""Render a small multi-clip audio/video timeline with FFmpeg."""
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path


def run(*args: str, timeout: int = 1200) -> None:
    try:
        result = subprocess.run(args, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise ValueError("Xuất video/audio quá thời gian xử lý.") from exc
    if result.returncode:
        raise ValueError(result.stderr.decode(errors="replace")[-600:])


def probe(path: Path) -> dict:
    try:
        result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type",
                                 "-of", "json", str(path)], capture_output=True, check=True, timeout=30)
        info = json.loads(result.stdout)
        duration = float(info["format"]["duration"])
        streams = {stream["codec_type"] for stream in info.get("streams", [])}
    except (ValueError, KeyError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise ValueError("Không đọc được thời lượng hoặc định dạng media.") from exc
    if not 0 < duration <= 7200 or not streams.intersection({"video", "audio"}):
        raise ValueError("Tệp cần có audio/video và dài tối đa 2 giờ.")
    return {"duration": duration, "video": "video" in streams, "audio": "audio" in streams}


def _time(value, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} không hợp lệ.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} không hợp lệ.")
    return number


def render(files: list[Path], timeline: dict, output: Path, workspace: Path,
           overlay: Path | None = None) -> dict:
    if not 1 <= len(files) <= 12 or not isinstance(timeline, dict):
        raise ValueError("Cần 1–12 tệp media cho một lần xuất.")
    clips = timeline.get("clips")
    if not isinstance(clips, list) or not 1 <= len(clips) <= 40:
        raise ValueError("Timeline cần 1–40 đoạn.")
    metadata = [probe(path) for path in files]
    video_project = any(metadata[int(clip.get("source", -1))]["video"]
                        for clip in clips if isinstance(clip, dict) and isinstance(clip.get("source"), int)
                        and 0 <= clip["source"] < len(files))
    if any(not isinstance(clip, dict) or type(clip.get("source")) is not int or not 0 <= clip["source"] < len(files)
           for clip in clips):
        raise ValueError("Một đoạn timeline trỏ tới tệp không hợp lệ.")
    durations = []
    pieces = []
    for index, clip in enumerate(clips):
        source = clip["source"]
        info = metadata[source]
        start = _time(clip.get("start"), "Điểm bắt đầu")
        end = _time(clip.get("end"), "Điểm kết thúc")
        volume = _time(clip.get("volume", 1), "Âm lượng")
        if not 0 <= start < end <= info["duration"] + .08 or not .1 <= end - start <= 7200:
            raise ValueError(f"Đoạn {index + 1} có điểm cắt vượt ngoài tệp gốc.")
        if not 0 <= volume <= 2:
            raise ValueError("Âm lượng mỗi đoạn phải từ 0 đến 200%.")
        if clip.get("muted"):
            volume = 0
        length = round(end - start, 3)
        durations.append(length)
        piece = workspace / f"piece_{index:02}.{'mp4' if video_project else 'wav'}"
        pieces.append(piece)
        if video_project:
            if info["video"]:
                inputs = ["-ss", str(start), "-i", str(files[source])]
                video_map = "0:v:0"
                audio_index = 0 if info["audio"] else 1
                if not info["audio"]:
                    inputs += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
            else:
                inputs = ["-f", "lavfi", "-i", "color=c=0x19182d:s=1280x720:r=30",
                          "-ss", str(start), "-i", str(files[source])]
                video_map, audio_index = "0:v:0", 1
            run("ffmpeg", "-nostdin", "-y", "-v", "error", *inputs, "-t", str(length),
                "-map", video_map, "-map", f"{audio_index}:a:0",
                "-vf", "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,fps=30,format=yuv420p",
                "-af", f"volume={volume},apad,atrim=duration={length}",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-c:a", "aac", "-ar", "48000", "-ac", "2", str(piece))
        else:
            run("ffmpeg", "-nostdin", "-y", "-v", "error", "-ss", str(start), "-i", str(files[source]),
                "-t", str(length), "-vn", "-af", f"volume={volume},apad,atrim=duration={length}",
                "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2", str(piece))
    concat = workspace / "clips.txt"
    concat.write_text("".join(f"file '{piece}'\n" for piece in pieces), encoding="utf-8")
    joined = workspace / f"joined.{'mp4' if video_project else 'wav'}"
    run("ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(concat),
        "-c", "copy" if video_project else "pcm_s16le", str(joined))
    total = round(sum(durations), 3)
    if total > 7200:
        raise ValueError("Timeline dài tối đa 2 giờ.")
    output = output.with_suffix(".mp4" if video_project else ".wav")
    if overlay is None:
        joined.replace(output)
    else:
        overlay_info = probe(overlay)
        if not overlay_info["audio"]:
            raise ValueError("Lớp âm thanh thêm vào không có track audio.")
        at = _time(timeline.get("overlay_at", 0), "Vị trí lớp âm thanh")
        gain = _time(timeline.get("overlay_volume", 1), "Âm lượng lớp âm thanh")
        if not 0 <= at < total or not 0 <= gain <= 2:
            raise ValueError("Vị trí hoặc âm lượng lớp âm thanh không hợp lệ.")
        delay = round(at * 1000)
        graph = f"[1:a]volume={gain},adelay={delay}:all=1[voice];[0:a][voice]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,alimiter=limit=0.95[a]"
        args = ["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(joined), "-i", str(overlay),
                "-filter_complex", graph]
        if video_project:
            args += ["-map", "0:v:0", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-movflags", "+faststart"]
        else:
            args += ["-map", "[a]", "-c:a", "pcm_s16le"]
        run(*args, str(output))
    return {"duration": total, "format": "mp4" if video_project else "wav", "clips": len(clips),
            "overlay": overlay is not None, "output": str(output)}
