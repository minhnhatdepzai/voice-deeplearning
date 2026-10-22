"""Find the speaker cluster with the most detected speech in a long recording."""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

from training.gpu_whisper import make_whisper
from training.uploaded_audio import transcribe_full, word_groups

RATE = 16000
MAX_SECONDS = 60 * 60
MAX_EMBEDDING_CLIPS = 600


def _decode(source: Path, target: Path) -> float:
    try:
        subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(source), "-vn", "-ac", "1",
             "-ar", str(RATE), "-c:a", "pcm_s16le", str(target)],
            check=True, capture_output=True, timeout=1200,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise ValueError("Không giải mã được audio/video để tìm người nói chính.") from exc
    with wave.open(str(target), "rb") as handle:
        duration = handle.getnframes() / handle.getframerate()
    if duration > MAX_SECONDS + 1:
        raise ValueError("Mỗi file tự tìm người nói được dài tối đa 60 phút.")
    return duration


def _normalized(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


def _spherical_kmeans(vectors: np.ndarray, weights: np.ndarray, clusters: int) -> tuple[np.ndarray, np.ndarray]:
    """Small deterministic cosine k-means, avoiding another runtime dependency."""
    centers = [vectors[int(np.argmax(weights))]]
    while len(centers) < clusters:
        similarity = np.max(vectors @ np.stack(centers).T, axis=1)
        centers.append(vectors[int(np.argmin(similarity))])
    centers = np.stack(centers)
    labels = np.zeros(len(vectors), dtype=np.int32)
    for _ in range(40):
        updated_labels = np.argmax(vectors @ centers.T, axis=1).astype(np.int32)
        if np.array_equal(updated_labels, labels) and _:
            break
        labels = updated_labels
        updated = []
        for cluster in range(clusters):
            members = labels == cluster
            if not np.any(members):
                similarity = np.max(vectors @ centers.T, axis=1)
                updated.append(vectors[int(np.argmin(similarity))])
            else:
                updated.append(_normalized(np.sum(vectors[members] * weights[members, None], axis=0)))
        centers = np.stack(updated)
    return labels, centers


def _silhouette(vectors: np.ndarray, labels: np.ndarray) -> float:
    similarity = np.clip(vectors @ vectors.T, -1, 1)
    distance = 1 - similarity
    scores = []
    for index, label in enumerate(labels):
        same = np.flatnonzero(labels == label)
        same = same[same != index]
        if not len(same):
            scores.append(0.0)
            continue
        own = float(np.mean(distance[index, same]))
        alternatives = [float(np.mean(distance[index, labels == other]))
                        for other in np.unique(labels) if other != label]
        nearest = min(alternatives)
        scores.append((nearest - own) / max(own, nearest, 1e-6))
    return float(np.mean(scores))


def cluster_embeddings(vectors: np.ndarray, durations: np.ndarray, max_clusters: int = 4) -> dict:
    """Estimate speaker clusters and return the duration-dominant cluster."""
    if len(vectors) < 2:
        return {"labels": np.zeros(len(vectors), dtype=np.int32), "centers": vectors.copy(),
                "clusters": 1, "silhouette": 0.0, "dominant": 0}
    best = None
    total = float(np.sum(durations))
    for count in range(2, min(max_clusters, len(vectors)) + 1):
        labels, centers = _spherical_kmeans(vectors, durations, count)
        cluster_seconds = np.array([np.sum(durations[labels == index]) for index in range(count)])
        if np.min(cluster_seconds) < max(6.0, total * .03):
            continue
        silhouette = _silhouette(vectors, labels)
        # A small complexity cost reduces splitting one voice by channel/noise changes.
        score = silhouette - .035 * (count - 2)
        if best is None or score > best[0]:
            best = (score, labels, centers, silhouette, cluster_seconds)
    if best is None or best[3] < .10:
        center = _normalized(np.sum(vectors * durations[:, None], axis=0))[None, :]
        return {"labels": np.zeros(len(vectors), dtype=np.int32), "centers": center,
                "clusters": 1, "silhouette": round(best[3], 4) if best else 0.0, "dominant": 0}
    _, labels, centers, silhouette, seconds = best
    # Speaker encoders can place two similar voices around .85-.90. Merge only
    # near-identical centers; otherwise an interview with two similar men can
    # collapse into one speaker.
    if len(centers) == 2 and float(centers[0] @ centers[1]) >= .93:
        center = _normalized(np.sum(vectors * durations[:, None], axis=0))[None, :]
        return {"labels": np.zeros(len(vectors), dtype=np.int32), "centers": center,
                "clusters": 1, "silhouette": round(float(silhouette), 4), "dominant": 0}
    dominant = int(np.argmax(seconds))
    return {"labels": labels, "centers": centers, "clusters": len(centers),
            "silhouette": round(float(silhouette), 4), "dominant": dominant}


def _sample_evenly(rows: list[dict], limit: int) -> list[dict]:
    if len(rows) <= limit:
        return rows
    indices = np.unique(np.linspace(0, len(rows) - 1, limit, dtype=int))
    return [rows[index] for index in indices]


def analyze(source: Path, workspace: Path, locale: str = "vi-VN") -> dict:
    """Create a clean reference candidate for the voice that speaks the longest."""
    source, workspace = source.resolve(), workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    decoded = workspace / "dominant_source.wav"
    duration = _decode(source, decoded)
    model, asr_device = make_whisper("small")
    language = "vi" if locale == "vi-VN" else "en"
    groups = word_groups(transcribe_full(model, decoded, language))
    rows = [{"start": max(0.0, start - .08), "end": min(duration, end + .08),
             "duration": min(duration, end + .08) - max(0.0, start - .08), "text": text}
            for start, end, text in groups if 2.5 <= end - start <= 12]
    if not rows:
        raise ValueError("Không tìm đủ đoạn lời nói rõ để tự xác định người nói nhiều nhất.")
    detected_speech = float(sum(row["duration"] for row in rows))
    sampled = _sample_evenly(rows, MAX_EMBEDDING_CLIPS)

    from vieneu._v3_turbo_engine.speaker import OnnxSpeakerEncoder
    encoder = OnnxSpeakerEncoder.from_pretrained("pnnbao-ump/VieNeu-TTS-v3-Turbo")
    accepted, vectors = [], []
    with wave.open(str(decoded), "rb") as wav:
        for row in sampled:
            start_frame = round(row["start"] * RATE)
            frames = max(1, round(row["duration"] * RATE))
            wav.setpos(min(start_frame, wav.getnframes()))
            raw = wav.readframes(frames)
            audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768
            if len(audio) < 2 * RATE:
                continue
            rms = float(np.sqrt(np.mean(audio * audio)))
            clipping = float(np.mean(np.abs(audio) >= .98))
            if rms < .002 or clipping > .05:
                continue
            vector = _normalized(encoder.embed(audio, RATE))
            if not np.any(vector):
                continue
            accepted.append({**row, "rms": rms, "clipping": clipping, "audio": audio})
            vectors.append(vector)
    if not accepted:
        raise ValueError("Các đoạn lời nói quá nhỏ hoặc quá nhiễu để nhận diện cụm giọng.")
    matrix = np.stack(vectors)
    durations = np.asarray([row["duration"] for row in accepted], dtype=np.float32)
    clustered = cluster_embeddings(matrix, durations)
    labels = clustered["labels"]
    dominant = clustered["dominant"]
    center = clustered["centers"][dominant]
    cluster_seconds = [float(np.sum(durations[labels == index])) for index in range(clustered["clusters"])]
    sampled_speech = max(sum(cluster_seconds), 1e-6)
    share = cluster_seconds[dominant] / sampled_speech

    candidates = [(index, row) for index, row in enumerate(accepted)
                  if labels[index] == dominant and 3 <= row["duration"] <= 8]
    if not candidates:
        candidates = [(index, row) for index, row in enumerate(accepted) if labels[index] == dominant]
    best_index, best = max(
        candidates,
        key=lambda pair: float(matrix[pair[0]] @ center) - pair[1]["clipping"] * 2
                         - abs(min(pair[1]["duration"], 8) - 5) * .005,
    )
    reference = workspace / "dominant_reference.wav"
    reference_length = min(8.0, max(3.2, best["duration"]))
    reference_start = max(0.0, min(best["start"], duration - reference_length))
    with wave.open(str(decoded), "rb") as source_wav:
        source_wav.setpos(round(reference_start * RATE))
        raw = source_wav.readframes(round(reference_length * RATE))
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768
    with wave.open(str(reference), "wb") as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(RATE)
        wav.writeframes(np.clip(audio * 32767, -32768, 32767).astype("<i2").tobytes())

    summaries = []
    for index, seconds in enumerate(cluster_seconds):
        summaries.append({"cluster": index + 1, "sampled_seconds": round(seconds, 1),
                          "speech_share": round(seconds / sampled_speech, 4),
                          "dominant": index == dominant})
    center_similarity_max = None
    if clustered["clusters"] > 1:
        center_matrix = clustered["centers"] @ clustered["centers"].T
        center_similarity_max = float(np.max(center_matrix - np.eye(len(center_matrix)) * 2))
    result = {
        "reference": str(reference), "source_duration": round(duration, 2),
        "detected_speech_seconds": round(detected_speech, 1),
        "analyzed_clips": len(accepted), "total_speech_clips": len(rows),
        "speaker_clusters_estimate": clustered["clusters"], "silhouette": clustered["silhouette"],
        "closest_cluster_similarity": round(center_similarity_max, 4) if center_similarity_max is not None else None,
        "dominant_speech_share": round(share, 4),
        "dominant_speech_seconds_estimate": round(detected_speech * share, 1),
        "reference_start": round(reference_start, 2),
        "reference_end": round(reference_start + len(audio) / RATE, 2),
        "reference_similarity": round(float(matrix[best_index] @ center), 4),
        "asr_device": asr_device, "clusters": summaries,
        "method": "Whisper speech segments + VieNeu speaker embeddings + cosine clustering",
    }
    (workspace / "dominant_speaker.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--locale", default="vi-VN", choices=("vi-VN", "en-US"))
    args = parser.parse_args()
    if args.output:
        result = analyze(args.source, args.output, args.locale)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        with tempfile.TemporaryDirectory(prefix="vocalab-dominant-") as directory:
            result = analyze(args.source, Path(directory), args.locale)
            result.pop("reference", None)
            print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
