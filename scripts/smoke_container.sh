#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
image="${VOCALAB_SMOKE_IMAGE:-vocalab:smoke}"
docker build -t "$image" .
container=$(docker run --rm -d -e VOICE_STUDIO_BOOTSTRAP_MODE=skip -p 127.0.0.1::8000 "$image")
trap 'docker stop "$container" >/dev/null 2>&1 || true' EXIT
address=$(docker port "$container" 8000/tcp)
for _ in $(seq 1 30); do
  if curl -fsS "http://$address/api/health" >/dev/null 2>&1 && curl -fsS "http://$address/" >/dev/null 2>&1; then
    docker exec "$container" python -c 'import fastapi, faster_whisper, vieneu, onnxruntime'
    printf 'Container HTTP, UI and Python imports OK at http://%s\n' "$address"
    exit 0
  fi
  sleep 1
done
docker logs "$container"
exit 1
