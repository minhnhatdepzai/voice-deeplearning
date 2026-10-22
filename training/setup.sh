#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UPSTREAM="$ROOT/data/upstream/vieneu"
COMMIT="04044e1b8f822f946bafb2c88b716459f4164152"
if [ ! -d "$UPSTREAM/.git" ]; then
  mkdir -p "$(dirname "$UPSTREAM")"
  git clone https://github.com/pnnbao97/VieNeu-TTS.git "$UPSTREAM"
fi
git -C "$UPSTREAM" checkout --detach "$COMMIT"
cd "$UPSTREAM"
uv sync --extra finetune --no-dev --python 3.12
uv pip install --python "$UPSTREAM/.venv/bin/python" 'onnxruntime<1.24'
printf 'Training environment ready: %s\n' "$UPSTREAM"
