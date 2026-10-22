#!/usr/bin/env bash
set -euo pipefail
cd /app
mkdir -p data/upstream data/huggingface
if [[ "${VOICE_STUDIO_BOOTSTRAP_MODE:-auto}" == auto ]]; then
  bootstrapped=0
  if [[ ! -f data/upstream/vieneu/.ready || ! -x data/upstream/vieneu/.venv/bin/python ]]; then
    bash training/setup.sh
    bootstrapped=1
  fi
  if [[ ! -f data/upstream/clearvoice/.ready || ! -x data/upstream/clearvoice/.venv/bin/python ]]; then
    bash training/setup_voice_filter.sh
    bootstrapped=1
  fi
  if [[ "$bootstrapped" == 1 ]]; then
    uv cache clean
  fi
fi
exec uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 1
