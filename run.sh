#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -x .venv/bin/python ]]; then
  if command -v uv >/dev/null 2>&1; then uv venv .venv; else python3 -m venv .venv; fi
fi
if command -v uv >/dev/null 2>&1; then
  uv pip install --python .venv/bin/python -r requirements.lock
else
  .venv/bin/python -m pip install -r requirements.lock
fi
if [[ ! -d node_modules ]]; then npm install; fi
npm run build
printf '\nVocalab đang mở tại http://127.0.0.1:8000\n'
exec .venv/bin/uvicorn server.app:app --host 127.0.0.1 --port 8000
