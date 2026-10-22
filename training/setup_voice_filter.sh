#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v uv >/dev/null 2>&1; then
  echo 'Cần cài uv để tạo môi trường ClearVoice riêng.' >&2
  exit 1
fi
uv venv --python 3.12 data/upstream/clearvoice/.venv
uv pip install --python data/upstream/clearvoice/.venv/bin/python 'clearvoice==0.1.2'
echo 'ClearVoice đã cài. Hai checkpoint MossFormer2_SS_16K và FRCRN_SE_16K sẽ tải ở lượt xử lý đầu tiên.'
