#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v uv >/dev/null 2>&1; then
  echo 'Cần cài uv để tạo môi trường ClearVoice riêng.' >&2
  exit 1
fi
uv venv --clear --python 3.12 data/upstream/clearvoice/.venv
python=data/upstream/clearvoice/.venv/bin/python
uv pip install --python "$python" --index-url https://download.pytorch.org/whl/cu128 \
  'torch==2.8.0' 'torchvision==0.23.0' 'torchaudio==2.8.0'
uv pip install --python "$python" --constraint training/clearvoice-constraints.txt 'clearvoice==0.1.2'
"$python" -c 'import torch, clearvoice; assert torch.version.cuda == "12.8"'
touch data/upstream/clearvoice/.ready
echo 'ClearVoice đã cài. Hai checkpoint MossFormer2_SS_16K và FRCRN_SE_16K sẽ tải ở lượt xử lý đầu tiên.'
