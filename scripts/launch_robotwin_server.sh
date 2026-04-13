#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_CONFIG="${1:-$ROOT_DIR/train_out/150M_lerobot_rdt_camera_high_wo_state/model_config.yaml}"
CHECKPOINT="${2:-$ROOT_DIR/train_out/150M_lerobot_rdt_camera_high_wo_state/checkpoint-step=00010000.ckpt}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-29056}"
DEVICE="${DEVICE:-}"

if [ -z "$DEVICE" ]; then
  DEVICE="$({
    uv run python -c 'import torch
if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
    print("cpu")
    raise SystemExit(0)
best_idx = 0
best_free = -1
for idx in range(torch.cuda.device_count()):
    try:
        free_bytes, _ = torch.cuda.mem_get_info(idx)
    except Exception:
        continue
    if free_bytes > best_free:
        best_free = free_bytes
        best_idx = idx
print(f"cuda:{best_idx}")'
  })"
fi
DEVICE="${DEVICE:-cuda:0}"

echo "Using DEVICE=$DEVICE" >&2

cd "$ROOT_DIR"
uv run python -m elefant.evaluation.robotwin.server \
  --config "$MODEL_CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --host "$HOST" \
  --port "$PORT" \
  --device "$DEVICE" \
  --no-compile
