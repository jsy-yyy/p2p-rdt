#!/usr/bin/env bash
set -euo pipefail

# Current baked-in preset:
# - adjust_bottle nofreeze checkpoint family
# - legacy inference enabled by default for old-vs-new evaluation comparison
# - warm start and smoothing explicitly disabled via zero-valued flags

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_CONFIG="${1:-$ROOT_DIR/train_out/150M_lerobot_rdt_adjust_bottle_multiview_cond_new_wo_stat_init_from_pretrain/stage3_finetune/model_config.yaml}"
CHECKPOINT="${2:-$ROOT_DIR/train_out/150M_lerobot_rdt_adjust_bottle_multiview_cond_new_wo_stat_init_from_pretrain/stage3_finetune/checkpoint-step=00020000.ckpt}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-29056}"
DEVICE="${DEVICE:-}"
LEGACY_INFERENCE="${LEGACY_INFERENCE:-1}"
WARM_START_BLEND="${WARM_START_BLEND:-0}"
WARM_START_NOISE_STD="${WARM_START_NOISE_STD:-0}"
ACTION_SMOOTHING_ALPHA="${ACTION_SMOOTHING_ALPHA:-0}"
SKIP_WARMUP="${SKIP_WARMUP:-0}"

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
echo "Using LEGACY_INFERENCE=$LEGACY_INFERENCE WARM_START_BLEND=$WARM_START_BLEND WARM_START_NOISE_STD=$WARM_START_NOISE_STD ACTION_SMOOTHING_ALPHA=$ACTION_SMOOTHING_ALPHA SKIP_WARMUP=$SKIP_WARMUP" >&2

SERVER_ARGS=(
  --config "$MODEL_CONFIG"
  --checkpoint "$CHECKPOINT"
  --host "$HOST"
  --port "$PORT"
  --device "$DEVICE"
  --warm-start-blend "$WARM_START_BLEND"
  --warm-start-noise-std "$WARM_START_NOISE_STD"
  --action-smoothing-alpha "$ACTION_SMOOTHING_ALPHA"
  --no-compile
)

if [ "$SKIP_WARMUP" = "1" ]; then
  SERVER_ARGS+=(--skip-warmup)
fi

if [ "$LEGACY_INFERENCE" = "1" ]; then
  SERVER_ARGS+=(--legacy-inference)
fi

cd "$ROOT_DIR"
uv run python -m elefant.evaluation.robotwin.server "${SERVER_ARGS[@]}"
