#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$ROOT_DIR/elefant/policy_model/train.py}"
BASE_CONFIG="${BASE_CONFIG:-$ROOT_DIR/config/policy_model/150M_lerobot_rdt.yaml}"
CONFIG_FILE="${CONFIG_FILE:-$BASE_CONFIG}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,4,5,7}"

read_config_output_path() {
  "$PYTHON_BIN" - <<'PY' "$CONFIG_FILE"
from pathlib import Path
import sys
import yaml

config_path = Path(sys.argv[1])
config = yaml.safe_load(config_path.read_text()) or {}
output_path = config.get("shared", {}).get("output_path")
if not output_path:
    raise SystemExit(f"Config {config_path} is missing shared.output_path")
print(output_path)
PY
}

find_resume_ckpt() {
  "$PYTHON_BIN" - <<'PY' "$CKPT_DIR"
from pathlib import Path
import sys

import torch

ckpt_dir = Path(sys.argv[1])
for path in sorted(ckpt_dir.glob('checkpoint-step=*.ckpt'), reverse=True):
    try:
        if path.stat().st_size <= 0:
            continue
        torch.load(path, map_location='cpu', weights_only=False)
    except Exception:
        continue
    print(path)
    break
PY
}

OUTPUT_PATH="$(read_config_output_path)"
CKPT_DIR="$OUTPUT_PATH/stage3_finetune"
RUN_NAME="${RUN_NAME:-$(basename "$OUTPUT_PATH")}"
LOG_FILE="${LOG_FILE:-$ROOT_DIR/${RUN_NAME}.log}"

LATEST_CKPT="$(find_resume_ckpt)"
if [[ -z "${LATEST_CKPT}" ]]; then
  echo "No usable checkpoint found in CKPT_DIR=$CKPT_DIR" >&2
  exit 1
fi

{
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting resume training"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] BASE_CONFIG=$BASE_CONFIG"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] CONFIG_FILE=$CONFIG_FILE"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] OUTPUT_PATH=$OUTPUT_PATH"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] CKPT_DIR=$CKPT_DIR"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] LATEST_CKPT=$LATEST_CKPT"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] CUDA_VISIBLE_DEVICES=$CUDA_DEVICES"
} | tee -a "$LOG_FILE"

cd "$ROOT_DIR"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
"$PYTHON_BIN" "$TRAIN_SCRIPT" \
  --config "$CONFIG_FILE" \
  --resume_from_ckpt "$LATEST_CKPT" 2>&1 | tee -a "$LOG_FILE"
