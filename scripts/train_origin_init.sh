#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$ROOT_DIR/elefant/policy_model/train.py}"
BASE_CONFIG="${BASE_CONFIG:-$ROOT_DIR/config/policy_model/150M_lerobot_rdt.yaml}"
CONFIG_FILE="${CONFIG_FILE:-$BASE_CONFIG}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,5,7}"
RUN_NAME="${RUN_NAME:-$(basename "$CONFIG_FILE" .yaml)}"
LOG_FILE="${LOG_FILE:-$ROOT_DIR/${RUN_NAME}.log}"
ORIGIN_CKPT="${ORIGIN_CKPT:-}"

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

if [[ -z "$ORIGIN_CKPT" ]]; then
  echo "Please set ORIGIN_CKPT to the origin checkpoint path." >&2
  exit 1
fi

OUTPUT_PATH="$(read_config_output_path)"
mkdir -p "$OUTPUT_PATH"

{
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting origin-init training"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] BASE_CONFIG=$BASE_CONFIG"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] CONFIG_FILE=$CONFIG_FILE"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] OUTPUT_PATH=$OUTPUT_PATH"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ORIGIN_CKPT=$ORIGIN_CKPT"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] CUDA_VISIBLE_DEVICES=$CUDA_DEVICES"
} | tee -a "$LOG_FILE"

cd "$ROOT_DIR"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
"$PYTHON_BIN" "$TRAIN_SCRIPT" \
  --config "$CONFIG_FILE" \
  --init_from_origin_ckpt "$ORIGIN_CKPT" 2>&1 | tee -a "$LOG_FILE"
