#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$ROOT_DIR/elefant/policy_model/train.py}"
BASE_CONFIG="${BASE_CONFIG:-$ROOT_DIR/config/policy_model/150M_lerobot_rdt.yaml}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,5,7}"
RUN_NAME="${RUN_NAME:-150M_lerobot_rdt_random_init}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/train_out}"
OUTPUT_PATH="${OUTPUT_PATH:-$OUTPUT_ROOT/$RUN_NAME}"
LOG_FILE="${LOG_FILE:-$ROOT_DIR/${RUN_NAME}.log}"
TEMP_CONFIG="$(mktemp "${TMPDIR:-/tmp}/open-p2p-random-init-XXXXXX.yaml")"

cleanup() {
  rm -f "$TEMP_CONFIG"
}
trap cleanup EXIT

mkdir -p "$OUTPUT_ROOT"

"$PYTHON_BIN" - <<'PY' "$BASE_CONFIG" "$TEMP_CONFIG" "$OUTPUT_PATH"
from pathlib import Path
import sys
import yaml

base_config, temp_config, output_path = sys.argv[1:4]
config = yaml.safe_load(Path(base_config).read_text())
config["shared"]["output_path"] = output_path
Path(temp_config).write_text(yaml.safe_dump(config, sort_keys=False))
PY

{
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting random-init training"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] BASE_CONFIG=$BASE_CONFIG"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] OUTPUT_PATH=$OUTPUT_PATH"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] CUDA_VISIBLE_DEVICES=$CUDA_DEVICES"
} | tee -a "$LOG_FILE"

cd "$ROOT_DIR"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
"$PYTHON_BIN" "$TRAIN_SCRIPT" --config "$TEMP_CONFIG" 2>&1 | tee -a "$LOG_FILE"
