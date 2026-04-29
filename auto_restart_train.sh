#!/usr/bin/env bash
set -u
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INIT_MODE="${INIT_MODE:-${TRAIN_MODE:-random_init}}"
ORIGIN_CKPT="${ORIGIN_CKPT:-/data/jsy/open-p2p/checkpoints/150M}"

DEFAULT_CONFIG="$ROOT_DIR/config/policy_model/150M_lerobot_rdt.yaml"
CONFIG_SOURCE="${BASE_CONFIG:-${CONFIG_FILE:-$DEFAULT_CONFIG}}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$ROOT_DIR/elefant/policy_model/train.py}"
BASE_CONFIG="$CONFIG_SOURCE"
CONFIG_FILE="$CONFIG_SOURCE"

CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2,3,4,5,7}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/train_out}"
RESTART_WAIT="${RESTART_WAIT:-20}"
INIT_SCRIPT="${INIT_SCRIPT:-}"
RESUME_SCRIPT="${RESUME_SCRIPT:-$ROOT_DIR/scripts/train_resume_latest.sh}"

resolve_script() {
  local script_path="$1"

  if [[ "$script_path" = /* ]]; then
    printf '%s\n' "$script_path"
  else
    printf '%s\n' "$ROOT_DIR/$script_path"
  fi
}

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

find_latest_ckpt() {
  if [[ ! -d "$CKPT_DIR" ]]; then
    return 0
  fi

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

default_init_script() {
  case "$INIT_MODE" in
    origin|origin_init)
      printf '%s\n' "$ROOT_DIR/scripts/train_origin_init.sh"
      ;;
    random|random_init)
      printf '%s\n' "$ROOT_DIR/scripts/train_random_init.sh"
      ;;
    *)
      printf '%s\n' ''
      ;;
  esac
}

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

CONFIG_OUTPUT_PATH="$(read_config_output_path)"
OUTPUT_PATH="$CONFIG_OUTPUT_PATH"
CKPT_DIR="$OUTPUT_PATH/stage3_finetune"
RUN_NAME="${RUN_NAME:-$(basename "$OUTPUT_PATH")}"
DEFAULT_LOG_FILE="$ROOT_DIR/${RUN_NAME}.log"
LOG_FILE="${LOG_FILE:-}"

if [[ -z "$LOG_FILE" ]]; then
  if [[ -t 1 ]]; then
    LOG_FILE="$DEFAULT_LOG_FILE"
  else
    LOG_FILE="/dev/null"
  fi
fi

log() {
  echo "[$(timestamp)] $*" | tee -a "$LOG_FILE"
}

validate_script() {
  local label="$1"
  local script_path="$2"

  if [[ ! -f "$script_path" ]]; then
    log "$label script not found: $script_path"
    exit 1
  fi
}

if [[ -z "$INIT_SCRIPT" ]]; then
  INIT_SCRIPT="$(default_init_script)"
fi

if [[ -z "$INIT_SCRIPT" ]]; then
  log "Unsupported INIT_MODE=$INIT_MODE. Use origin_init or random_init."
  exit 1
fi

INIT_SCRIPT="$(resolve_script "$INIT_SCRIPT")"
RESUME_SCRIPT="$(resolve_script "$RESUME_SCRIPT")"

validate_script "Init" "$INIT_SCRIPT"
validate_script "Resume" "$RESUME_SCRIPT"

if [[ "$(basename "$INIT_SCRIPT")" == "train_origin_init.sh" && -z "$ORIGIN_CKPT" ]]; then
  log "INIT_SCRIPT=train_origin_init.sh requires ORIGIN_CKPT to be set."
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_PATH"

log "========== Auto-restart launcher started =========="
log "INIT_MODE: $INIT_MODE"
log "INIT_SCRIPT: $INIT_SCRIPT"
log "RESUME_SCRIPT: $RESUME_SCRIPT"
log "BASE_CONFIG: $BASE_CONFIG"
log "CONFIG_FILE: $CONFIG_FILE"
log "RUN_NAME: $RUN_NAME"
log "CONFIG_OUTPUT_PATH: $CONFIG_OUTPUT_PATH"
log "OUTPUT_PATH: $OUTPUT_PATH"
log "CKPT_DIR: $CKPT_DIR"
log "LOG_FILE: $LOG_FILE"

while true; do
  LATEST_CKPT="$(find_latest_ckpt)"

  if [[ -n "$LATEST_CKPT" && -f "$LATEST_CKPT" ]]; then
    SCRIPT_TO_RUN="$RESUME_SCRIPT"
    RUN_PHASE="resume"
    log "Found latest usable checkpoint: $LATEST_CKPT"
  else
    SCRIPT_TO_RUN="$INIT_SCRIPT"
    RUN_PHASE="init"
    log "No usable checkpoint found in CKPT_DIR. Initializing with mode $INIT_MODE."
  fi

  log "Launching $RUN_PHASE training via $(basename "$SCRIPT_TO_RUN")"
  log "Settings: CUDA_VISIBLE_DEVICES=$CUDA_DEVICES RUN_NAME=$RUN_NAME"
  log "Settings: OUTPUT_ROOT=$OUTPUT_ROOT OUTPUT_PATH=$OUTPUT_PATH"

  if [[ -n "$ORIGIN_CKPT" ]]; then
    log "Settings: ORIGIN_CKPT=$ORIGIN_CKPT"
  fi

  (
    cd "$ROOT_DIR" || exit 1
    PYTHON_BIN="$PYTHON_BIN" \
    TRAIN_SCRIPT="$TRAIN_SCRIPT" \
    BASE_CONFIG="$BASE_CONFIG" \
    CONFIG_FILE="$CONFIG_FILE" \
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
    RUN_NAME="$RUN_NAME" \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    OUTPUT_PATH="$OUTPUT_PATH" \
    CKPT_DIR="$CKPT_DIR" \
    LOG_FILE="$LOG_FILE" \
    ORIGIN_CKPT="$ORIGIN_CKPT" \
    bash "$SCRIPT_TO_RUN"
  )
  EXIT_CODE=$?
  log "Training process exited with code: $EXIT_CODE"

  if [[ $EXIT_CODE -eq 0 ]]; then
    log "Training exited normally. Stop restarting."
    break
  fi

  log "Training crashed. Restart after ${RESTART_WAIT}s..."
  sleep "$RESTART_WAIT"
done

log "========== Auto-restart launcher stopped =========="
