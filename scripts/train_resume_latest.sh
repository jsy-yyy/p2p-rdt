#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$ROOT_DIR/elefant/policy_model/train.py}"
CONFIG_FILE="${CONFIG_FILE:-$ROOT_DIR/config/policy_model/150M_lerobot_rdt.yaml}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,5,7}"
CKPT_DIR="${CKPT_DIR:-$ROOT_DIR/train_out/150M_lerobot_rdt_adjust_bottle_multiview_cond_new_wo_stat_init_from_pretrain_nofreeze/stage3_finetune}"
LOG_FILE="${LOG_FILE:-$ROOT_DIR/train_resume_latest.log}"

LATEST_CKPT="$(find "$CKPT_DIR" -maxdepth 1 -type f -name 'checkpoint-step=*.ckpt' | sort | tail -n 1)"
if [[ -z "${LATEST_CKPT}" ]]; then
  echo "No checkpoint found in CKPT_DIR=$CKPT_DIR" >&2
  exit 1
fi

{
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting resume training"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] CONFIG_FILE=$CONFIG_FILE"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] CKPT_DIR=$CKPT_DIR"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] LATEST_CKPT=$LATEST_CKPT"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] CUDA_VISIBLE_DEVICES=$CUDA_DEVICES"
} | tee -a "$LOG_FILE"

cd "$ROOT_DIR"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
"$PYTHON_BIN" "$TRAIN_SCRIPT" \
  --config "$CONFIG_FILE" \
  --resume_from_ckpt "$LATEST_CKPT" 2>&1 | tee -a "$LOG_FILE"
