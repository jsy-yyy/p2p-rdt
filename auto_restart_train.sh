#!/usr/bin/env bash
set -u
set -o pipefail

SESSION_NAME="policy_train"
LOG_FILE="train_multiview_noise_action_cond_wo_state_cond_new_with_state_cond.log"
CUDA_DEVICES="2,3,4,5,7"

PYTHON_BIN="./.venv/bin/python"
TRAIN_SCRIPT="elefant/policy_model/train.py"
CONFIG_FILE="config/policy_model/150M_lerobot_rdt.yaml"

CKPT_DIR="/data/jsy/open-p2p/train_out/150M_lerobot_rdt_multiview_noised_action_cond_new_with_stat_cond/stage3_finetune"

RESTART_WAIT=20

timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

log() {
    echo "[$(timestamp)] $*" | tee -a "$LOG_FILE"
}

find_latest_ckpt() {
    find "$CKPT_DIR" -maxdepth 1 -type f -name 'checkpoint-step=*.ckpt' | sort | tail -n 1
}

log "========== Auto-restart launcher started =========="
log "CKPT_DIR: $CKPT_DIR"

while true; do
    LATEST_CKPT="$(find_latest_ckpt)"

    CMD=(
        "$PYTHON_BIN" "$TRAIN_SCRIPT"
        --config "$CONFIG_FILE"
    )

    if [[ -n "${LATEST_CKPT}" && -f "${LATEST_CKPT}" ]]; then
        RESUME_CKPT="$LATEST_CKPT"
        log "Found latest checkpoint: $RESUME_CKPT"
        CMD+=(--resume_from_ckpt "$RESUME_CKPT")
    else
        log "No valid checkpoint found in CKPT_DIR, start training from scratch."
    fi

    log "Launching training..."
    log "Command: CUDA_VISIBLE_DEVICES=$CUDA_DEVICES ${CMD[*]}"

    CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
    "${CMD[@]}" >> "$LOG_FILE" 2>&1

    EXIT_CODE=$?
    log "Training process exited with code: $EXIT_CODE"

    if [[ $EXIT_CODE -eq 0 ]]; then
        log "Training exited normally. Stop restarting."
        break
    else
        log "Training crashed. Restart after ${RESTART_WAIT}s..."
        sleep "$RESTART_WAIT"
    fi
done

log "========== Auto-restart launcher stopped =========="