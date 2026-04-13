#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_NAME="${1:-adjust_bottle}"
TASK_CONFIG="${2:-demo_clean}"
TEST_NUM="${TEST_NUM:-10}"
SEED="${SEED:-0}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-29056}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/data/jsy/RoboTwin}"
ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON:-/home/zmz/miniconda3/envs/RoboTwin/bin/python}"
CKPT_TAG="${CKPT_TAG:-150M_lerobot_rdt_camera_high_wo_state}"
OPEN_P2P_ROBOTWIN_DISABLE_OIDN="${OPEN_P2P_ROBOTWIN_DISABLE_OIDN:-1}"
OPEN_P2P_ROBOTWIN_FORCE_RASTER="${OPEN_P2P_ROBOTWIN_FORCE_RASTER:-0}"
export OPEN_P2P_ROBOTWIN_DISABLE_OIDN OPEN_P2P_ROBOTWIN_FORCE_RASTER

if [ ! -x "$ROBOTWIN_PYTHON" ]; then
  echo "RoboTwin Python not found: $ROBOTWIN_PYTHON" >&2
  echo "Set ROBOTWIN_PYTHON to the interpreter inside your RoboTwin environment." >&2
  exit 1
fi

cd "$ROOT_DIR"
"$ROBOTWIN_PYTHON" -m elefant.evaluation.robotwin.eval_policy   --config "$ROOT_DIR/config/evaluation/robotwin_eval.yaml"   --overrides   --robowin_root "$ROBOTWIN_ROOT"   --task_name "$TASK_NAME"   --task_config "$TASK_CONFIG"   --test_num "$TEST_NUM"   --seed "$SEED"   --host "$HOST"   --port "$PORT"   --ckpt_setting "$CKPT_TAG"
