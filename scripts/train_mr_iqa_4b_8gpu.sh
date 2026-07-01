#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_PATH="${MODEL_PATH:-}"
DATA_FILES="${DATA_FILES:-}"
IMAGE_ROOT="${IMAGE_ROOT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/mr-iqa-4b}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${REPO_ROOT}/src/mr_iqa/train_mr_iqa.py}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${REPO_ROOT}/configs/zero3-offload-auto.json}"
MASTER_PORT="${MASTER_PORT:-29661}"
VARIANCE_MODE="${VARIANCE_MODE:-unit}"
PROMPT_MODE="${PROMPT_MODE:-non_thinking}"
NUM_GPUS="${NUM_GPUS:-8}"
RUN_VALIDATION="${RUN_VALIDATION:-0}"

VAL_DATA_FILE="${VAL_DATA_FILE:-}"
VAL_IMAGE_ROOT="${VAL_IMAGE_ROOT:-${IMAGE_ROOT}}"
EVAL_DIR="${EVAL_DIR:-${OUTPUT_DIR}/validation}"
BEST_DIR="${BEST_DIR:-${OUTPUT_DIR}/best_model}"
FINAL_DIR="${FINAL_DIR:-${OUTPUT_DIR}/final_model}"
STATE_JSON="${STATE_JSON:-${EVAL_DIR}/best_state.json}"
VARIANCE_LOG_PATH="${VARIANCE_LOG_PATH:-${EVAL_DIR}/margin_variance_metrics.jsonl}"

MAX_SAMPLES="${MAX_SAMPLES:-}"
MAX_STEPS="${MAX_STEPS:--1}"
MAX_PIXELS="${MAX_PIXELS:-196608}"
MIN_PIXELS="${MIN_PIXELS:-3136}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-${PER_DEVICE_BATCH_SIZE:-48}}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
NUM_GENERATIONS="${NUM_GENERATIONS:-6}"
NUM_ITERATIONS="${NUM_ITERATIONS:-4}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-256}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-${TOTAL_EPOCHS:-10}}"
START_EPOCH="${START_EPOCH:-1}"
TEMPERATURE="${TEMPERATURE:-0.7}"
BETA="${BETA:-0.02}"
SAVE_STRATEGY="${SAVE_STRATEGY:-epoch}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
USE_LORA="${USE_LORA:-false}"
REPORT_TO="${REPORT_TO:-none}"
MIN_GT_STD="${MIN_GT_STD:-1e-4}"

if [[ -z "${CUDA_HOME:-}" ]]; then
  for cuda_dir in /usr/local/cuda-12.5 /usr/local/cuda; do
    if [[ -x "${cuda_dir}/bin/nvcc" ]]; then
      export CUDA_HOME="${cuda_dir}"
      break
    fi
  done
fi

if [[ -n "${CUDA_HOME:-}" ]]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export DS_SKIP_CUDA_CHECK="${DS_SKIP_CUDA_CHECK:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VARIANCE_LOG_PATH
export PROMPT_MODE

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 2
  fi
}

latest_checkpoint() {
  find "${OUTPUT_DIR}" -maxdepth 1 -type d -name "checkpoint-*" \
    | awk -F'checkpoint-' '{print $2 "\t" $0}' \
    | sort -n \
    | tail -1 \
    | cut -f2-
}

common_train_args() {
  local target_epochs="$1"
  shift
  TRAIN_ARGS=(
    --model_name_or_path "${MODEL_PATH}"
    --data_files "${DATA_FILES}"
    --image_root "${IMAGE_ROOT}"
    --output_dir "${OUTPUT_DIR}"
    --deepspeed "${DEEPSPEED_CONFIG}"
    --torch_dtype bfloat16
    --attn_implementation sdpa
    --max_pixels "${MAX_PIXELS}"
    --min_pixels "${MIN_PIXELS}"
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    --num_generations "${NUM_GENERATIONS}"
    --num_iterations "${NUM_ITERATIONS}"
    --max_completion_length "${MAX_COMPLETION_LENGTH}"
    --learning_rate "${LEARNING_RATE}"
    --num_train_epochs "${target_epochs}"
    --max_steps "${MAX_STEPS}"
    --temperature "${TEMPERATURE}"
    --beta "${BETA}"
    --seed "${SEED:-42}"
    --data_seed "${DATA_SEED:-42}"
    --dataset_seed "${DATASET_SEED:-42}"
    --reward_funcs margin,format
    --variance_mode "${VARIANCE_MODE}"
    --prompt_mode "${PROMPT_MODE}"
    --min_gt_std "${MIN_GT_STD}"
    --bf16 true
    --gradient_checkpointing true
    --ddp_find_unused_parameters false
    --save_strategy "${SAVE_STRATEGY}"
    --logging_steps 1
    --save_total_limit "${SAVE_TOTAL_LIMIT}"
    --report_to "${REPORT_TO}"
    --use_lora "${USE_LORA}"
    "$@"
  )
  if [[ -n "${MAX_SAMPLES}" ]]; then
    TRAIN_ARGS+=(--max_samples "${MAX_SAMPLES}")
  fi
}

run_train() {
  PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}" torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT}" \
    "${TRAIN_SCRIPT}" \
    "${TRAIN_ARGS[@]}"
}

run_validation() {
  local epoch="$1"
  local model_dir="$2"
  local out_json="${EVAL_DIR}/val_epoch_${epoch}.json"

  echo "[validation] epoch=${epoch} model=${model_dir}"
  MODEL_DIR="${model_dir}" \
  VAL_DATA_FILE="${VAL_DATA_FILE}" \
  IMAGE_ROOT="${VAL_IMAGE_ROOT}" \
  OUT_JSON="${out_json}" \
  NUM_GPUS="${NUM_GPUS}" \
  bash "${SCRIPT_DIR}/validation_eval_8gpu.sh"

  python3 - "${out_json}" "${model_dir}" "${epoch}" "${BEST_DIR}" "${STATE_JSON}" <<'MR_IQA_BEST_MODEL_PY'
import json
import math
import shutil
import sys
from pathlib import Path

out_json = Path(sys.argv[1])
model_dir = Path(sys.argv[2])
epoch = int(sys.argv[3])
best_dir = Path(sys.argv[4])
state_json = Path(sys.argv[5])

summary = json.loads(out_json.read_text())["summary"]
score = summary.get("srcc")
score = float(score) if score is not None and math.isfinite(float(score)) else float("-inf")
state = {"best_srcc": float("-inf"), "best_epoch": None, "best_model": None}
if state_json.exists():
    state = json.loads(state_json.read_text())

if score > float(state.get("best_srcc", float("-inf"))):
    best_dir.parent.mkdir(parents=True, exist_ok=True)
    if best_dir.exists():
        shutil.rmtree(best_dir)
    shutil.copytree(model_dir, best_dir)
    state = {
        "best_srcc": score,
        "best_epoch": epoch,
        "best_model": str(best_dir),
        "best_validation_json": str(out_json),
    }
    state_json.parent.mkdir(parents=True, exist_ok=True)
    state_json.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"new_best": True, **state, "summary": summary}, ensure_ascii=False, indent=2))
else:
    print(json.dumps({"new_best": False, "current_srcc": score, "state": state, "summary": summary}, ensure_ascii=False, indent=2))
MR_IQA_BEST_MODEL_PY
}

require_env MODEL_PATH
require_env DATA_FILES
require_env IMAGE_ROOT

mkdir -p "${OUTPUT_DIR}" "${EVAL_DIR}"

if [[ "${RUN_VALIDATION}" != "1" ]]; then
  echo "[train] mode=no_validation epochs=${NUM_TRAIN_EPOCHS}"
  common_train_args "${NUM_TRAIN_EPOCHS}"
  run_train
  exit 0
fi

require_env VAL_DATA_FILE
require_env VAL_IMAGE_ROOT

echo "[train] mode=with_validation start_epoch=${START_EPOCH} total_epochs=${NUM_TRAIN_EPOCHS}"
for epoch in $(seq "${START_EPOCH}" "${NUM_TRAIN_EPOCHS}"); do
  resume_args=()
  resume_checkpoint="$(latest_checkpoint || true)"
  if [[ -n "${resume_checkpoint}" ]]; then
    resume_args=(--resume_from_checkpoint "${resume_checkpoint}")
    echo "[train] resume=${resume_checkpoint}"
  fi

  echo "[train] target_epoch=${epoch}/${NUM_TRAIN_EPOCHS}"
  common_train_args "${epoch}" "${resume_args[@]}"
  run_train

  checkpoint="$(latest_checkpoint || true)"
  if [[ -z "${checkpoint}" ]]; then
    echo "[error] no checkpoint found after epoch ${epoch}" >&2
    exit 1
  fi
  run_validation "${epoch}" "${checkpoint}"
done

final_checkpoint="$(latest_checkpoint || true)"
if [[ -z "${final_checkpoint}" ]]; then
  echo "[error] no final checkpoint found" >&2
  exit 1
fi
rm -rf "${FINAL_DIR}"
cp -a "${final_checkpoint}" "${FINAL_DIR}"
echo "[done] final_model=${FINAL_DIR}"
echo "[done] best_state=$(cat "${STATE_JSON}" 2>/dev/null || true)"
