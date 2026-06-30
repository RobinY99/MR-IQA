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
NUM_GPUS="${NUM_GPUS:-8}"
RUN_VALIDATION="${RUN_VALIDATION:-0}"
VAL_DATA_FILE="${VAL_DATA_FILE:-}"
VAL_IMAGE_ROOT="${VAL_IMAGE_ROOT:-${IMAGE_ROOT}}"
VAL_OUTPUT_JSON="${VAL_OUTPUT_JSON:-${OUTPUT_DIR}/validation/final.json}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
MAX_STEPS="${MAX_STEPS:--1}"
MAX_PIXELS="${MAX_PIXELS:-196608}"
MIN_PIXELS="${MIN_PIXELS:-3136}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-48}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
NUM_GENERATIONS="${NUM_GENERATIONS:-6}"
NUM_ITERATIONS="${NUM_ITERATIONS:-4}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-256}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-10}"
TEMPERATURE="${TEMPERATURE:-0.7}"
BETA="${BETA:-0.02}"
SAVE_STRATEGY="${SAVE_STRATEGY:-epoch}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
USE_LORA="${USE_LORA:-false}"

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 2
  fi
}

require_env MODEL_PATH
require_env DATA_FILES
require_env IMAGE_ROOT
if [[ "${RUN_VALIDATION}" == "1" ]]; then
  require_env VAL_DATA_FILE
  require_env VAL_IMAGE_ROOT
fi

mkdir -p "${OUTPUT_DIR}"

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
  --num_train_epochs "${NUM_TRAIN_EPOCHS}"
  --max_steps "${MAX_STEPS}"
  --temperature "${TEMPERATURE}"
  --beta "${BETA}"
  --seed "${SEED:-42}"
  --data_seed "${DATA_SEED:-42}"
  --dataset_seed "${DATASET_SEED:-42}"
  --reward_funcs margin
  --variance_mode "${VARIANCE_MODE}"
  --min_gt_std 1e-4
  --bf16 true
  --gradient_checkpointing true
  --ddp_find_unused_parameters false
  --save_strategy "${SAVE_STRATEGY}"
  --logging_steps 1
  --save_total_limit "${SAVE_TOTAL_LIMIT}"
  --report_to "${REPORT_TO:-none}"
  --use_lora "${USE_LORA}"
)

if [[ -n "${MAX_SAMPLES}" ]]; then
  TRAIN_ARGS+=(--max_samples "${MAX_SAMPLES}")
fi

PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}" torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" "${TRAIN_SCRIPT}" "${TRAIN_ARGS[@]}"

if [[ "${RUN_VALIDATION}" == "1" ]]; then
  MODEL_DIR="${OUTPUT_DIR}" \
  VAL_DATA_FILE="${VAL_DATA_FILE}" \
  IMAGE_ROOT="${VAL_IMAGE_ROOT}" \
  OUT_JSON="${VAL_OUTPUT_JSON}" \
  NUM_GPUS="${NUM_GPUS}" \
  bash "${SCRIPT_DIR}/validation_eval_8gpu.sh"
fi
