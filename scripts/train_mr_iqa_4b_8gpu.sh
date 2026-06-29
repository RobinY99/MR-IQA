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

PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}" torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" "${TRAIN_SCRIPT}" \
  --model_name_or_path "${MODEL_PATH}" \
  --data_files "${DATA_FILES}" \
  --image_root "${IMAGE_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --deepspeed "${DEEPSPEED_CONFIG}" \
  --torch_dtype bfloat16 \
  --attn_implementation sdpa \
  --max_pixels 196608 \
  --min_pixels 3136 \
  --per_device_train_batch_size 48 \
  --gradient_accumulation_steps 1 \
  --num_generations 6 \
  --num_iterations 4 \
  --max_completion_length 256 \
  --learning_rate 1e-5 \
  --num_train_epochs 10 \
  --temperature 0.7 \
  --beta 0.02 \
  --seed "${SEED:-42}" \
  --data_seed "${DATA_SEED:-42}" \
  --dataset_seed "${DATASET_SEED:-42}" \
  --reward_funcs margin \
  --variance_mode "${VARIANCE_MODE}" \
  --min_gt_std 1e-4 \
  --bf16 true \
  --gradient_checkpointing true \
  --ddp_find_unused_parameters false \
  --save_strategy epoch \
  --logging_steps 1 \
  --save_total_limit 2 \
  --report_to "${REPORT_TO:-none}" \
  --use_lora false

if [[ "${RUN_VALIDATION}" == "1" ]]; then
  MODEL_DIR="${OUTPUT_DIR}" \
  VAL_DATA_FILE="${VAL_DATA_FILE}" \
  IMAGE_ROOT="${VAL_IMAGE_ROOT}" \
  OUT_JSON="${VAL_OUTPUT_JSON}" \
  NUM_GPUS="${NUM_GPUS}" \
  bash "${SCRIPT_DIR}/validation_eval_8gpu.sh"
fi
