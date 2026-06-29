#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_DIR="${MODEL_DIR:-}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/test_manifests}"
IMAGE_ROOT="${IMAGE_ROOT:-}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/outputs/generalization}"
EVAL_SCRIPT="${EVAL_SCRIPT:-${REPO_ROOT}/src/mr_iqa/evaluate_mr_iqa.py}"
NUM_GPUS="${NUM_GPUS:-8}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 2
  fi
}

require_env MODEL_DIR
require_env IMAGE_ROOT

mkdir -p "${OUT_DIR}/shards" "${OUT_DIR}/results"

read -r -a datasets <<< "${DATASETS:-agiqa3k csiq kadid_full koniq livew pipal spaq_full tid2013}"

for dataset in "${datasets[@]}"; do
  data_file="${DATA_DIR}/${dataset}.json"
  shard_dir="${OUT_DIR}/shards/${dataset}"
  mkdir -p "${shard_dir}"
  pids=()
  for shard_id in $(seq 0 $((NUM_GPUS - 1))); do
    CUDA_VISIBLE_DEVICES="${shard_id}" PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}" "${PYTHON_BIN}" "${EVAL_SCRIPT}" \
      --model_name_or_path "${MODEL_DIR}" \
      --data_file "${data_file}" \
      --image_root "${IMAGE_ROOT}" \
      --output_json "${shard_dir}/shard_${shard_id}.json" \
      --num_shards "${NUM_GPUS}" \
      --shard_id "${shard_id}" \
      --max_new_tokens 64 \
      --temperature 0.0 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
  "${PYTHON_BIN}" - "${dataset}" "${shard_dir}" "${OUT_DIR}/results/${dataset}.json" "${NUM_GPUS}" <<'MR_IQA_MERGE_DATASET_PY'
import json, math, sys
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

dataset, shard_dir, out_json, num_shards = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
all_results = []
missing = 0
raw_out_of_range = 0
for shard_id in range(num_shards):
    payload = json.loads((shard_dir / f"shard_{shard_id}.json").read_text())
    summary = payload["summary"]
    missing += int(summary.get("num_missing_or_bad_gold") or 0)
    raw_out_of_range += int(summary.get("raw_out_of_range") or 0)
    for row in payload.get("results", []):
        row["eval_shard_id"] = shard_id
        all_results.append(row)
valid = [x for x in all_results if x.get("pred_score") is not None]
if len(valid) < 2 or len({x["pred_score"] for x in valid}) < 2:
    plcc = math.nan
    srcc = math.nan
else:
    gold = [x["gold_score"] for x in valid]
    pred = [x["pred_score"] for x in valid]
    plcc = float(pearsonr(gold, pred).statistic)
    srcc = float(spearmanr(gold, pred).statistic)
summary = {
    "dataset": dataset,
    "num_total": len(all_results),
    "num_valid": len(valid),
    "num_missing_or_bad_gold": missing,
    "raw_out_of_range": raw_out_of_range,
    "plcc": plcc,
    "srcc": srcc,
    "num_shards": num_shards,
    "validation_num_gpus": num_shards,
}
out_json.write_text(json.dumps({"summary": summary, "results": all_results}, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(summary, ensure_ascii=False, indent=2))
MR_IQA_MERGE_DATASET_PY
done

"${PYTHON_BIN}" - "${OUT_DIR}/results" "${OUT_DIR}/summary.json" <<'MR_IQA_SUMMARY_PY'
import json, sys
from pathlib import Path
result_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
rows = []
for path in sorted(result_dir.glob("*.json")):
    rows.append(json.loads(path.read_text())["summary"])
out.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(rows, ensure_ascii=False, indent=2))
MR_IQA_SUMMARY_PY
