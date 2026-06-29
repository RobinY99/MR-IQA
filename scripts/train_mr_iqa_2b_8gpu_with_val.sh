#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_VALIDATION=1 bash "${SCRIPT_DIR}/train_mr_iqa_2b_8gpu.sh"
