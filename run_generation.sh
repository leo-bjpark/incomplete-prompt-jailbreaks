#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_NAMES=(
  # "meta-llama/Llama-3.1-8B-Instruct"
  "Qwen/Qwen3-4B-Instruct-2507"
  # "google/gemma-3-4b-it"
)

for model_name in "${MODEL_NAMES[@]}"; do
  python "${SCRIPT_DIR}/main_generation.py" \
    --model-name "${model_name}" \
    --setting all \
    --output-dir "${SCRIPT_DIR}/outputs"
done
