#!/usr/bin/env bash

# .env must be set
source .env

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OPEN_MODEL_NAMES=(
  # "meta-llama/Llama-3.1-8B-Instruct"
  # "meta-llama/Llama-3.1-70B-Instruct"
  # "Qwen/Qwen3-4B-Instruct-2507"
  # "Qwen/Qwen2.5-7B-Instruct"
  # "Qwen/Qwen3-32B"

  # "google/gemma-3-4b-it"
  
  # "google/gemma-3-270m-it"
  # "google/gemma-3-12b-it"
  # "google/gemma-3-27b-it"
)

CLOSED_OPENAI_MODELS=(
  "gpt-3.5-turbo"
  # "gpt-4o" 
  # "gpt-5.2"
)
CLOSED_ANTHROPIC_MODELS=(
  # "claude-sonnet-4.6"
)

for model_name in "${OPEN_MODEL_NAMES[@]}"; do
  python "${SCRIPT_DIR}/main_generation.py" \
    --generation-backend hf \
    --model-name "${model_name}" \
    --setting all \
    --output-dir "${SCRIPT_DIR}/outputs"
done

# Closed-model generation.
if [[ -n "${OPENAI_API_KEY:-}" ]]; then
  for model_name in "${CLOSED_OPENAI_MODELS[@]}"; do
    python "${SCRIPT_DIR}/main_generation.py" \
      --generation-backend openai \
      --model-name "${model_name}" \
      --setting all \
      --output-dir "${SCRIPT_DIR}/outputs"
  done
fi

if [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
  for model_name in "${CLOSED_ANTHROPIC_MODELS[@]}"; do
    python "${SCRIPT_DIR}/main_generation.py" \
      --generation-backend anthropic \
      --model-name "${model_name}" \
      --setting all \
      --output-dir "${SCRIPT_DIR}/outputs"
  done
fi
