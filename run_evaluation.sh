#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_NAME="${1:-}"

if [[ -z "${MODEL_NAME}" ]]; then
  echo "Usage: ${0} <model_name>"
  exit 1
fi

JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-google/gemma-3-12b-it}"
MODERATION_MAX_WORKERS="${MODERATION_MAX_WORKERS:-16}"

EXTRA_ARGS=()
if [[ "${SKIP_EXISTING_EVALS:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--skip-existing-evals)
fi

python "${SCRIPT_DIR}/main_evaluation.py" \
  --model-name "${MODEL_NAME}" \
  --setting all \
  --evaluator judge_model \
  --judge-model-name "${JUDGE_MODEL_NAME}" \
  "${EXTRA_ARGS[@]}" \
  --output-dir "${SCRIPT_DIR}/outputs"

if [[ -n "${OPENAI_API_KEY:-}" ]]; then
  python "${SCRIPT_DIR}/main_evaluation.py" \
    --model-name "${MODEL_NAME}" \
    --setting all \
    --evaluator openai_moderation_api \
    --openai-api-key "${OPENAI_API_KEY}" \
    --moderation-max-workers "${MODERATION_MAX_WORKERS}" \
    "${EXTRA_ARGS[@]}" \
    --output-dir "${SCRIPT_DIR}/outputs"
else
  echo "OPENAI_API_KEY is not set. Skipping openai_moderation_api evaluation."
fi
