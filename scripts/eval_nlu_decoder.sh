#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${1:-"models/pretrain/decoder/final"}
EVAL_DIR=${2:-"evaluation_data/full_eval"}
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if [[ "$MODEL_PATH" != /* && "$MODEL_PATH" != ?:* ]]; then
    MODEL_PATH="${ROOT_DIR}/${MODEL_PATH#./}"
fi

cd "${ROOT_DIR}/eval-pipeline"
python -m evaluation_pipeline.sentence_zero_shot.run \
    --model_path_or_name "$MODEL_PATH" \
    --backend causal \
    --task zhoblimp \
    --data_path "${EVAL_DIR}/zhoblimp" \
    --save_predictions
