#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${1:-"models/pretrain/encoder/final"}
LR=${2:-"3e-5"}
BATCH_SIZE=${3:-"64"}
MAX_EPOCHS=${4:-"5"}
WSC_EPOCHS=${5:-"5"}
SEED=${6:-"42"}
SEQ_LEN=${7:-"128"}
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if [[ "$MODEL_PATH" != /* && "$MODEL_PATH" != ?:* ]]; then
    MODEL_PATH="${ROOT_DIR}/${MODEL_PATH#./}"
fi

cd "${ROOT_DIR}/eval-pipeline"
bash eval_finetuning.sh "$MODEL_PATH" mlm "$LR" "$BATCH_SIZE" "$MAX_EPOCHS" "$WSC_EPOCHS" "$SEED" "$SEQ_LEN"
