#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 BENCHMARK MODEL_PATH_OR_ID NUM_FEWSHOT" >&2
    echo "Example: $0 belebele Qwen/Qwen3.5-4B 5" >&2
}

if [[ $# -ne 3 || -z $1 || -z $2 || ! $3 =~ ^[0-9]+$ ]]; then
    usage
    exit 2
fi

TASK_LIST=$1
MODEL=${2%/}
NUM_FEWSHOT=$3
MODEL_NAME=${MODEL##*/}

if [[ -z $MODEL_NAME ]]; then
    usage
    exit 2
fi

# MODEL can be a local directory or a Hugging Face repository ID.

SEED=313
BATCH_SIZE=auto
export CUDA_VISIBLE_DEVICES=1
OUTPUT_DIR="./lm-evaluation-harness/eval_results"
COMMON_MODEL_ARGS="dtype=bfloat16"

mkdir -p "$OUTPUT_DIR"

if [[ $MODEL_NAME == *Ministral-3* ]]; then
    MODEL_TYPE=hf-mistral3
else
    MODEL_TYPE=hf
fi

echo "Model: $MODEL | Tasks: $TASK_LIST | Type: $MODEL_TYPE | Few-shot: $NUM_FEWSHOT"
#OUTPUT_PATH="${OUTPUT_DIR}/${MODEL_NAME}__${TASK_LIST}_${NUM_FEWSHOT}shot_seed${SEED}_modified.json"

accelerate launch -m lm_eval \
    --model "$MODEL_TYPE" \
    --model_args "pretrained=${MODEL},${COMMON_MODEL_ARGS}" \
    --tasks "$TASK_LIST" \
    --batch_size "$BATCH_SIZE" \
    --output_path "$OUTPUT_DIR" \
    --seed "$SEED" \
    --num_fewshot "$NUM_FEWSHOT"

echo "Evaluation completed."
