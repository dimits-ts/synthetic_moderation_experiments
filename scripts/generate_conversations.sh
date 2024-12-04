#!/bin/bash

THIS_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT_DIR="$(dirname "$THIS_DIR")"
SDF_DIR="$PROJECT_ROOT_DIR/synthetic_discussion_framework/src"

INPUT_DIR="$PROJECT_ROOT_DIR/data/discussions_input/generated"
OUTPUT_DIR="$PROJECT_ROOT_DIR/data/discussions_output/collective_constitution"

MODEL_PATH="$PROJECT_ROOT_DIR/models/llama-3-8B-instruct.gguf"

LOG_DIR="$PROJECT_ROOT_DIR/logs"
CURRENT_DATE=$(date +'%Y-%m-%d')

mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

bash "$SDF_DIR/scripts/conversation_execute_all.sh" \
    --input_dir "$PROJECT_ROOT_DIR/data/discussions_input/generated" \
    --output_dir  "$OUTPUT_DIR" \
    --model_path "$MODEL_PATH" \
    --max_tokens 200 \
    --ctx_width_tokens 1024 \
    --inference_threads 10 \
    --gpu_layers 3  \
    2>&1 | tee -a "$LOG_DIR/$CURRENT_DATE.txt"