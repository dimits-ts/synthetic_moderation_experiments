#!/bin/bash

THIS_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT_DIR="$(dirname "$THIS_DIR")"
SDF_DIR="$PROJECT_ROOT_DIR/synthetic_discussion_framework/src"

INPUT_DIR="$PROJECT_ROOT_DIR/data/generated_discussions_input/conv_data/generated"
OUTPUT_DIR="$PROJECT_ROOT_DIR/data/generated_discussions_output/new"

MODEL_PATH="$PROJECT_ROOT_DIR/models/llama-3-8B-instruct.gguf"

LOG_DIR="$PROJECT_ROOT_DIR/logs"
CURRENT_DATE=$(date +'%Y-%m-%d')
OUTPUT_DIR="$PROJECT_ROOT_DIR/data/generated_discussions_output/arguments"

mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

bash "$SDF_DIR/scripts/conversation_execute_all.sh" \
    --input_dir "$PROJECT_ROOT_DIR/data/generated_discussions_input/conv_data/generated" \
    --output_dir  "$OUTPUT_DIR" \
    --model_path "$MODEL_PATH" \
    --max_tokens 500 \
    --ctx_width_tokens 2048 \
    --inference_threads 10 \
    --gpu_layers 10  \
    2>&1 | tee -a "$LOG_DIR/$CURRENT_DATE.txt"