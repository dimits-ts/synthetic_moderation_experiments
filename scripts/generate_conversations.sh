#!/bin/bash

THIS_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT_DIR="$(dirname "$THIS_DIR")"
SDF_DIR="$PROJECT_ROOT_DIR/synthetic_discussion_framework/src"

INPUT_DIR="$PROJECT_ROOT_DIR/data/discussions_input/generated"
OUTPUT_DIR="$PROJECT_ROOT_DIR/data/discussions_output/collective_constitution"

MODEL_NAME="mistral-small" 
MODEL_PATH="byroneverson/Mistral-Small-Instruct-2409-abliterated"
LIBRARY_TYPE="transformers"  # library to load the model with, update as necessary

LOG_DIR="$PROJECT_ROOT_DIR/logs"
CURRENT_DATE=$(date +'%Y-%m-%d')

mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

bash "$THIS_DIR/__internal_conversation_execute_batch.sh" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model_path "$MODEL_PATH" \
    --max_tokens 1100 \
    --ctx_width_tokens 4048 \
    --inference_threads 10 \
    --gpu_layers 9999999 \
    --type "$LIBRARY_TYPE" \
    --model_name "$MODEL_NAME" \
    2>&1 | tee -a "$LOG_DIR/$CURRENT_DATE.txt"
