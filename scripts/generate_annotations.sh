#!/bin/bash

THIS_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT_DIR="$(dirname "$THIS_DIR")"
SDF_DIR="$PROJECT_ROOT_DIR/synthetic_discussion_framework/src"

OUTPUT_DIR="$PROJECT_ROOT_DIR/data/annotations_output"
MODEL_PATH="$PROJECT_ROOT_DIR/models/llama-3-8B-instruct.gguf"
CONV_INPUT_DIR="$PROJECT_ROOT_DIR/data/discussions_output/collective_constitution"
ANNOTATOR_PROMPT_DIR="$PROJECT_ROOT_DIR/data/annotation_input/generated"

LOG_DIR="$PROJECT_ROOT_DIR/logs"
CURRENT_DATE=$(date +'%Y-%m-%d')
mkdir -p "$LOG_DIR"
mkdir -p "$ANNOTATOR_PROMPT_DIR"

for ANNOTATOR_PROMPT_PATH in "$ANNOTATOR_PROMPT_DIR"/*; do
    bash "__internal_annotation_execute_batch.sh" \
    --conv_input_dir "$CONV_INPUT_DIR" \
    --prompt_path "$ANNOTATOR_PROMPT_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --model_path "$MODEL_PATH" \
    2>&1 | tee -a "$LOG_DIR/$CURRENT_DATE.txt"
done

