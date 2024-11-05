#!/bin/bash
THIS_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT_DIR="$(dirname "$THIS_DIR")"
SDF_DIR="$PROJECT_ROOT_DIR/synthetic_discussion_framework/src"

INPUT_DIR="$PROJECT_ROOT_DIR/data/annotation_input/modular_configurations"
OUTPUT_DIR="$PROJECT_ROOT_DIR/data/annotation_input/generated"

echo "Removing old generated files..."
rm "$OUTPUT_DIR"/* # scary
mkdir -p "$OUTPUT_DIR"

python -u "$SDF_DIR/generate_annotation_configs.py" \
    --output_dir "$OUTPUT_DIR" \
    --persona_dir "$INPUT_DIR/personas" \
    --instruction_path "$INPUT_DIR/instructions/toxicity.txt" \
