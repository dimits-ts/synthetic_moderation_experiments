#!/bin/bash
THIS_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT_DIR="$(dirname "$THIS_DIR")"
SDF_DIR="$PROJECT_ROOT_DIR/synthetic_discussion_framework"

INPUT_DIR="$PROJECT_ROOT_DIR/data/discussions_input/modular_configurations"
OUTPUT_DIR="$PROJECT_ROOT_DIR/data/discussions_input/generated"

echo "Removing old generated files..."
mkdir -p "$OUTPUT_DIR"
rm -rf "$OUTPUT_DIR"/* # scary

python -u "$SDF_DIR/src/generate_conv_configs.py" \
          --output_dir  "$OUTPUT_DIR"\
          --persona_dir "$INPUT_DIR/personas" \
          --topics_dir "$INPUT_DIR/topics" \
          --configs_path "$INPUT_DIR/other_configs/standard_multi_user.json" \
          --user_instruction_path "$INPUT_DIR/user_instructions/vanilla.txt" \
          --mod_instruction_path "$INPUT_DIR/mod_instructions/collective_constitution.txt" \
          --num_generated_files 30 \
          --num_users 5 \
          --include_mod