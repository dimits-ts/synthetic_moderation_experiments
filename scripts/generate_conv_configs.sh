#!/bin/bash
THIS_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT_DIR="$(dirname "$THIS_DIR")"
SDF_DIR="$PROJECT_ROOT_DIR/synthetic_discussion_framework/src"

INPUT_DIR="$PROJECT_ROOT_DIR/data/generated_discussions_input/modular_configurations"
OUTPUT_DIR="$PROJECT_ROOT_DIR/data/generated_discussions_input/conv_data/generated"

echo "Removing old generated files..."
mkdir -p "$OUTPUT_DIR"
rm "$OUTPUT_DIR"/* # scary

python -u "$SDF_DIR/scripts/generate_conv_configs.py" \
          --output_dir  "$OUTPUT_DIR"\
          --persona_dir "$INPUT_DIR/personas" \
          --topics_dir "$INPUT_DIR/topics" \
          --configs_path "$INPUT_DIR/other_configs/standard_multi_user.json" \
          --user_instruction_path "$INPUT_DIR/user_instructions/vanilla.txt" \
          --mod_instruction_path "$INPUT_DIR/mod_instructions/no_instructions.txt" \
          --num_generated_files 30 \
          --num_users 5 \
          --include_mod