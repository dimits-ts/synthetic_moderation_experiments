#!/bin/bash

set -uo pipefail


mod_models=(
    "unsloth/OLMo-2-0325-32B-Instruct-unsloth-bnb-4bit"
    "unsloth/Qwen2.5-32B-Instruct-bnb-4bit"
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    "unsloth/Olmo-3-7B-Instruct-unsloth-bnb-4bit"
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
)
mod_pseudos=(
    "olmo32b"
    "qwen32b"
    "llama8b"
    "olmo7b"
    "qwen7b"
)

user_models=(
    "unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit"
    "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
)
user_pseudos=(
    "gemma4b"
    "llama3b"
)

OUTPUT_DIR="./data/discussions_output/${name}"


for mod_strat_file in data/discussions_input/mod_instructions/*; do
    for mod_idx in "${!mod_models[@]}"; do
        for user_idx in "${!user_models[@]}"; do

            MOD_MODEL_URL="${mod_models[$mod_idx]}"
            MOD_MODEL_PSEUDO="${mod_pseudos[$mod_idx]}"

            USER_MODEL_URL="${user_models[$user_idx]}"
            USER_MODEL_PSEUDO="${user_pseudos[$user_idx]}"

            file_base=$(basename "$mod_strat_file" .yaml)
            name="${USER_MODEL_PSEUDO}_${MOD_MODEL_PSEUDO}_${file_base}"
            output_dir="$OUTPUT_DIR/${name}"

            if [[ -d "$output_dir" ]]; then
                echo "Skipping experiment (already exists): $output_dir"
                continue
            fi

            echo "Running: user=$USER_MODEL_PSEUDO mod=$MOD_MODEL_PSEUDO strategy=$file_base"

            python src/run_experiment.py \
                --config-file ./data/discussions_input/run_config.yml \
                --user-model-url "$USER_MODEL_URL" \
                --user-model-pseudo "$USER_MODEL_PSEUDO" \
                --mod-model-url "$MOD_MODEL_URL" \
                --mod-model-pseudo "$MOD_MODEL_PSEUDO" \
                --mod-strategy-file "$mod_strat_file" \
                --turn-manager "random-weighted" \
                --output-dir "$output_dir" \
                --user-persona-path "./data/discussions_input/personas/personas_employed.json" \
                --user-instruction-path "./data/discussions_input/user_instructions/vanilla.txt" \
                --mod-active \
                --num-experiments 20 \
                --trolls-active
        done
    done
done


# =====================================================
# 1. NO-MOD BASELINES
# =====================================================

for user_idx in "${!user_models[@]}"; do
    USER_MODEL_URL="${user_models[$user_idx]}"
    USER_MODEL_PSEUDO="${user_pseudos[$user_idx]}"

    name="${USER_MODEL_PSEUDO}_nomod"
    output_dir="${OUTPUT_DIR}/${name}"

    if [[ -d "$output_dir" ]]; then
        echo "Skipping (exists): $output_dir"
        continue
    fi

    echo "Running NO-MOD: user=$USER_MODEL_PSEUDO"

    python src/run_experiment.py \
        --config-file "$CONFIG" \
        --user-model-url "$USER_MODEL_URL" \
        --user-model-pseudo "$USER_MODEL_PSEUDO" \
        --mod-model-url "none" \
        --mod-model-pseudo "none" \
        --mod-strategy-file "data/discussions_input/mod_instructions/vanilla.txt" \
        --turn-manager "random-weighted" \
        --output-dir "$output_dir" \
        --user-persona-path "$PERSONAS" \
        --user-instruction-path "$USER_INSTR" \
        --num-experiments 20 \
        --trolls-active \
        --no-mod-active
done