#!/bin/bash

set -uo pipefail


# =====================================================
# MODELS
# =====================================================
models=(
  "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit"
  "unsloth/Mistral-Small-24B-Instruct-2501-bnb-4bit"
  "unsloth/Qwen2.5-32B-Instruct-bnb-4bit"
  "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
  "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
  "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
)

pseudos=(
    "llama70b"
    "mistral24b"
    "qwen32b"
    "llama8b"
    "mistral7b"
    "qwen7b"
)

OUTPUT_DIR="./data/discussions_output/main"


for mod_strat_file in data/discussions_input/mod_instructions/*; do
    for mod_idx in "${!models[@]}"; do
        MODEL_URL="${models[$mod_idx]}"
        MODEL_PSEUDO="${pseudos[$mod_idx]}"

        file_base=$(basename "$mod_strat_file" .yaml)
        name="${MODEL_PSEUDO}_${file_base}"
        output_dir="$OUTPUT_DIR/${name}"

        echo "Running: model=$MODEL_PSEUDO strategy=$file_base"

        python src/run_experiment.py \
            --config-file ./data/discussions_input/run_config.yml \
            --model-url "$MODEL_URL" \
            --model-pseudo "$MODEL_PSEUDO" \
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


# =====================================================
# NO-MOD BASELINE
# =====================================================

for idx in "${!models[@]}"; do
    MODEL_URL="${models[$idx]}"
    MODEL_PSEUDO="${pseudos[$idx]}"

    name="${MODEL_PSEUDO}_nomod"
    output_dir="${OUTPUT_DIR}/${name}"

    echo "Running NO-MOD: model=$MODEL_PSEUDO"

    python src/run_experiment.py \
        --config-file ./data/discussions_input/run_config.yml \
        --model-url "$MODEL_URL" \
        --model-pseudo "$MODEL_PSEUDO" \
        --mod-strategy-file "data/discussions_input/mod_instructions/vanilla.txt" \
        --turn-manager "random-weighted" \
        --output-dir "$output_dir" \
        --user-persona-path "./data/discussions_input/personas/personas_employed.json" \
        --user-instruction-path "./data/discussions_input/user_instructions/vanilla.txt" \
        --num-experiments 25 \
        --trolls-active \
        --no-mod-active
done


# =====================================================
# OPTIMAL BASELINE
# =====================================================

for mod_strat_file in data/discussions_input/mod_instructions/*; do
    for mod_idx in "${!models[@]}"; do
        MODEL_URL="${models[$mod_idx]}"
        MODEL_PSEUDO="${pseudos[$mod_idx]}"

        file_base=$(basename "$mod_strat_file" .yaml)
        name="${MODEL_PSEUDO}_${file_base}"
        output_dir="$OUTPUT_DIR/${name}"

        echo "Running: model=$MODEL_PSEUDO strategy=$file_base"

        python src/run_experiment.py \
            --config-file ./data/discussions_input/run_config.yml \
            --model-url "$MODEL_URL" \
            --model-pseudo "$MODEL_PSEUDO" \
            --mod-strategy-file "$mod_strat_file" \
            --turn-manager "random-weighted" \
            --output-dir "$output_dir" \
            --user-persona-path "./data/discussions_input/personas/personas_employed.json" \
            --user-instruction-path "./data/discussions_input/user_instructions/vanilla.txt" \
            --mod-active \
            --num-experiments 50 \
            --trolls-active
    done
done