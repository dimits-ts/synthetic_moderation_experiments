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

# =====================================================
# PATHS
# =====================================================
CONFIG="./data/discussions_input/run_config.yml"

PERSONAS="./data/discussions_input/personas/personas_employed.json"
NO_SDBS_PERSONAS="./data/discussions_input/personas/no_sdbs.json"

USER_INSTR="./data/discussions_input/user_instructions/vanilla.txt"
NO_USER_INSTR="./data/discussions_input/user_instructions/no_instructions.txt"

OUTPUT_DIR="./data/discussions_output/ablations"

# =====================================================
# ABLATIONS
# name | turn-manager | persona-path | user-instr-path | trolls-flag
# =====================================================
ablations=(
  "noinstructions|random-weighted|$PERSONAS|$NO_USER_INSTR|--trolls-active"
  "notroll|random-weighted|$PERSONAS|$USER_INSTR|--no-trolls-active"
  "nosdbs|random-weighted|$NO_SDBS_PERSONAS|$USER_INSTR|--trolls-active"
  "roundrobin|round-robin|$PERSONAS|$USER_INSTR|--trolls-active"
  "random|random|$PERSONAS|$USER_INSTR|--trolls-active"
)

# =====================================================
# MAIN LOOP
# =====================================================
for model_idx in "${!models[@]}"; do
    MODEL_URL="${models[$model_idx]}"
    MODEL_PSEUDO="${pseudos[$model_idx]}"

    for mod_strat_file in data/discussions_input/mod_instructions/*; do
        file_base=$(basename "$mod_strat_file" .yaml)

        for ablation in "${ablations[@]}"; do
            IFS="|" read -r \
                ablation_name \
                turn_manager \
                persona_path \
                user_instr_path \
                troll_flag <<< "$ablation"

            name="${MODEL_PSEUDO}_${file_base}_${ablation_name}"
            output_dir="${OUTPUT_DIR}/${name}"

            echo "Running model=$MODEL_PSEUDO strategy=$file_base ablation=$ablation_name"

            python src/run_experiment.py \
                --config-file "$CONFIG" \
                --model-url "$MODEL_URL" \
                --model-pseudo "$MODEL_PSEUDO" \
                --mod-strategy-file "$mod_strat_file" \
                --turn-manager "$turn_manager" \
                --output-dir "$output_dir" \
                --user-persona-path "$persona_path" \
                --user-instruction-path "$user_instr_path" \
                --mod-active \
                --num-experiments 5 \
                $troll_flag
        done
    done
done
