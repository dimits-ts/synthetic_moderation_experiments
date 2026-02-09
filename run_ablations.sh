#!/bin/bash

set -uo pipefail


models=(
  "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit"
  "unsloth/Mistral-Small-3.1-24B-Instruct-2503-bnb-4bit"
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


CONFIG="./data/discussions_input/run_config.yml"
PERSONAS="./data/discussions_input/personas/personas_employed.json"
USER_INSTR="./data/discussions_input/user_instructions/vanilla.txt"
OUTPUT_DIR="./data/discussions_output/ablations"

# =====================================================
# 1. NO-USER INSTRUCTION BASELINES
# =====================================================
for mod_strat_file in data/discussions_input/mod_instructions/*; do
    file_base=$(basename "$mod_strat_file" .yaml)

    for mod_idx in "${!mod_models[@]}"; do
        for user_idx in "${!user_models[@]}"; do

            MOD_MODEL_URL="${mod_models[$mod_idx]}"
            MOD_MODEL_PSEUDO="${mod_pseudos[$mod_idx]}"

            USER_MODEL_URL="${user_models[$user_idx]}"
            USER_MODEL_PSEUDO="${user_pseudos[$user_idx]}"

            name="${USER_MODEL_PSEUDO}_${MOD_MODEL_PSEUDO}_${file_base}_noinstructions"
            output_dir="${OUTPUT_DIR}/${name}"

            echo "Running MOD: user=$USER_MODEL_PSEUDO mod=$MOD_MODEL_PSEUDO strategy=$file_base"

            python src/run_experiment.py \
                --config-file "$CONFIG" \
                --user-model-url "$USER_MODEL_URL" \
                --user-model-pseudo "$USER_MODEL_PSEUDO" \
                --mod-model-url "$MOD_MODEL_URL" \
                --mod-model-pseudo "$MOD_MODEL_PSEUDO" \
                --mod-strategy-file "$mod_strat_file" \
                --turn-manager "random-weighted" \
                --output-dir "$output_dir" \
                --user-persona-path "$PERSONAS" \
                --user-instruction-path "./data/discussions_input/user_instructions/no_instructions.txt" \
                --mod-active \
                --num-experiments 5 \
                --trolls-active
        done
    done
done


# =====================================================
# 2. NO-TROLL BASELINES
# =====================================================
for mod_strat_file in data/discussions_input/mod_instructions/*; do
    file_base=$(basename "$mod_strat_file" .yaml)

    for mod_idx in "${!mod_models[@]}"; do
        for user_idx in "${!user_models[@]}"; do
            MOD_MODEL_URL="${mod_models[$mod_idx]}"
            MOD_MODEL_PSEUDO="${mod_pseudos[$mod_idx]}"

            USER_MODEL_URL="${user_models[$user_idx]}"
            USER_MODEL_PSEUDO="${user_pseudos[$user_idx]}"

            name="${USER_MODEL_PSEUDO}_${MOD_MODEL_PSEUDO}_${file_base}_notroll"
            output_dir="${OUTPUT_DIR}/${name}"

            echo "Running MOD: user=$USER_MODEL_PSEUDO mod=$MOD_MODEL_PSEUDO strategy=$file_base"

            python src/run_experiment.py \
                --config-file "$CONFIG" \
                --user-model-url "$USER_MODEL_URL" \
                --user-model-pseudo "$USER_MODEL_PSEUDO" \
                --mod-model-url "$MOD_MODEL_URL" \
                --mod-model-pseudo "$MOD_MODEL_PSEUDO" \
                --mod-strategy-file "$mod_strat_file" \
                --turn-manager "random-weighted" \
                --output-dir "$output_dir" \
                --user-persona-path "$PERSONAS" \
                --user-instruction-path "$USER_INSTR" \
                --mod-active \
                --num-experiments 5 \
                --no-trolls-active
        done
    done
done


# =====================================================
# 3. NO-SDBS BASELINES
# =====================================================
for mod_strat_file in data/discussions_input/mod_instructions/*; do
    file_base=$(basename "$mod_strat_file" .yaml)

    for mod_idx in "${!mod_models[@]}"; do
        for user_idx in "${!user_models[@]}"; do

            MOD_MODEL_URL="${mod_models[$mod_idx]}"
            MOD_MODEL_PSEUDO="${mod_pseudos[$mod_idx]}"

            USER_MODEL_URL="${user_models[$user_idx]}"
            USER_MODEL_PSEUDO="${user_pseudos[$user_idx]}"

            name="${USER_MODEL_PSEUDO}_${MOD_MODEL_PSEUDO}_${file_base}_nosdbs"
            output_dir="${OUTPUT_DIR}/${name}"

            echo "Running MOD: user=$USER_MODEL_PSEUDO mod=$MOD_MODEL_PSEUDO strategy=$file_base"

            python src/run_experiment.py \
                --config-file "$CONFIG" \
                --user-model-url "$USER_MODEL_URL" \
                --user-model-pseudo "$USER_MODEL_PSEUDO" \
                --mod-model-url "$MOD_MODEL_URL" \
                --mod-model-pseudo "$MOD_MODEL_PSEUDO" \
                --mod-strategy-file "$mod_strat_file" \
                --turn-manager "random-weighted" \
                --output-dir "$output_dir" \
                --user-persona-path "data/discussions_input/personas/no_sdbs.json" \
                --user-instruction-path "$USER_INSTR" \
                --mod-active \
                --num-experiments 5 \
                --trolls-active
        done
    done
done


# =====================================================
# 4. TURN-MANAGERS BASELINES
# =====================================================
turn_managers=( "round-robin" "random" )

for mod_strat_file in data/discussions_input/mod_instructions/*; do
    file_base=$(basename "$mod_strat_file" .yaml)

    for mod_idx in "${!mod_models[@]}"; do
        for user_idx in "${!user_models[@]}"; do
            for turn_manager in "${turn_managers[@]}"; do

                MOD_MODEL_URL="${mod_models[$mod_idx]}"
                MOD_MODEL_PSEUDO="${mod_pseudos[$mod_idx]}"

                USER_MODEL_URL="${user_models[$user_idx]}"
                USER_MODEL_PSEUDO="${user_pseudos[$user_idx]}"

                name="${USER_MODEL_PSEUDO}_${MOD_MODEL_PSEUDO}_${turn_manager}_${file_base}"
                output_dir="${OUTPUT_DIR}/${name}"

                echo "Running MOD: user=$USER_MODEL_PSEUDO mod=$MOD_MODEL_PSEUDO strategy=$file_base tm=$turn_manager"

                python src/run_experiment.py \
                    --config-file "$CONFIG" \
                    --user-model-url "$USER_MODEL_URL" \
                    --user-model-pseudo "$USER_MODEL_PSEUDO" \
                    --mod-model-url "$MOD_MODEL_URL" \
                    --mod-model-pseudo "$MOD_MODEL_PSEUDO" \
                    --mod-strategy-file "$mod_strat_file" \
                    --turn-manager "$turn_manager" \
                    --output-dir "$output_dir" \
                    --user-persona-path "$PERSONAS" \
                    --user-instruction-path "$USER_INSTR" \
                    --mod-active \
                    --num-experiments 5 \
                    --trolls-active
            done
        done
    done
done