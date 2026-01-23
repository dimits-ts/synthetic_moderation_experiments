#!/bin/bash

set -euo pipefail


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
    "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
    "unsloth/Qwen3-4B-unsloth-bnb-4bit"
    "unsloth/Phi-4-mini-instruct-bnb-4bit"
)
user_pseudos=(
    "llama3b"
    "qwen4b"
    "phi4b"
)

turn_managers=( "round-robin" "random" "random-weighted" )


for mod_strat_file in data/discussions_input/mod_instructions/*; do
    for turn_manager in "${turn_managers[@]}"; do
        for mod_idx in "${!mod_models[@]}"; do
            for user_idx in "${!user_models[@]}"; do

                MOD_MODEL_URL="${mod_models[$mod_idx]}"
                MOD_MODEL_PSEUDO="${mod_pseudos[$mod_idx]}"

                USER_MODEL_URL="${user_models[$user_idx]}"
                USER_MODEL_PSEUDO="${user_pseudos[$user_idx]}"

                file_base=$(basename "$mod_strat_file" .yaml)
                name="${USER_MODEL_PSEUDO}_${MOD_MODEL_PSEUDO}_${turn_manager}_${file_base}"

                echo "Running: user=$USER_MODEL_PSEUDO mod=$MOD_MODEL_PSEUDO strategy=$file_base turn=$turn_manager"

                python src/run_experiment.py \
                    --config-file ./data/discussions_input/run_config.yml \
                    --user-model-url "$USER_MODEL_URL" \
                    --user-model-pseudo "$USER_MODEL_PSEUDO" \
                    --mod-model-url "$MOD_MODEL_URL" \
                    --mod-model-pseudo "$MOD_MODEL_PSEUDO" \
                    --mod-strategy-file "$mod_strat_file" \
                    --turn-manager "$turn_manager" \
                    --output-dir "./data/discussions_output/${name}" \
                    --user-persona-path "./data/discussions_input/personas/personas.json" \
                    --user-instruction-path "./data/discussions_input/user_instructions/vanilla.txt" \
                    --mod-active \
                    --num-experiments 5 \
                    --trolls-active

            done
        done
    done
done
