#!/bin/bash

models=( 
    "unsloth/OLMo-2-0325-32B-Instruct-unsloth-bnb-4bit"
    "unsloth/Qwen2.5-32B-Instruct-bnb-4bit"
    "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"
    "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit"
)

pseudos=(
    "olmo32b"
    "qwen32b"
    "llama70b"
    "mistralnemo"
)

turn_managers=( "round-robin" "random" "random-weighted" )

for i in "${!models[@]}"; do
    for turn_manager in "${turn_managers[@]}"; do
        name="${pseudos[i]}_${turn_manager}_nomod"

        python run_experiment.py \
            --config-file ./data/discussions_input/run_config.yml \
            --model-url "${models[i]}" \
            --model-pseudo "${pseudos[i]}" \
            --mod-strategy-file "${mod_strat_file}" \
            --turn-manager "${turn_manager}" \
            --output-dir  "./data/discussions_output/${name}" \
            --no-mod-active \
            --trolls-active
    done
done

for i in "${!models[@]}"; do
    for turn_manager in "${turn_managers[@]}"; do
        name="${pseudos[i]}_${turn_manager}_notrolls"

        python run_experiment.py \
            --config-file ./data/discussions_input/run_config.yml \
            --model-url "${models[i]}" \
            --model-pseudo "${pseudos[i]}" \
            --mod-strategy-file "${mod_strat_file}" \
            --turn-manager "${turn_manager}" \
            --output-dir  "./data/discussions_output/${name}" \
            --no-mod-active
    done
done
