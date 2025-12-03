#!/bin/bash

models=( 
    "unsloth/OLMo-2-0325-32B-Instruct-unsloth-bnb-4bit"
    "unsloth/Qwen2.5-32B-Instruct-bnb-4bit"
    "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"
    "unsloth/Ministral-3-14B-Base-2512-bnb-4bit"
)

pseudos=(
    "olmo32b"
    "qwen32b"
    "llama70b"
    "mistral"
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
            --user-instruction-path "./data/discussions_input/user_instructions/vanilla.txt" \
            --user-persona-path "./data/discussions_input/personas/personas.json" \
            --no-mod-active \
            --trolls-active
    done
done

for i in "${!models[@]}"; do
    for mod_strat_file in data/discussions_input/mod_instructions/*; do
        for turn_manager in "${turn_managers[@]}"; do
            file_base=$(basename "$mod_strat_file" .yaml)
            name="${pseudos[i]}_${turn_manager}_${file_base}_notrolls"

            python run_experiment.py \
                --config-file ./data/discussions_input/run_config.yml \
                --model-url "${models[i]}" \
                --model-pseudo "${pseudos[i]}" \
                --mod-strategy-file "${mod_strat_file}" \
                --turn-manager "${turn_manager}" \
                --output-dir  "./data/discussions_output/${name}" \
                --user-instruction-path "./data/discussions_input/user_instructions/vanilla.txt" \
                --user-persona-path "./data/discussions_input/personas/personas.json" \
                --no-trolls-active \
                --mod-active \
                --num-experiments=5
        done
    done
done


for i in "${!models[@]}"; do
    for mod_strat_file in data/discussions_input/mod_instructions/*; do
        for turn_manager in "${turn_managers[@]}"; do
            file_base=$(basename "$mod_strat_file" .yaml)
            name="${pseudos[i]}_${turn_manager}_${file_base}_noinstr"

            python run_experiment.py \
                --config-file ./data/discussions_input/run_config.yml \
                --model-url "${models[i]}" \
                --model-pseudo "${pseudos[i]}" \
                --mod-strategy-file "${mod_strat_file}" \
                --turn-manager "${turn_manager}" \
                --output-dir  "./data/discussions_output/${name}" \
                --user-instruction-path "./data/discussions_input/user_instructions/no_instructions.txt" \
                --user-persona-path "./data/discussions_input/personas/personas.json" \
                --num-experiments=5
        done
    done
done


for i in "${!models[@]}"; do
    for mod_strat_file in data/discussions_input/mod_instructions/*; do
        for turn_manager in "${turn_managers[@]}"; do
            file_base=$(basename "$mod_strat_file" .yaml)
            name="${pseudos[i]}_${turn_manager}_${file_base}_nosdbs"

            python run_experiment.py \
                --config-file ./data/discussions_input/run_config.yml \
                --model-url "${models[i]}" \
                --model-pseudo "${pseudos[i]}" \
                --mod-strategy-file "${mod_strat_file}" \
                --turn-manager "${turn_manager}" \
                --output-dir  "./data/discussions_output/${name}" \
                --user-instruction-path "./data/discussions_input/user_instructions/vanilla.txt" \
                --user-persona-path "./data/discussions_input/personas/no_sdbs.json" \
                --num-experiments=5
        done
    done
done
