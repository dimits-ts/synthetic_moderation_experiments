#!/bin/bash

# Since clearing GPU memory is notoriously difficult, just run these using a bash script
models=( 
    "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"
    "unsloth/OLMo-2-0325-32B-Instruct-unsloth-bnb-4bit"
    "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit"
    "unsloth/Qwen2.5-32B-Instruct-bnb-4bit"
)

pseudos=(
    "llama70b"
    "olmo32b"
    "mistralnemo"
    "qwen32b"
)

turn_managers=( "random-weighted" "round-robin" "random" )

for turn_manager in "${turn_managers[@]}"; do
    for i in "${!models[@]}"; do
        name="${pseudos[i]}_${turn_manager}_nomod"

        echo "======================================================"
        echo "[NO-MOD RUN]"
        echo " Model        : ${pseudos[i]}"
        echo " Turn Manager : ${turn_manager}"
        echo "======================================================"

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

for mod_strat_file in data/discussions_input/mod_instructions/*; do
    for turn_manager in "${turn_managers[@]}"; do
        for i in "${!models[@]}"; do
            file_base=$(basename "$mod_strat_file" .yaml)
            name="${pseudos[i]}_${turn_manager}_${file_base}_yesmod"

            echo "======================================================"
            echo "[YES-MOD RUN]"
            echo " Model        : ${pseudos[i]}"
            echo " Turn Manager : ${turn_manager}"
            echo " Mod File     : ${file_base}"
            echo " Output Dir   : ./data/discussions_output/${name}"
            echo "======================================================"

            python run_experiment.py \
                --config-file ./data/discussions_input/run_config.yml \
                --model-url "${models[i]}" \
                --model-pseudo "${pseudos[i]}" \
                --mod-strategy-file "${mod_strat_file}" \
                --turn_manager "${turn_manager}" \
                --output-dir  "./data/discussions_output/${name}" \
                --mod-active
        done
    done
done
