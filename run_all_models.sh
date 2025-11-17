#!/bin/bash

# Since clearing GPU memory is notoriously difficult, just run these using a bash script
models=( "unsloth/Llama-3.3-70B-Instruct-bnb-4bit" "unsloth/OLMo-2-0325-32B-Instruct-unsloth-bnb-4bit" "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit" "unsloth/Qwen2.5-32B-Instruct-bnb-4bit" )
pseudos=( "llama70b" "olmo32b" "mistralnemo" "qwen32b" )

for i in "${!models[@]}"; do
    python run.py --config-file data/discussions_input/run_config.yml --model-url "${models[i]}" --model-pseudo "${pseudos[i]}"
done