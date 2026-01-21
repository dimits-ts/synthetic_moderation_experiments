#!/bin/bash

python src/create_datasets.py \
    --discussions-root-dir data/discussions_output \
    --output-dir data/main_output

python src/cost_calculation.py \
    --num-tasks=2273 \
    --isl-tokens=31580 \
    --osl-tokens=7860

python eval/toxicity.py \
    --input-csv data/main_output/vmd.csv \
    --output-path data/eval_output/vmd.csv \
    --api-key-path perspective.key

python eval/toxicity.py \
    --input-csv data/main_output/ablation.csv \
    --output-path data/eval_output/ablation.csv \
    --api-key-path perspective.key

