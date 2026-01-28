#!/bin/bash

python src/create_datasets.py \
    --discussions-root-dir data/discussions_output \
    --output-dir data/main_output

python src/cost_calculation.py \
  --mode proprietary \
  --num-tasks 2273 \
  --isl-tokens 31580 \
  --osl-tokens 7860

python src/cost_calculation.py \
  --mode human \
  --num-tasks 2273 \
  --time-per-task-seconds 300

python src/cost_calculation.py \
  --mode open-source \
  --num-tasks 2273 \
  --requests-per-task 21 \
  --experiment-duration-days 4

python src/cost_calculation.py \
  --mode open-source \
  --num-tasks 2273 \
  --requests-per-task 21 \
  --initial-server-cost 0 \
  --experiment-duration-days 4 \
  --power-watts-per-server 2000

python src/generate_toxicity_ratings.py \
    --input-csv data/main_output/vmd.csv \
    --output-path data/eval_output/vmd.csv \
    --api-key-path perspective.key

python src/generate_toxicity_ratings.py \
    --input-csv data/main_output/ablation.csv \
    --output-path data/eval_output/ablation.csv \
    --api-key-path perspective.key

python src/eval_moderation.py \
    --input-csv data/main_output/vmd.csv \
    --output-dir graphs

python src/eval_dataset_analysis.py \
    --input-csv data/main_output/vmd.csv \
    --output-dir graphs   

python src/eval_toxicity.py \
    --main-output-dir data/main_output \
    --toxicity-rating-dir data/eval_output \
    --graph-output-dir graphs