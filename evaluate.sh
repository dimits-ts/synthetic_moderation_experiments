#!/bin/bash

python src/create_datasets.py \
    --discussions-root-dir data/discussions_output/main \
    --output-path data/main_output/vmd.csv

python src/create_datasets.py \
    --discussions-root-dir data/discussions_output/ablations \
    --output-path data/main_output/ablation.csv

python src/cost_calculation.py \
  --mode proprietary \
  --n-tasks 1800 \
  --prop-isl 47370 \
  --prop-osl 11790 \
  --prop-price-in-per-million 1.75 \
  --prop-price-out-per-million 14

python src/cost_calculation.py \
  --mode human \
  --n-tasks 1800 \
  --human-time-per-task-seconds 300 \
  --human-wage-gross 12 \
  --human-platform-fee-frac 0.33 \
  --human-n-humans 7

python src/cost_calculation.py \
  --mode open-source \
  --n-tasks 1800 \
  --experiment-days 7 \
  --os-utilization 0.9 \
  --os-rps-per-instance 0.0017 \
  --os-server-cost 000 \
  --os-power-watts 1500

python src/cost_calculation.py \
  --mode open-source \
  --n-tasks 1800 \
  --os-server-cost 0 \
  --experiment-days 5 \
  --os-utilization 0.9 \
  --os-rps-per-instance 0.0034 \
  --os-power-watts 500

python src/generate_toxicity_ratings.py \
    --input-csv data/main_output/vmd.csv \
    --output-path data/eval_output/vmd.csv \
    --api-key-path perspective.key

python src/generate_toxicity_ratings.py \
    --input-csv data/main_output/ablation.csv \
    --output-path data/eval_output/ablation.csv \
    --api-key-path perspective.key

python src/generate_toxicity_ratings.py \
    --input-csv data/cmv_awry2.csv \
    --output-path data/eval_output/cmv_awry2.csv \
    --api-key-path perspective.key

python src/eval_moderation.py \
    --input-csv data/main_output/vmd.csv \
    --graph-output-dir graphs \
    --stats-output-dir data/eval_output

python src/eval_dataset_analysis.py \
    --main-output-dir data/main_output \
    --graph-output-dir graphs \
    --human-csv data/cmv_awry2.csv \
    --cache-dir data/cache \
    --stats-output-dir data/eval_output

python src/eval_toxicity.py \
  --vmd-path data/main_output/vmd.csv \
  --ablation-path data/main_output/ablation.csv \
  --toxicity-rating-dir data/eval_output \
  --graph-output-dir graphs \
  --human-path data/cmv_awry2.csv \
  --stats-output-dir data/eval_output