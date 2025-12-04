#!/bin/bash
python eval/create_datasets.py \ 
    --discussions-root-dir data/discussions_output/ \
    --output-dir data/main_output

python eval/toxicity.py \
    --input-csv data/main_output/vmd.csv \
    --output-path data/eval_output/vmd.csv \
    --api-key-path perspective.key

python eval/toxicity.py \
    --input-csv data/main_output/ablation.csv \
    --output-path data/eval_output/ablation.csv \
    --api-key-path perspective.key

