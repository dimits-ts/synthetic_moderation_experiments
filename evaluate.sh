#!/bin/bash
python eval/create_datasets.py \ 
    --discussions-root-dir data/discussions_output/ \
    --output-dir data

python eval/toxicity.py \
    --input-csv data/vmd.csv \
    --output-path data/eval_output/vmd.csv \
    --api-key-path perspective.key

python eval/toxicity.py \
    --input-csv data/ablatyion.csv \
    --output-path data/eval_output/ablation.csv \
    --api-key-path perspective.key