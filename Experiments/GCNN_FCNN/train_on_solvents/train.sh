#!/bin/bash

python Experiments/GCNN_FCNN/train_on_solvents/train.py \
  --train-sdf "Data/logS/train.sdf" \
  --experiment-name "test" \
  --folds 5 \
  --seed 23 \
  --batch-size 64 \
  --epochs 1000 \
  --es-patience 100 \
  --mode "regression"