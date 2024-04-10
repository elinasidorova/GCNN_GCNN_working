#!/bin/bash

python Experiments/GCNN_FCNN/train_on_solvents/train.py \
  --train-sdf "Data/logS/train.sdf" \
  --test-sdf "Data/logS/test.sdf" \
  --experiment-name "Train on solvents" \
  --folds 5 \
  --seed 23 \
  --batch-size 20 \
  --epochs 1000 \
  --es-patience 100 \
  --mode "regression"