#!/bin/bash

python Experiments/GCNN_FCNN/train_on_solvents/optimize_hparams/optimize_hparams.py \
  --timeout 172800 \
  --train-sdf "Data/logS/train.sdf" \
  --test-sdf "Data/logS/test.sdf" \
  --experiment-name "optimize_hparams_test" \
  --folds 1 \
  --seed 12 \
  --epochs 1000 \
  --es-patience 100 \
  --mode "regression"