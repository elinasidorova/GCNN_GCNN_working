#!/bin/sh

python Experiments/GCNN/optimize_hparams.py \
  --timeout 172796 \
  --train-sdf "Data/operaqsol_3447_train.sdf" \
  --test-sdf "Data/operaqsol_3447_test.sdf" \
  --experiment-name "optuna_Solubility" \
  --folds 1 \
  --seed 12 \
  --epochs 1000 \
  --es-patience 100 \
  --mode "regression"