#!/bin/sh

python Experiments/GCNN_FCNN/train_by_classes/optimize_hparams.py \
  --timeout 172800 \
  --train-data "Data/CoNiZnCu.csv" \
  --conditions "charge temperature ionic_str" \
  --experiment-name "CoNiZnCu_2" \
  --folds 1 \
  --seed 12 \
  --epochs 1000 \
  --es-patience 100 \
  --mode "regression"