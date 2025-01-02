#!/bin/bash

python Experiments/GCNN_FCNN/train_on_solvents/train.py \
  --sdf "Data/bigsoldb_24_10_14.sdf" \
  --experiment-name "all_descr_log_no_n-propanol_methanol" \
  --folds 5 \
  --seed 23 \
  --batch-size 20 \
  --epochs 1000 \
  --es-patience 100 \
  --mode "regression" \
  --include-descriptors "dG eps BP_mols BP_solvs" \
  --test-solvents "n-propanol methanol"