#!/bin/bash

python Experiments/GCNN_GCNN/train_solubility.py \
  --train-sdf "Data/bigsoldb_train_24_10_14.sdf" \
  --test-sdf "Data/bigsoldb_test_24_10_14.sdf" \
  --experiment-name "all_descr"