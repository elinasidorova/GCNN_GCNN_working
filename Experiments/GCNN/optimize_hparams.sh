python Experiments/GCNN/optimize_hparams.py \
  --timeout 172796 \
  --train-data "Data/Cu.csv" \
  --experiment-name "optuna_Cu" \
  --folds 1 \
  --seed 12 \
  --epochs 1000 \
  --es-patience 100 \
  --mode "regression"