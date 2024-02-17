python train.py \
  --train-data = "Data/CoNiZnCu.csv" \
  --conditions = "charge temperature ionic_str" \
  --experiment-name = "test" \
  --folds = 5 \
  --seed = 23 \
  --batch-size = 64 \
  --epochs = 1000 \
  --es-patience = 100 \
  --mode = "regression"