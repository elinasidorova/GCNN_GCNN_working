This folder contains code for uncertainty estimators, one sub-fold per method.

To reproduce all the results, do the following:

1. Go to the selected sub-folder 
2. Run `train.py` file to train model
3. In `save_predictions.py` change if needed:
   * `train_folder` - path for new trained model
   * `output_dir` - path to save predictions
4. Run `save_predictions.py` to get predictions on test dataset and save them into `.json`
5. Repeat 1-4 for all sub-folders you want
6. Go to [Notebooks/uncertainty.ipynb](../../../../../Notebooks/uncertainty.ipynb)
   and in "Loading information" paragraph check that data are loaded from appropriate locations
   (not needed if you didn't change `output_dir` in step 3)

You can skip steps 1-5 if you just want to play around with pre-calculated predictions, 
and directly go to [Notebooks/uncertainty.ipynb](../../../../../Notebooks/uncertainty.ipynb).
