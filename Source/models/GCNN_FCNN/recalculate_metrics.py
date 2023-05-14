import logging
import os.path
import sys

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from torch_geometric.loader import DataLoader

from Source.models.GCNN_FCNN.old_featurizer import ConvMolFeaturizer
from config import ROOT_DIR

sys.path.append(os.path.abspath("."))

from model import GCNN_FCNN
from Source.models.GCNN.trainer import GCNNTrainer
from Source.trainer import ModelShell
from Source.data import balanced_train_test_valid_split
from Source.models.GCNN_FCNN.featurizers import SkipatomFeaturizer, featurize_sdf_with_metal_and_conditions

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

cv_folds = 5
seed = 27
batch_size = 32
epochs = 1000
es_patience = 100
train_sdf_folder = ROOT_DIR / "Data/OneM_cond"
train_folder = ROOT_DIR / "Output/WithCond/La_1fold_regression_2023_05_14_13_34_39"

targets = ("logK",)
target_metrics = {
    target: {
        "R2": (r2_score, {}),
        "RMSE": (lambda *args, **kwargs: np.sqrt(mean_squared_error(*args, **kwargs)), {}),
        "MAE": (mean_absolute_error, {})
    } for target in targets
}

test_metal = "La"

all_metals = ['Ac', 'Ag', 'Al', 'Am', 'Au', 'Ba', 'Be', 'Bi', 'Bk', 'Ca', 'Cd', 'Ce', 'Cf', 'Cm', 'Co', 'Cr', 'Cs',
              'Cu', 'Dy', 'Er', 'Eu', 'Fe', 'Ga', 'Gd', 'Hf', 'Hg', 'Ho', 'In', 'K', 'La', 'Li', 'Lu', 'Mg', 'Mn', 'Mo',
              'Na', 'Nd', 'Ni', 'Np', 'Pa', 'Pb', 'Pd', 'Pm', 'Pr', 'Pt', 'Pu', 'Rb', 'Re', 'Rh', 'Sb', 'Sc', 'Sm',
              'Sn', 'Sr', 'Tb', 'Th', 'Ti', 'Tl', 'Tm', 'U', 'V', 'Y', 'Yb', 'Zn', 'Zr']

super_model = ModelShell(GCNN_FCNN, train_folder)

logging.info("Featurizig...")
train_datasets = [featurize_sdf_with_metal_and_conditions(path_to_sdf=os.path.join(train_sdf_folder, f"{metal}.sdf"),
                                                          mol_featurizer=ConvMolFeaturizer(),
                                                          metal_featurizer=SkipatomFeaturizer())
                  for metal in all_metals if metal != test_metal]
folds = balanced_train_test_valid_split(train_datasets, n_folds=cv_folds,
                                        batch_size=batch_size,
                                        shuffle_every_epoch=True,
                                        seed=seed)

test_loader = DataLoader(featurize_sdf_with_metal_and_conditions(
    path_to_sdf=os.path.join(train_sdf_folder, f"{test_metal}.sdf"),
    mol_featurizer=ConvMolFeaturizer(),
    metal_featurizer=SkipatomFeaturizer()),
    batch_size=batch_size)

trainer = GCNNTrainer(
    model=None,
    train_valid_data=folds,
    test_data=test_loader,
    target_metrics=target_metrics,
    seed=seed,
)
trainer.models = super_model.models
result = trainer.calculate_metrics()

print(result)
