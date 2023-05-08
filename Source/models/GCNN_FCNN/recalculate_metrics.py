import logging
import os.path
import sys

import numpy as np
from dgllife.utils import CanonicalAtomFeaturizer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from torch_geometric.loader import DataLoader

sys.path.append(os.path.abspath("."))

from model import GCNNBimodal
from Source.models.GCNN.trainer import GCNNTrainer
from Source.trainer import ModelShell
from Source.data import balanced_train_test_valid_split
from Source.featurizers.featurizers import DGLFeaturizer, SkipatomFeaturizer, featurize_sdf_with_metal

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

cv_folds = 5
seed = 27
batch_size = 32
epochs = 1000
es_patience = 100
train_sdf_folder = "../../../Data/OneM"
train_folder = "Output/Test_regression_2023_05_05_21_06_28"

targets = ("logK",)
target_metrics = {
    target: {
        "R2": (r2_score, {}),
        "RMSE": (lambda *args, **kwargs: np.sqrt(mean_squared_error(*args, **kwargs)), {}),
        "MAE": (mean_absolute_error, {})
    } for target in targets
}

test_metal = "Cu"

transition_metals = ["Mg", "Al", "Ca", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Y", "Ag", "Cd", "Hg", "Pb", "Bi"]
Ln_metals = ["La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]
Ac_metals = ["Th", "Am", "Cm", "Bk", "Cf"]  # "U", "Pu",
all_metals = ["La"]  # transition_metals + Ln_metals + Ac_metals

logging.info("Featurizig...")
train_datasets = [featurize_sdf_with_metal(path_to_sdf=os.path.join(train_sdf_folder, f"{metal}.sdf"),
                                           mol_featurizer=DGLFeaturizer(add_self_loop=False,
                                                                        node_featurizer=CanonicalAtomFeaturizer()),
                                           metal_featurizer=SkipatomFeaturizer(
                                               "../../featurizers/skipatom_vectors_dim200.json"))
                  for metal in all_metals if metal != test_metal]
folds = balanced_train_test_valid_split(train_datasets, n_folds=cv_folds,
                                        batch_size=batch_size,
                                        shuffle_every_epoch=True,
                                        seed=seed)

test_loader = DataLoader(featurize_sdf_with_metal(
    path_to_sdf=os.path.join(train_sdf_folder, f"{test_metal}.sdf"),
    mol_featurizer=DGLFeaturizer(add_self_loop=False, node_featurizer=CanonicalAtomFeaturizer()),
    metal_featurizer=SkipatomFeaturizer("../../featurizers/skipatom_vectors_dim200.json")),
    batch_size=batch_size)


super_model = ModelShell(GCNNBimodal, train_folder)
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
