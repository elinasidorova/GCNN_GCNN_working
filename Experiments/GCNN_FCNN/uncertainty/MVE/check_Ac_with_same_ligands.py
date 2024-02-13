import logging
import os.path
import sys

import pandas as pd
import torch
from rdkit import Chem
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MFConv, global_mean_pool
from tqdm import tqdm

from Source.models.GCNN.trainer import GCNNTrainer
from Source.models.GCNN_FCNN.experiments.uncertainty.MVE.metrics import r2_score_MVE, root_mean_squared_error_MVE, \
    mean_absolute_error_MVE, negative_log_likelihood
from Source.models.GCNN_FCNN.model import GCNN_FCNN
from Source.models.GCNN_FCNN.old_featurizer import ConvMolFeaturizer
from Source.models.global_poolings import MaxPooling
from config import ROOT_DIR

sys.path.append(os.path.abspath("."))

from Source.trainer import ModelShell
from Source.data import balanced_train_valid_test_split, balanced_train_valid_split
from Source.models.GCNN_FCNN.featurizers import SkipatomFeaturizer, featurize_sdf_with_metal_and_conditions

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

other_metals = ['Li', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                'Ga', 'Rb', 'Sr', 'Y', 'Zr', 'Mo', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Cs', 'Ba', 'Hf', 'Re',
                'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi']
Ln_metals = ['La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', ]
Ac_metals = ['Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf']

train_metals = list(set(["Y", "Sc"] + Ln_metals))
test_metals = list(set(Ac_metals) - {"Ac", "Pa"})


cv_folds = 1
test_size = 0.1
seed = 23
batch_size = 64
epochs = 1000
es_patience = 100
mode = "regression"
train_sdf_folder = ROOT_DIR / "Data/OneM_cond_adds"
output_dir = ROOT_DIR / "Output/Uncertainty/MVE_AcTest/"
train_folder = ROOT_DIR / f"Output/Uncertainty_MVE/AcTest_1fold_regression_2023_08_07_13_36_03"

targets = ({
               "name": "logK",
               "mode": "regression",
               "dim": 2,
               "metrics": {
                   "R2": (r2_score_MVE, {}),
                   "RMSE": (root_mean_squared_error_MVE, {}),
                   "MAE": (mean_absolute_error_MVE, {})
               },
               "loss": negative_log_likelihood,
           },)
model_parameters = {
    "metal_fc_params": {
        "hidden": (256, 128, 128, 64, 64,),
        "dropout": 0.25108912274809364,
        "use_bn": False,
        "actf": nn.LeakyReLU(),
    },
    "gcnn_params": {
        "pre_fc_params": {
            "hidden": (),
            "dropout": 0,
            "actf": nn.LeakyReLU(),
        },
        "post_fc_params": {
            "hidden": (),
            "dropout": 0,
            "use_bn": False,
            "actf": nn.LeakyReLU(),
        },
        "hidden_conv": (128, 128, 64,),
        "conv_dropout": 0.27936243337975536,
        "conv_actf": nn.LeakyReLU(),
        "conv_layer": MFConv,
        "conv_parameters": None,
        "graph_pooling": global_mean_pool
    },
    "post_fc_params": {
        "hidden": (256,),
        "dropout": 0.06698879155641034,
        "use_bn": False,
        "actf": nn.LeakyReLU(),
    },
    "global_pooling": MaxPooling,
}

train_datasets = [featurize_sdf_with_metal_and_conditions(path_to_sdf=os.path.join(train_sdf_folder, f"{metal}.sdf"),
                                                          mol_featurizer=ConvMolFeaturizer(),
                                                          metal_featurizer=SkipatomFeaturizer())
                  for metal in tqdm(train_metals, desc="Featurizig")]
logging.info("Splitting...")
folds = balanced_train_valid_split(train_datasets, n_folds=cv_folds,
                                   batch_size=batch_size,
                                   shuffle_every_epoch=True,
                                   seed=seed)
train_mols = []
for metal in train_metals:
    train_mols += [mol for mol in Chem.SDMolSupplier(os.path.join(train_sdf_folder, f"{metal}.sdf")) if mol is not None]

train_smiles = [Chem.MolToSmiles(m) for m in train_mols]

test_data = []
for test_metal in test_metals:
    mols = [mol for mol in Chem.SDMolSupplier(os.path.join(train_sdf_folder, f"{test_metal}.sdf"))]
    cleaned_mols = [m for m in mols if Chem.MolToSmiles(m) in train_smiles and m is not None]

    test_data += featurize_sdf_with_metal_and_conditions(
        molecules=cleaned_mols,
        mol_featurizer=ConvMolFeaturizer(),
        metal_featurizer=SkipatomFeaturizer())

test_loader = DataLoader(test_data, batch_size=batch_size)

trainer = GCNNTrainer(
    model=None,
    train_valid_data=folds,
    test_data=test_loader,
    targets=targets,
    seed=seed,
)
trainer.models = ModelShell(GCNN_FCNN, train_folder).models
result = trainer.calculate_metrics()

print(result)
