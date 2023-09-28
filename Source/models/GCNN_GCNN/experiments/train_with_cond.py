import logging
import os.path
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, MFConv

from Source.models.GCNN.trainer import GCNNTrainer

sys.path.append(os.path.abspath("."))

from Source.models.GCNN_FCNN.old_featurizer import ConvMolFeaturizer
from Source.models.GCNN.featurizers import DGLFeaturizer
from Source.models.GCNN_GCNN.featurizers import DglMetalFeaturizer, featurize_df
from Source.models.GCNN_GCNN.model import GCNNGCNN
from Source.data import balanced_train_valid_split, root_mean_squared_error
from Source.models.global_poolings import ConcatPooling
from config import ROOT_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
time_mark = str(datetime.now()).replace(" ", "_").replace("-", "_").replace(":", "_").split(".")[0]

cv_folds = 1
seed = 23
batch_size = 32
epochs = 1000
es_patience = 100
mode = "regression"
output_folder = ROOT_DIR / f"Output/Check_U_1fold_{mode}_{time_mark}"
train_csv_folder = ROOT_DIR / "Data/OneM_cond"
test_csv = ""

targets = ({
               "name": "logK",
               "mode": "regression",
               "dim": 1,
               "metrics": {
                   "R2": (r2_score, {}),
                   "RMSE": (root_mean_squared_error, {}),
                   "MAE": (mean_absolute_error, {})
               },
               "loss": nn.MSELoss(),
           },)
model_parameters = {
    "mol_gcnn_params": {
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
    "metal_gcnn_params": {
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
    "global_pooling": ConcatPooling,
}

test_metal = "U"  # sys.argv[1]

all_metals = ['Ac']  # , 'Ag', 'Al', 'Am', 'Au', 'Ba', 'Be', 'Bi', 'Bk', 'Ca', 'Cd', 'Ce', 'Cf', 'Cm', 'Co', 'Cr', 'Cs',
# 'Cu', 'Dy', 'Er', 'Eu', 'Fe', 'Ga', 'Gd', 'Hf', 'Hg', 'Ho', 'In', 'K', 'La', 'Li', 'Lu', 'Mg', 'Mn', 'Mo',
# 'Na', 'Nd', 'Ni', 'Np', 'Pa', 'Pb', 'Pd', 'Pm', 'Pr', 'Pt', 'Pu', 'Rb', 'Re', 'Rh', 'Sb', 'Sc', 'Sm',
# 'Sn', 'Sr', 'Tb', 'Th', 'Ti', 'Tl', 'Tm', 'U', 'V', 'Y', 'Yb', 'Zn', 'Zr']

logging.info("Featurizig...")
train_datasets = [featurize_df(df=pd.read_csv(train_csv_folder / f"{metal}.csv"),
                               mol_featurizer=ConvMolFeaturizer(),
                               metal_featurizer=DGLFeaturizer(add_self_loop=False,
                                                              node_featurizer=DglMetalFeaturizer()))
                  for metal in all_metals if metal != test_metal]
folds = balanced_train_valid_split(train_datasets, n_folds=cv_folds,
                                   batch_size=batch_size,
                                   shuffle_every_epoch=True,
                                   seed=seed)

test_loader = DataLoader(featurize_df(
    df=pd.read_csv(test_csv),
    mol_featurizer=ConvMolFeaturizer(),
    metal_featurizer=DGLFeaturizer(add_self_loop=False,
                                   node_featurizer=DglMetalFeaturizer())),
    batch_size=batch_size)

model = GCNNGCNN(
    metal_node_features=next(iter(test_loader)).x_metal.shape[-1],
    mol_node_features=next(iter(test_loader)).x_mol.shape[-1],
    targets=targets,
    **model_parameters,
    optimizer=torch.optim.Adam,
    optimizer_parameters=None,
)

trainer = GCNNTrainer(
    model=model,
    train_valid_data=folds,
    test_data=test_loader,
    output_folder=output_folder,
    epochs=epochs,
    es_patience=es_patience,
    targets=targets,
    seed=seed,
)

trainer.train_cv_models()
