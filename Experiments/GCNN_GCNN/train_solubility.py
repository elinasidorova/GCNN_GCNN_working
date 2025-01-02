import logging
from argparse import ArgumentParser #добавила аргпарсер
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
from Source.models.GCNN_GCNN.featurizers import DglMetalFeaturizer, featurize_sdf_mol_solv
from Source.models.GCNN_GCNN.model import GCNN_GCNN
from Source.data import balanced_train_valid_split, root_mean_squared_error
from Source.models.global_poolings import ConcatPooling
from config import ROOT_DIR
from argparse import Namespace

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
time_mark = str(datetime.now()).replace(" ", "_").replace("-", "_").replace(":", "_").split(".")[0]


# parser = ArgumentParser()
# parser.add_argument('--train-sdf', type=str, help='Path to the data file')  # required=True,
# parser.add_argument('--test-sdf', type=str, help='Path to the data file')
#
# parser.add_argument('--experiment-name', type=str, help='The name of the experiment')  # required=True,
#
# args = parser.parse_args()
args = Namespace(train_sdf="Data/bigsoldb_24_10_14.sdf", test_sdf="Data/bigsoldb_24_10_14.sdf", experiment_name="test")
#теперь: metal -> molecule, mol -> solvent

targets = ({
               "name": 'Solubility', #logK -> Solubility
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
    "solvent_gcnn_params": { #mol_gcnn_params -> solvent_gcnn_params
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
    "molecule_gcnn_params": { #metal_gcnn_params -> molecule_gcnn_params
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

#это пока не надо
# test_metal = "U"
# all_metals = ['Ac']

cv_folds = 1
seed = 23
batch_size = 16
epochs = 1000
es_patience = 100
mode = "regression"

logging.info("Featurizing...")


train_datasets = [featurize_sdf_mol_solv(path_to_sdf=str(ROOT_DIR / args.train_sdf),
                                         solvent_featurizer=ConvMolFeaturizer(),
                                         molecule_featurizer=ConvMolFeaturizer())] #списки списков PairData по элементам -> список PairData по точкам

print('train datasets')
print(train_datasets)
folds= balanced_train_valid_split(datasets=train_datasets,
                                   n_folds=1,
                                   batch_size=batch_size,
                                   shuffle_every_epoch=True,
                                   seed=seed)


test_loader = DataLoader(featurize_sdf_mol_solv(path_to_sdf=str(ROOT_DIR / args.test_sdf),
                                         solvent_featurizer=ConvMolFeaturizer(),
                                         molecule_featurizer=ConvMolFeaturizer()), batch_size=batch_size)


model = GCNN_GCNN(
    molecule_node_features=75,
    solvent_node_features=75,
    targets=targets,
    **model_parameters,
    optimizer=torch.optim.Adam,
    optimizer_parameters=None,
)

#это точно оставим как есть

trainer = GCNNTrainer(
    model=model,
    train_valid_data=folds,
    test_data=test_loader,
    output_folder=str(ROOT_DIR / "Output" / args.experiment_name),
    epochs=epochs,
    es_patience=es_patience,
    targets=targets,
    seed=seed,
)

trainer.train_cv_models()
