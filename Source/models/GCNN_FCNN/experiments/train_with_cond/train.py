import logging
import os.path
import sys
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, MFConv

sys.path.append(os.path.abspath("."))
from Source.data import balanced_train_test_valid_split
from Source.models.GCNN.trainer import GCNNTrainer
from Source.models.GCNN_FCNN.featurizers import SkipatomFeaturizer, featurize_sdf_with_metal_and_conditions
from Source.models.GCNN_FCNN.model import GCNN_FCNN
from Source.models.GCNN_FCNN.old_featurizer import ConvMolFeaturizer
from Source.models.global_poolings import MaxPooling
from config import ROOT_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
time_mark = str(datetime.now()).replace(" ", "_").replace("-", "_").replace(":", "_").split(".")[0]

all_metals = ['Ac', 'Ag', 'Al', 'Am', 'Au', 'Ba', 'Be', 'Bi', 'Bk', 'Ca', 'Cd', 'Ce', 'Cf', 'Cm', 'Co', 'Cr', 'Cs',
              'Cu', 'Dy', 'Er', 'Eu', 'Fe', 'Ga', 'Gd', 'Hf', 'Hg', 'Ho', 'In', 'K', 'La', 'Li', 'Lu', 'Mg', 'Mn', 'Mo',
              'Na', 'Nd', 'Ni', 'Np', 'Pa', 'Pb', 'Pd', 'Pm', 'Pr', 'Pt', 'Pu', 'Rb', 'Re', 'Rh', 'Sb', 'Sc', 'Sm',
              'Sn', 'Sr', 'Tb', 'Th', 'Ti', 'Tl', 'Tm', 'U', 'V', 'Y', 'Yb', 'Zn', 'Zr']
test_metal = sys.argv[1]

cv_folds = 5
seed = 23
batch_size = 64
epochs = 1000
es_patience = 100
mode = "regression"
train_sdf_folder = ROOT_DIR / "Data/OneM_cond_adds"
output_folder = ROOT_DIR / f"Output/WithCondAdd/5fold/{test_metal}_{cv_folds}fold_{mode}_{time_mark}"

max_data = None
targets = ("logK",)
target_metrics = {
    target: {
        "R2": (r2_score, {}),
        "RMSE": (lambda *args, **kwargs: np.sqrt(mean_squared_error(*args, **kwargs)), {}),
        "MAE": (mean_absolute_error, {})
    } for target in targets
}
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

model = GCNN_FCNN(
    metal_features=next(iter(test_loader)).metal_x.shape[-1],
    node_features=next(iter(test_loader)).x.shape[-1],
    num_targets=len(targets),
    **model_parameters,
    optimizer=torch.optim.Adam,
    optimizer_parameters=None,
    mode="regression",
)

trainer = GCNNTrainer(
    model=model,
    train_valid_data=folds,
    test_data=test_loader,
    output_folder=output_folder,
    epochs=epochs,
    es_patience=es_patience,
    target_metrics=target_metrics,
    seed=seed,
)

trainer.train_cv_models()
