import logging
import os.path
import sys
from datetime import datetime

import numpy as np
import torch
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from torch import nn
from torch_geometric.loader import DataLoader

sys.path.append(os.path.abspath("."))
from Source.models.GCNN.trainer import GCNNTrainer
from Source.models.GCNN_FCNN.featurizers import DGLFeaturizer
from Source.models.MEGNet_FCNN.featurizers import featurize_sdf_with_metal_and_conditions
from Source.models.MEGNet_FCNN.model import MEGNetBimodal
from Source.models.global_poolings import ConcatPooling
from Source.data import balanced_train_test_valid_split
from config import ROOT_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
time_mark = str(datetime.now()).replace(" ", "_").replace("-", "_").replace(":", "_").split(".")[0]

test_metal = sys.argv[1]
all_metals = ['Ac', 'Ag', 'Al', 'Am', 'Au', 'Ba', 'Be', 'Bi', 'Bk', 'Ca', 'Cd', 'Ce', 'Cf', 'Cm', 'Co', 'Cr', 'Cs',
              'Cu', 'Dy', 'Er', 'Eu', 'Fe', 'Ga', 'Gd', 'Hf', 'Hg', 'Ho', 'In', 'K', 'La', 'Li', 'Lu', 'Mg', 'Mn', 'Mo',
              'Na', 'Nd', 'Ni', 'Np', 'Pa', 'Pb', 'Pd', 'Pm', 'Pr', 'Pt', 'Pu', 'Rb', 'Re', 'Rh', 'Sb', 'Sc', 'Sm',
              'Sn', 'Sr', 'Tb', 'Th', 'Ti', 'Tl', 'Tm', 'U', 'V', 'Y', 'Yb', 'Zn', 'Zr']

cv_folds = 5
seed = 23
batch_size = 32
epochs = 1000
es_patience = 100
mode = "regression"
output_folder = ROOT_DIR / f"Output/MengetChargeMetal/{test_metal}_megnet-cond-charge-metal_{mode}_{time_mark}"
train_sdf_folder = ROOT_DIR / "Data/OneM_cond"

mol_featurizer = DGLFeaturizer(add_self_loop=False,
                               node_featurizer=CanonicalAtomFeaturizer(),
                               edge_featurizer=CanonicalBondFeaturizer())
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
        "hidden": (256, 128,),
        "dropout": 0,
        "use_bn": True,
        "actf": nn.LeakyReLU(),
    },
    "megnet_params": {
        "n_megnet_blocks": 1,
        "pre_fc_edge_params": {
            "hidden": (),
            "dropout": 0,
            "use_bn": True,
            "actf": nn.LeakyReLU(),
        },
        "pre_fc_node_params": {
            "hidden": (224,),
            "dropout": 0,
            "use_bn": True,
            "actf": nn.LeakyReLU(),
        },
        "pre_fc_general_params": {
            "hidden": (),
            "dropout": 0,
            "use_bn": True,
            "actf": nn.LeakyReLU(),
        },
        "megnet_fc_edge_params": {
            "hidden": (256, 256, 256, 256,),
            "dropout": 0,
            "use_bn": True,
            "actf": nn.LeakyReLU(),
        },
        "megnet_fc_node_params": {
            "hidden": (256, 256, 256, 256,),
            "dropout": 0,
            "use_bn": True,
            "actf": nn.LeakyReLU(),
        },
        "megnet_fc_general_params": {
            "hidden": (256, 256, 256, 256,),
            "dropout": 0,
            "use_bn": True,
            "actf": nn.LeakyReLU(),
        },
        "megnet_conv_edge_params": {
            "hidden": (256,),
            "dropout": 0,
            "use_bn": True,
            "actf": nn.LeakyReLU(),
        },
        "megnet_conv_node_params": {
            "hidden": (256,),
            "dropout": 0,
            "use_bn": True,
            "actf": nn.LeakyReLU(),
        },
        "megnet_conv_general_params": {
            "hidden": (256,),
            "dropout": 0,
            "use_bn": True,
            "actf": nn.LeakyReLU(),
        },
        "post_fc_params": {
            "hidden": (256, 128,),
            "dropout": 0,
            "use_bn": True,
            "actf": nn.LeakyReLU(),
        },
        "pool": "mean",
    },
    "post_fc_params": {
        "hidden": (256, 128, 64, 32,),
        "dropout": 0,
        "use_bn": True,
        "actf": nn.LeakyReLU(),
    },
    "global_pooling": ConcatPooling,
}

logging.info("Featurizig...")
train_datasets = [
    featurize_sdf_with_metal_and_conditions(
        path_to_sdf=os.path.join(train_sdf_folder, f"{metal}.sdf"),
        mol_featurizer=mol_featurizer,
        z_in_metal=False,
    ) for metal in all_metals if metal != test_metal
]
folds = balanced_train_test_valid_split(train_datasets, n_folds=cv_folds,
                                        batch_size=batch_size,
                                        shuffle_every_epoch=True,
                                        seed=seed)
test_loader = DataLoader(
    featurize_sdf_with_metal_and_conditions(
        path_to_sdf=os.path.join(train_sdf_folder, f"{test_metal}.sdf"),
        mol_featurizer=mol_featurizer,
        z_in_metal=False,
    ),
    batch_size=batch_size)

model = MEGNetBimodal(
    metal_features=next(iter(test_loader)).metal_x.shape[-1],
    edge_features=next(iter(test_loader)).edge_attr.shape[-1],
    node_features=next(iter(test_loader)).x.shape[-1],
    global_features=next(iter(test_loader)).u.shape[-1],
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
