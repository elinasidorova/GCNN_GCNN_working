import logging
import os.path
import sys
from datetime import datetime

import torch
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from sklearn.metrics import r2_score, mean_absolute_error
from torch import nn
from torch_geometric.loader import DataLoader

from Source.models.GCNN.featurizers import DGLFeaturizer
from Source.models.GCNN.trainer import GCNNTrainer
from Source.models.GCNN_FCNN.featurizers import SkipatomFeaturizer
from Source.models.MEGNet.featurizers import featurize_sdf_with_metal
from Source.models.MEGNet.model import MEGNet
from config import ROOT_DIR

sys.path.append(os.path.abspath("."))

from Source.data import balanced_train_valid_split, root_mean_squared_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
time_mark = str(datetime.now()).replace(" ", "_").replace("-", "_").replace(":", "_").split(".")[0]

cv_folds = 5
seed = 27
batch_size = 32
epochs = 1
es_patience = 100
mode = "regression"
output_folder = ROOT_DIR / f"Output/MEGNet_Test_{mode}_{time_mark}"
train_sdf_folder = ROOT_DIR / "Data/OneM"

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
    "n_megnet_blocks": 1,
    "pre_fc_edge_params": {
        "hidden": (64,),
        "dropout": 0,
        "use_bn": False,
        "actf": nn.LeakyReLU(),
    },
    "pre_fc_node_params": {
        "hidden": (64,),
        "dropout": 0,
        "use_bn": False,
        "actf": nn.LeakyReLU(),
    },
    "pre_fc_general_params": {
        "hidden": (64,),
        "dropout": 0,
        "use_bn": False,
        "actf": nn.LeakyReLU(),
    },
    "megnet_fc_edge_params": {
        "hidden": (64,),
        "dropout": 0,
        "use_bn": False,
        "actf": nn.LeakyReLU(),
    },
    "megnet_fc_node_params": {
        "hidden": (64,),
        "dropout": 0,
        "use_bn": False,
        "actf": nn.LeakyReLU(),
    },
    "megnet_fc_general_params": {
        "hidden": (64,),
        "dropout": 0,
        "use_bn": False,
        "actf": nn.LeakyReLU(),
    },
    "megnet_conv_edge_params": {
        "hidden": (64,),
        "dropout": 0,
        "use_bn": False,
        "actf": nn.LeakyReLU(),
    },
    "megnet_conv_node_params": {
        "hidden": (64,),
        "dropout": 0,
        "use_bn": False,
        "actf": nn.LeakyReLU(),
    },
    "megnet_conv_general_params": {
        "hidden": (64,),
        "dropout": 0,
        "use_bn": False,
        "actf": nn.LeakyReLU(),
    },
    "post_fc_params": {
        "hidden": (64,),
        "dropout": 0,
        "use_bn": False,
        "actf": nn.LeakyReLU(),
    },
    "pool": "mean",
}

test_metal = "Cf"  # sys.argv[1]

transition_metals = ["Mg", "Al", "Ca", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Y", "Ag", "Cd", "Hg", "Pb", "Bi"]
Ln_metals = ["La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]
Ac_metals = ["Th", "Am", "Cm", "Bk", "Cf"]  # "U", "Pu",
all_metals = ["La"]  # transition_metals + Ln_metals + Ac_metals

logging.info("Featurizig...")
train_datasets = [featurize_sdf_with_metal(path_to_sdf=os.path.join(train_sdf_folder, f"{metal}.sdf"),
                                           mol_featurizer=DGLFeaturizer(add_self_loop=False,
                                                                        node_featurizer=CanonicalAtomFeaturizer(),
                                                                        edge_featurizer=CanonicalBondFeaturizer()),
                                           metal_featurizer=SkipatomFeaturizer())
                  for metal in all_metals if metal != test_metal]
folds = balanced_train_valid_split(train_datasets, n_folds=cv_folds,
                                   batch_size=batch_size,
                                   shuffle_every_epoch=True,
                                   seed=seed)

test_loader = DataLoader(featurize_sdf_with_metal(
    path_to_sdf=os.path.join(train_sdf_folder, f"{test_metal}.sdf"),
    mol_featurizer=DGLFeaturizer(add_self_loop=False, node_featurizer=CanonicalAtomFeaturizer(),
                                 edge_featurizer=CanonicalBondFeaturizer()),
    metal_featurizer=SkipatomFeaturizer()),
    batch_size=batch_size)

model = MEGNet(
    edge_features=next(iter(test_loader)).edge_attr.shape[-1],
    node_features=next(iter(test_loader)).x.shape[-1],
    global_features=next(iter(test_loader)).u.shape[-1],
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
