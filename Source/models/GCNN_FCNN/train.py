import logging
import numpy as np
import os.path
import sys
import torch
from datetime import datetime
from dgllife.utils import CanonicalAtomFeaturizer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, MFConv

from Source.featurizers.featurizers import featurize_sdf_with_metal
from Source.models.global_poolings import ConcatPooling

sys.path.append(os.path.abspath("../../../"))

from model import GCNNBimodal
from trainer import GCNNTrainer
from Source.data import balanced_train_test_valid_split
from Source.featurizers.featurizers import DGLFeaturizer, SkipatomFeaturizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
time_mark = str(datetime.now()).replace(" ", "_").replace("-", "_").replace(":", "_").split(".")[0]

cv_folds = 5
seed = 27
batch_size = 32
epochs = 1000
es_patience = 100
mode = "regression"
output_folder = f"Output/Test_{mode}_{time_mark}"
train_sdf_folder = "../../../Data/OneM"

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
        "hidden": (64,),
        "dropout": 0,
        "use_bn": False,
        "actf": nn.LeakyReLU(),
    },
    "gcnn_params": {
        "hidden_pre_fc": (64,),
        "pre_fc_dropout": 0,
        "pre_fc_actf": nn.LeakyReLU(),
        "hidden_conv": (64,),
        "conv_dropout": 0,
        "conv_actf": nn.LeakyReLU(),
        "hidden_post_fc": (64,),
        "post_fc_dropout": 0,
        "post_fc_bn": False,
        "post_fc_actf": nn.LeakyReLU(),
        "conv_layer": MFConv,
        "conv_parameters": None,
        "graph_pooling": global_mean_pool
    },
    "post_fc_params": {
        "hidden": (64,),
        "dropout": 0,
        "use_bn": False,
        "actf": nn.LeakyReLU(),
    },
    "global_pooling": ConcatPooling,
}

test_metal = "Cu"  # sys.argv[1]

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

model = GCNNBimodal(
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
