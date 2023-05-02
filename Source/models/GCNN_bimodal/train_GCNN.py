import logging
import os.path
import sys
from datetime import datetime

import numpy as np
import torch
from dgllife.utils import CanonicalAtomFeaturizer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from torch import nn
from torch_geometric.nn import global_mean_pool, MFConv

from Source.featurizers.featurizers import featurize_sdf_with_metal, SkipatomFeaturizer
from Source.models.global_poolings import ConcatPooling

sys.path.append(os.path.abspath("."))

from GCNN_model import GCNN
from GCNN_trainer import GCNNTrainer
from Source.data import get_num_node_features, train_test_valid_split, get_num_metal_features
from Source.featurizers import featurize_csv_to_graph, DGLFeaturizer

time_mark = str(datetime.now()).replace(" ", "_").replace("-", "_").replace(":", "_").split(".")[0]

cv_folds = 5
batch_size = 32
epochs = 1000
es_patience = 100
mode = "regression"
output_folder = f"Output/TestGCNN_IC50_{mode}_{time_mark}"
train_sdf = "Data/OneM"
test_sdf = ""
test_metal = sys.argv[1]

transition_metals = ["Mg", "Al", "Ca", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Y", "Ag", "Cd", "Hg", "Pb", "Bi"]
Ln_metals = ["La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]
Ac_metals = ["Th", "Am", "Cm", "Bk", "Cf"]  # "U", "Pu",
all_metals = transition_metals + Ln_metals + Ac_metals

max_data = None
targets = ("logK",)
target_metrics = {
    target: {
        "R2": (r2_score, {}),
        "RMSE": (lambda *args, **kwargs: np.sqrt(mean_squared_error(*args, **kwargs)), {}),
        "MAE": (mean_absolute_error, {})
    } for target in targets
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.info("Featurizig...")

balanced_train_test_valid_split

test_loader = DataLoader(featurize_sdf_with_metal(
    path_to_sdf=test_sdf,
    mol_featurizer=ConvMolFeaturizer(),
    metal_featurizer=SkipatomFeaturizer("Source/featurizers/skipatom_vectors_dim200.json")),
    batch_size=batch_size)

model = MolGraphHeteroNet(
    node_features=get_num_node_features(folds[0][0]),
    metal_features=get_num_metal_features(folds[0][0]),
    num_targets=get_num_targets(folds[0][0]),
    batch_size=get_batch_size(folds[0][0]),
    **model_parameters,
)
trainer = MolGraphHeteroNetTrainer(
    model=model,
    train_valid_data=folds,
    test_data=test_loader,
    output_folder=output_path,
    out_folder_mark=output_mark,
    epochs=epochs,
    es_patience=es_patience,
    seed=seed)
trainer.train_cv_models()

data = featurize_csv_to_graph(path_to_csv=path_to_csv,
                              targets=targets,
                              featurizer=DGLFeaturizer(add_self_loop=True,
                                                       node_featurizer=CanonicalAtomFeaturizer()),
                              max_data=max_data)

logging.info("Splitting...")
folds, test_data = train_test_valid_split(data, n_splits=cv_folds, test_ratio=0.1, batch_size=batch_size)
logging.info("Data preprocessing done. Starting training...")

model_parameters = {
    "hidden_metal_fc": (64,),
    "metal_fc_dropout": 0,
    "metal_fc_bn": False,
    "metal_fc_actf": nn.LeakyReLU(),
    "hidden_pre_fc": (),
    "pre_fc_dropout": None,
    "pre_fc_actf": None,
    "hidden_conv": (64,),
    "conv_dropout": 0,
    "conv_actf": nn.LeakyReLU(),
    "hidden_post_fc": (64,),
    "post_fc_dropout": 0,
    "post_fc_bn": False,
    "post_fc_actf": nn.LeakyReLU(),
    "conv_layer": MFConv,
    "conv_parameters": None,
    "graph_pooling": global_mean_pool,
    "global_pooling": ConcatPooling,
    "optimizer": torch.optim.Adam,
    "optimizer_parameters": None,
    "mode": "regression"
}

model = GCNN(node_features=get_num_node_features(folds[0][0]),
             metal_features=get_num_metal_features(folds[0][0]),
             num_targets=len(targets),
             batch_size=batch_size,
             mode="regression",
             **model_parameters)

trainer = GCNNTrainer(
    model=model,
    train_valid_data=folds,
    test_data=test_data,
    output_folder=output_folder,
    epochs=epochs,
    es_patience=es_patience,
    target_metrics=target_metrics,
)

trainer.train_cv_models()
