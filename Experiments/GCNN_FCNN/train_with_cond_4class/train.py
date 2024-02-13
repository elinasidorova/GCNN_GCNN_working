import copy
import logging
import os.path
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef, f1_score
from torch import nn
from torch_geometric.nn import global_mean_pool, MFConv

sys.path.append(os.path.abspath("."))
from Source.data import train_test_valid_split
from Source.models.GCNN.trainer import GCNNTrainer
from Source.models.GCNN_FCNN.featurizers import SkipatomFeaturizer, featurize_sdf_with_metal_and_conditions
from Source.models.GCNN_FCNN.model import GCNN_FCNN
from Source.models.GCNN_FCNN.old_featurizer import ConvMolFeaturizer
from Source.models.global_poolings import MaxPooling
from config import ROOT_DIR


def classification_dataset(dataset, num_classes, thresholds=None):
    dataset = copy.deepcopy(dataset)
    all_values = np.array([graph.y["logK"].item() for graph in dataset])
    thresholds = thresholds or [np.percentile(all_values, (i + 1) * 100 / num_classes) for i in range(num_classes)]
    for graph in dataset:
        class_id = torch.tensor([num_classes - 1], dtype=torch.int64)
        for i, threshold in enumerate(thresholds[::-1]):
            if graph.y["logK"].item() < threshold:
                class_id = torch.tensor([num_classes - i - 1], dtype=torch.int64)
        graph.y = {"logK_class": class_id}
    return dataset, thresholds


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
time_mark = str(datetime.now()).replace(" ", "_").replace("-", "_").replace(":", "_").split(".")[0]

all_metals = ['Ac', 'Ag', 'Al', 'Am', 'Au', 'Ba', 'Be', 'Bi', 'Bk', 'Ca', 'Cd', 'Ce', 'Cf', 'Cm', 'Co', 'Cr', 'Cs',
              'Cu', 'Dy', 'Er', 'Eu', 'Fe', 'Ga', 'Gd', 'Hf', 'Hg', 'Ho', 'In', 'K', 'La', 'Li', 'Lu', 'Mg', 'Mn', 'Mo',
              'Na', 'Nd', 'Ni', 'Np', 'Pa', 'Pb', 'Pd', 'Pm', 'Pr', 'Pt', 'Pu', 'Rb', 'Re', 'Rh', 'Sb', 'Sc', 'Sm',
              'Sn', 'Sr', 'Tb', 'Th', 'Ti', 'Tl', 'Tm', 'U', 'V', 'Y', 'Yb', 'Zn', 'Zr']
test_metal = "Cu"

num_classes = 4
cv_folds = 5
seed = 23
batch_size = 64
epochs = 1000
es_patience = 100
mode = "regression"
train_sdf_folder = ROOT_DIR / "Data/OneM_cond"
output_folder = ROOT_DIR / f"Output/4class/{test_metal}_{cv_folds}fold_{mode}_{time_mark}"

targets = ({
               "name": "logK_class",
               "dim": num_classes,
               "mode": "multiclass_classification",
               "metrics": {
                   "confusion_matrix": (confusion_matrix, {}),
                   "accuracy": (accuracy_score, {}),
                   "f1": (f1_score, {"average": None}),
                   "matthews": (matthews_corrcoef, {}),
               },
               "loss": F.cross_entropy,
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

logging.info("Featurizing...")
dataset, _ = classification_dataset(
    featurize_sdf_with_metal_and_conditions(path_to_sdf=os.path.join(train_sdf_folder, f"{test_metal}.sdf"),
                                            mol_featurizer=ConvMolFeaturizer(),
                                            metal_featurizer=SkipatomFeaturizer()),
    num_classes=num_classes)

logging.info("Splitting...")
folds, test_loader = train_test_valid_split(dataset, n_folds=1,
                                            batch_size=batch_size,
                                            seed=seed)

model = GCNN_FCNN(
    metal_features=next(iter(test_loader)).metal_x.shape[-1],
    node_features=next(iter(test_loader)).x.shape[-1],
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
