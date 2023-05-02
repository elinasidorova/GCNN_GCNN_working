import copy
import os
import sys

import torch
from sklearn.metrics import r2_score
from torch import nn
from torch_geometric.data import Batch

from Source.models.GCNN_bimodal.GCNN_bimodal_trainer import GCNNTrainer
from Source.trainer import ModelShell

sys.path.append("Source")

from Source.models.GCNN_bimodal import GCNN_bimodal
from Source.featurizers.featurizers import featurize_sdf_with_metal, ConvMolFeaturizer, SkipatomFeaturizer


def create_path(path):
    if os.path.exists(path) or path == "":
        return
    head, tail = os.path.split(path)
    create_path(head)
    os.mkdir(path)


train_folder = "Output/General_MgCdLa_regression_2023_03_27_21_50_20"
train_sdf = "Data/OptunaMgCdLa/train.sdf"
path_to_config = os.path.join(train_folder, "model_config")

model = ModelShell(GCNN_bimodal, train_folder)
batch = Batch.from_data_list(featurize_sdf_with_metal(path_to_sdf=train_sdf,
                                                      mol_featurizer=ConvMolFeaturizer(),
                                                      metal_featurizer=SkipatomFeaturizer("Source/featurizers/skipatom_vectors_dim200.torch")
                                                      ))

trainer = GCNNTrainer(
    model=None,
    train_valid_data=folds,
    test_data=test_data,
    target_metrics=target_metrics,
)
trainer.models = super_model.models
result = trainer.calculate_metrics()

print(result)