import sys

import numpy as np

sys.path.append("Source")

from Source.data import train_test_valid_split
import copy
import os

import torch
from sklearn.metrics import r2_score
from torch import nn
from torch_geometric.data import Batch
from Source.models.GCNN_FCNN import MolGraphHeteroNet
from Source.featurizers.featurizers import featurize_sdf_with_metal, SkipatomFeaturizer, ConvMolFeaturizer


def create_path(path):
    if os.path.exists(path) or path == "":
        return
    head, tail = os.path.split(path)
    create_path(head)
    os.mkdir(path)


class SuperModel(nn.Module):
    def __init__(self, train_folder, device=torch.device("cpu")):
        super().__init__()
        self.models = []
        self.device = device
        path_to_config = os.path.join(train_folder, "model_config")
        model = MolGraphHeteroNet(**torch.load(path_to_config))
        for folder_name in os.listdir(train_folder):
            if folder_name.startswith("fold_"):
                path_to_state = os.path.join(train_folder, folder_name, "best_model")
                state_dict = torch.load(path_to_state, map_location=device)
                new_model = copy.deepcopy(model)
                new_model.load_state_dict(state_dict)
                new_model.eval()
                new_model.to(device)
                self.models += [new_model]

    def forward(self, x, edge_index, metal_x, batch=None):
        outs = torch.cat([model(x, edge_index, metal_x, batch=batch).unsqueeze(-1) for model in self.models], dim=-1)
        return outs.mean(dim=-1)


n_split = 10
batch_size = 64
train_sdf = "Data/GeneralModel_train.sdf"
test_sdf = "Data/GeneralModel_test.sdf"

featurized_train = featurize_sdf_with_metal(path_to_sdf=train_sdf,
                                            mol_featurizer=ConvMolFeaturizer(),
                                            metal_featurizer=SkipatomFeaturizer("Source/featurizers/skipatom_vectors_dim200.torch"))
folds = train_test_valid_split(featurized_train, n_split, test_ratio=0.1, batch_size=batch_size,
                               subsample_size=False, return_test=False)
train_folder = "Output/GeneralModel_1_seed15_regression_2023_03_09_10_14_27"
path_to_config = os.path.join(train_folder, "model_config")
base_model = MolGraphHeteroNet(**torch.load(path_to_config))
models = []
for i in range(len(folds)):
    folder_name = f"fold_{i+1}"
    path_to_state = os.path.join(train_folder, folder_name, "best_model")
    state_dict = torch.load(path_to_state)
    new_model = copy.deepcopy(base_model)
    new_model.load_state_dict(state_dict)
    new_model.eval()
    models += [new_model]

test_preds = []
test_trues = []
for i, (model, fold) in enumerate(zip(models, folds)):
    batch = Batch.from_data_list(fold[1].dataset)
    test_preds += [model(batch.x, batch.edge_index, batch.metal_x, batch=batch.batch).detach().numpy().reshape(-1, 1)]
    test_trues += [batch.y.reshape(-1, 1).numpy()]
    print(f"fold_{i+1} va_r2: {r2_score(test_preds[-1], test_trues[-1])}")
test_true = np.concatenate(test_trues, axis=0)
test_pred = np.concatenate(test_preds, axis=0)
print(f"general val_r2: {r2_score(test_true, test_pred)}")
