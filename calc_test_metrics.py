import copy
import os
import sys

import torch
from sklearn.metrics import r2_score
from torch import nn
from torch_geometric.data import Batch

sys.path.append("Source")

from Source.models.GCNN_bimodal import MolGraphHeteroNet
from Source.featurizers.featurizers import featurize_sdf_with_metal, ConvMolFeaturizer, SkipatomFeaturizer


def create_path(path):
    if os.path.exists(path) or path == "":
        return
    head, tail = os.path.split(path)
    create_path(head)
    os.mkdir(path)


class SuperModel(nn.Module):
    def __init__(self, model_class, train_folder, device=torch.device("cpu")):
        super().__init__()
        self.models = []
        self.device = device
        path_to_config = os.path.join(train_folder, "model_config")
        model = model_class(**torch.load(path_to_config))
        for folder_name in os.listdir(train_folder):
            if folder_name.startswith("fold_"):
                path_to_state = os.path.join(train_folder, folder_name, "best_model")
                state_dict = torch.load(path_to_state, map_location=device)
                new_model = copy.deepcopy(model)
                new_model.load_state_dict(state_dict)
                new_model.eval()
                new_model.to(device)
                self.models += [new_model]

    def forward(self, **kwargs):
        outs = torch.cat([model(**kwargs).unsqueeze(-1) for model in self.models], dim=-1).mean(dim=-1)
        return outs


train_folder = "Output/General_MgCdLa_regression_2023_03_27_21_50_20"
train_sdf = "Data/OptunaMgCdLa/train.sdf"
path_to_config = os.path.join(train_folder, "model_config")

model = SuperModel(MolGraphHeteroNet, train_folder)
batch = Batch.from_data_list(featurize_sdf_with_metal(path_to_sdf=train_sdf,
                                                      mol_featurizer=ConvMolFeaturizer(),
                                                      metal_featurizer=SkipatomFeaturizer("Source/featurizers/skipatom_vectors_dim200.torch")
                                                      ))
test_pred = model(x=batch.x, edge_index=batch.edge_index, metal_x=batch.metal_x, batch=batch.batch).detach().numpy().reshape(-1, 1)
test_true = batch.y.reshape(-1, 1).numpy()
print(r2_score(test_true, test_pred))

# # #

metrics = {}
for metal in ["Mg", "Cd", "La"]:
    test_sdf = f"Data/OptunaMgCdLa/{metal}_val.sdf"
    batch = Batch.from_data_list(featurize_sdf_with_metal(path_to_sdf=test_sdf))
    test_pred = model(x=batch.x, edge_index=batch.edge_index, metal_x=batch.metal_x, batch=batch.batch).detach().numpy().reshape(-1, 1)
    test_true = batch.y.reshape(-1, 1).numpy()
    metrics[metal] = r2_score(test_true, test_pred)

print("val", metrics)

metrics = {}
for metal in ["Mg", "Cd", "La"]:
    test_sdf = f"Data/OptunaMgCdLa/{metal}_test.sdf"
    batch = Batch.from_data_list(featurize_sdf_with_metal(path_to_sdf=test_sdf))
    test_pred = model(x=batch.x, edge_index=batch.edge_index, metal_x=batch.metal_x, batch=batch.batch).detach().numpy().reshape(-1, 1)
    test_true = batch.y.reshape(-1, 1).numpy()
    metrics[metal] = r2_score(test_true, test_pred)

print("test", metrics)
