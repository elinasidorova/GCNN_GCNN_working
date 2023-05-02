import copy
import os
import sys

import torch
from skipatom import SkipAtomInducedModel
from sklearn.metrics import r2_score
from torch import nn
from torch_geometric.data import Batch

sys.path.append("Source")

from Source.model import MolGraphHeteroNet
from Source.mol_featurizer import featurize_sdf_with_metal, ConvMolFeaturizer, SkipatomFeaturizer


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
        config = torch.load(os.path.join(train_folder, "model_config"))
        config["linear_bn"] = False
        model = model_class(**config)
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


train_folder = "Output/GeneralModel/General_Bi_seed25_testonly_10folds_regression_2023_03_12_14_45_44"
test_sdf = "Data/OneM/U.sdf"
model = SuperModel(MolGraphHeteroNet, train_folder)

batch = Batch.from_data_list(featurize_sdf_with_metal(path_to_sdf=test_sdf,
                                                      mol_featurizer=ConvMolFeaturizer(),
                                                      metal_featurizer=SkipatomFeaturizer(models=[
                                                          SkipAtomInducedModel.load(
                                                              "skipatom_models/AmBkCfCm_2022_11_23.dim200.model",
                                                              "skipatom_models/AmBkCfCm_2022_11_23.training.data",
                                                              min_count=2e7, top_n=5
                                                          ),
                                                          SkipAtomInducedModel.load(
                                                              "skipatom_models/mp_2020_10_09.dim200.model",
                                                              "skipatom_models/mp_2020_10_09.training.data",
                                                              min_count=2e7, top_n=5
                                                          ),
                                                      ])
                                                      ))
test_pred = model(x=batch.x, edge_index=batch.edge_index, metal_x=batch.metal_x, batch=batch.batch).detach().numpy().reshape(-1, 1)
test_true = batch.y.reshape(-1, 1).numpy()
print(r2_score(test_true, test_pred))
