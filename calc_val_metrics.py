import sys

from scipy.spatial.distance import cosine

sys.path.append("Source")

from Source.data import train_test_valid_split
import copy
import os

import torch
from sklearn.metrics import r2_score
from torch_geometric.data import Batch
from Source.models.GCNN_FCNN import MolGraphHeteroNet
from Source.featurizers.featurizers import featurize_sdf_with_metal, SkipatomFeaturizer, ConvMolFeaturizer


def create_path(path):
    if os.path.exists(path) or path == "":
        return
    head, tail = os.path.split(path)
    create_path(head)
    os.mkdir(path)


def get_metal(features):
    metal_by_similarity = {cosine(skipatom_features[metal], features): metal for metal in skipatom_features}
    best_similarity = min(metal_by_similarity.keys())
    return metal_by_similarity[best_similarity]


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
    if not os.path.exists(path_to_state): continue
    state_dict = torch.load(path_to_state)
    new_model = copy.deepcopy(base_model)
    new_model.load_state_dict(state_dict)
    new_model.eval()
    models += [new_model]

skipatom_features = torch.load("Data/skipatom_features.torch")

graphs = {metal: [] for metal in skipatom_features}
test_pred = {metal: [] for metal in skipatom_features}
test_true = {metal: [] for metal in skipatom_features}

for model, fold in zip(models, folds):
    for graph in fold[1].dataset:
        metal = get_metal(graph.metal_x)
        graphs[metal] += [graph]
    for metal in skipatom_features:
        if len(graphs[metal]) == 0: continue
        batch = Batch.from_data_list(graphs[metal])
        test_pred[metal] += model(
            batch.x,
            batch.edge_index,
            batch.metal_x,
            batch=batch.batch
        ).detach().reshape(-1, 1).tolist()
        test_true[metal] += batch.y.reshape(-1, 1).tolist()
for metal in skipatom_features:
    if len(test_true[metal]) == 0:
        print(f"{metal}: zero samples")
        continue
    print(f"{metal}: {r2_score(test_true[metal], test_pred[metal])}")
