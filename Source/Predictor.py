import json
import torch
import os
from Source.models.GCNN_bimodal.GCNN_bimodal import MolGraphNet
import pandas as pd
from torch_geometric.data import Batch


def process_predict_input(input_data):
    return input_data


class Predictor:
    def __init__(self, model_folder, model_class=MolGraphNet):
        self.model_class = model_class
        self.restore_params = {}
        self.model_folder = model_folder
        self.folds = []
        self.mols_num = None
        self.mols = []
        self.valuename = None
        self.featurized_data = None
        self.predicted_data = pd.DataFrame()
        self.load_models()

    def load_models(self):
        fold_folders = [os.path.join(self.model_folder, i) for i in os.listdir(self.model_folder) if i.startswith("fold")]
        with open(os.path.join(self.model_folder, "model_structure.json")) as jf:
            model_arch = json.load(jf)

        for fold in fold_folders:
            model = self.model_class((model_arch["node_features"], model_arch["num_targets"]))
            model.load_state_dict(
                torch.load(os.path.join(fold, "best_model")))
            self.folds.append(model)

    @staticmethod
    def single_fold_predict(model, features):
        batch = Batch.from_data_list(features.dataset)
        X, y = batch, batch.y
        x, edge_index, batch = X.x, X.edge_index, X.batch
        return model.forward(x, edge_index, batch).detach().numpy().flatten()

    def predict(self, input_data):
        input_data = process_predict_input(input_data)
        results = {}
        for i, fold in enumerate(self.folds):
            results[i + 1] = self.single_fold_predict(fold, input_data)

        return results

