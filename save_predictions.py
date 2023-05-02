import os

import numpy as np
import torch
from sklearn.metrics import r2_score
from torch_geometric.data import Batch

from Source.models.GCNN_bimodal import MolGraphHeteroNet
from Source.featurizers.featurizers import featurize_sdf_with_metal, SkipatomFeaturizer, ConvMolFeaturizer

transition_metals = ["Mg", "Al", "Ca", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Y", "Ag", "Cd", "Hg", "Pb", "Bi"]
Ln_metals = ["La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]
Ac_metals = ["Th", "Am", "Cm", "Bk", "Cf"] # "U", "Pu",
all_metals = transition_metals + Ln_metals + Ac_metals

for metal in all_metals:
    if os.path.exists(f"Output/GeneralModel_predictions/{metal}_predictions.torch"): continue
    train_folder = None
    for folder in os.listdir("Output/GeneralModel"):
        if folder.startswith(f"General_{metal}") and os.path.exists(f"Output/GeneralModel/{folder}/metrics.json"):
            train_folder = f"Output/GeneralModel/{folder}"
            break
    if train_folder is None: continue
    try:
        path_to_config = os.path.join(train_folder, "model_config")
        config = torch.load(path_to_config)
    except Exeption as e:
        continue
    models = []
    for i in range(10):
        path_to_state = os.path.join(train_folder, f"fold_{i + 1}", "best_model")
        model = MolGraphHeteroNet(**config)
        state_dict = torch.load(path_to_state)

        model.load_state_dict(state_dict)
        model.eval()
        models += [model]

    featurized_test = featurize_sdf_with_metal(path_to_sdf=f"Data/GeneralModel/{metal}_testonly_test.sdf",
                                               mol_featurizer=ConvMolFeaturizer(),
                                               metal_featurizer=SkipatomFeaturizer("Source/featurizers/skipatom_vectors_dim200.torch"))

    batch = Batch.from_data_list(featurized_test)

    mean_test_pred = np.mean(np.array([
        model.forward(batch.x, batch.edge_index, batch.metal_x, batch=batch.batch).detach().numpy().reshape(-1, 1)
        for model in models]), axis=0)
    test_true = batch.y.reshape(-1, 1).numpy()
    print(metal, r2_score(test_true, mean_test_pred))
    torch.save((mean_test_pred, test_true), f"Output/GeneralModel_predictions/{metal}_predictions.torch")
