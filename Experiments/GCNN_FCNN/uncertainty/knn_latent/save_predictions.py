import logging
import os.path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from torch import nn
from torch_geometric.nn import MFConv, global_mean_pool
from tqdm import tqdm

from Source.models.GCNN_FCNN.model import GCNN_FCNN
from Source.models.GCNN_FCNN.old_featurizer import ConvMolFeaturizer
from Source.models.global_poolings import MaxPooling
from config import ROOT_DIR

sys.path.append(os.path.abspath("."))

from Source.trainer import ModelShell
from Source.data import root_mean_squared_error, balanced_train_valid_test_split
from Source.models.GCNN_FCNN.featurizers import SkipatomFeaturizer, featurize_sdf_with_metal_and_conditions

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

other_metals = ['Li', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                'Ga', 'Rb', 'Sr', 'Y', 'Zr', 'Mo', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Cs', 'Ba', 'Hf', 'Re',
                'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi']
Ln_metals = ['La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', ]
Ac_metals = ['Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf']

train_metals = list(set(["Y", "Sc"] + Ln_metals + Ac_metals) - {"Ac", "Pa"})

cv_folds = 1
test_size = 0.1
seed = 23
batch_size = 64
epochs = 1000
es_patience = 100
mode = "regression"
train_sdf_folder = ROOT_DIR / "Data/OneM_cond_adds"
output_dir = ROOT_DIR / f"Output/Uncertainty/knn_latent"
train_folder = ROOT_DIR / f"Output/Uncertainty_knn_latent/1fold_regression_2023_08_04_13_29_56"

targets = ({
               "name": "logK",
               "mode": "regression",
               "dim": 1,
               "metrics": {
                   "R2": (r2_score, {}),
                   "RMSE": (root_mean_squared_error, {}),
                   "MAE": (mean_absolute_error, {})
               },
               "loss": nn.MSELoss(),
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

train_datasets = [featurize_sdf_with_metal_and_conditions(path_to_sdf=os.path.join(train_sdf_folder, f"{metal}.sdf"),
                                                          mol_featurizer=ConvMolFeaturizer(),
                                                          metal_featurizer=SkipatomFeaturizer())
                  for metal in tqdm(train_metals, desc="Featurizig")]
logging.info("Splitting...")
folds, test_loader = balanced_train_valid_test_split(train_datasets, n_folds=cv_folds,
                                                     batch_size=batch_size,
                                                     test_size=test_size,
                                                     shuffle_every_epoch=True,
                                                     seed=seed)
train_loader, val_loader = folds[0]

super_model = ModelShell(GCNN_FCNN, train_folder)
model = super_model.models[0]

os.makedirs(ROOT_DIR / "Output/Uncertainty/knn_latent/", exist_ok=True)
for loader, name in zip((train_loader, val_loader, test_loader), ("train", "valid", "test")):
    if os.path.exists(output_dir / f"{name}_latents.json"): continue
    data = []
    for batch in loader.dataset:
        result, latent = model(batch, return_latent=True)
        data += [{"latent": latent.squeeze().detach().tolist(), "true_logK": batch.y["logK"].item(),
                  "pred_logK": result["logK"].item()}]
    df = pd.DataFrame(data)
    df.to_json(output_dir / f"{name}_latents.json")

#####################################

k = 8


def distance(v1, v2):
    return np.linalg.norm(np.array(v1) - np.array(v2))


train_df = pd.read_json(output_dir / "train_latents.json")

for name in ("train", "valid", "test"):
    data = []
    df = pd.read_json(output_dir / f"{name}_latents.json")
    for i in df.index:
        distances = [distance(df["latent"][i], train_vector) for train_vector in train_df["latent"].tolist()]
        uncertainty = np.mean(np.sort(distances)[:k])
        data += [{"true_logK": df["true_logK"][i],
                  "pred_logK": df["pred_logK"][i],
                  "pred_logK_std": uncertainty}]
    pd.DataFrame(data).to_json(output_dir / f"{name}.json")

    print(f"{name} done")

print("All done")
