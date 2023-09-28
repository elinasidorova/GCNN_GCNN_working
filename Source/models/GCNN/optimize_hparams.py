import json
import os
import sys
from datetime import datetime
from inspect import isfunction, isclass, signature

import optuna
import torch.nn as nn
from dgllife.utils import CanonicalAtomFeaturizer, AttentiveFPAtomFeaturizer, PAGTNAtomFeaturizer
from sklearn.metrics import r2_score, mean_absolute_error
from torch.optim import Adam
from torch_geometric.nn import MFConv, global_mean_pool, global_max_pool

sys.path += [os.path.abspath(".")]

from Source.data import root_mean_squared_error, train_test_valid_split
from Source.models.FCNN.optimize_hparams import GeneralParams as FCNNGeneralParams, FCNNParams
from Source.models.GCNN.featurizers import DGLFeaturizer, featurize_sdf
from Source.models.GCNN.model import GCNN
from Source.models.GCNN.trainer import GCNNTrainer
from Source.models.GCNN_FCNN.old_featurizer import ConvMolFeaturizer
from config import ROOT_DIR

sys.path.append(os.path.abspath("."))

time_mark = str(datetime.now()).replace(" ", "_").replace("-", "_").replace(":", "_").split(".")[0]

n_trials = 150
timeout = None  # 60 * 60 * 3

cv_folds = 1
batch_size = 64
epochs = 1000
es_patience = 100
seed = 42

mode = "regression"
general_output_path = f"Output/Optuna_BaselineGCNN_{time_mark}"
path_to_sdf = str(ROOT_DIR / "Data/OneM/Cu_2.sdf")
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


class GeneralParams(FCNNGeneralParams):
    def __init__(self, trial: optuna.Trial, optimizer_variants=None, lr_lims=(1e-5, 1e-1),
                 featurizer_variants=None):
        super(GeneralParams, self).__init__(trial, optimizer_variants, lr_lims, featurizer_variants)

    def get_featurizer(self):
        featurizer_name = self.trial.suggest_categorical("featurizer_name", list(self.featurizer_variants.keys()))
        featurizer = self.featurizer_variants[featurizer_name]

        if featurizer_name != "ConvMolFeaturizer":
            add_self_loop = self.trial.suggest_categorical("add_self_loop", (True, False))
            featurizer = DGLFeaturizer(add_self_loop=add_self_loop, node_featurizer=featurizer,
                                       require_edge_features=False)

        return featurizer


class GCNNParams:
    def __init__(self, trial: optuna.Trial,
                 pre_fc_params=None, post_fc_params=None,
                 n_conv_lims=(2, 10), dropout_lims=(0, 0.8),
                 actf_variants=None, hidden_lims=(32, 128, 32),
                 conv_layer_variants=None, pooling_layer_variants=None, prefix=""):
        self.trial = trial
        self.prefix = prefix
        self.pre_fc_params = pre_fc_params or {
            "dim_lims": (1, 5),
            "n_layers_lims": (1, 5),
            "actf_variants": None,
            "dropout_lims": (0, 0.8),
            "bn_variants": (True, False)
        }
        self.post_fc_params = post_fc_params or {
            "dim_lims": (1, 5),
            "n_layers_lims": (1, 5),
            "actf_variants": None,
            "dropout_lims": (0, 0.8),
            "bn_variants": (True, False)
        }
        self.conv_layer_variants = conv_layer_variants or {
            "MFConv": MFConv,
        }
        self.pooling_layer_variants = pooling_layer_variants or {
            "global_mean_pool": global_mean_pool,
        }
        self.n_layers_lims = n_conv_lims
        self.hidden_lims = hidden_lims
        self.dropout_lims = dropout_lims
        self.actf_variants = actf_variants or {
            "nn.ReLU()": nn.ReLU(),
            "nn.LeakyReLU()": nn.LeakyReLU(),
        }

    def get_conv_layer(self):
        self.conv_layer_name = self.trial.suggest_categorical(
            f"{self.prefix}_conv_layer_name",
            list(self.conv_layer_variants.keys())
        )
        return self.conv_layer_variants[self.conv_layer_name]

    def get_pooling_layer(self):
        self.pooling_layer_name = self.trial.suggest_categorical(
            f"{self.prefix}_pooling_layer_name",
            list(self.pooling_layer_variants.keys())
        )
        pooling_layer = self.pooling_layer_variants[self.pooling_layer_name]
        if isfunction(pooling_layer):
            return pooling_layer
        if isclass(pooling_layer) and "in_channels" in signature(pooling_layer).parameters:
            params = {"in_channels": self.hidden_lims[-1]}
            return pooling_layer(**params)

    def get_actf(self):
        self.actf_name = self.trial.suggest_categorical(
            f"{self.prefix}_actf_name",
            list(self.actf_variants.keys())
        )
        return self.actf_variants[self.actf_name]

    def get_conv_parameters(self):
        conv_parameters = {}
        if self.conv_layer.__name__ == "SSGConv":
            conv_parameters = {
                "alpha": 0.5,
            }
        return conv_parameters

    def get(self):
        self.n_layers = self.trial.suggest_int(f"{self.prefix}_n_layers", *self.n_layers_lims)
        self.conv_layer = self.get_conv_layer()
        self.conv_parameters = self.get_conv_parameters()
        self.dropout = self.trial.suggest_float(f"{self.prefix}_dropout", *self.dropout_lims)
        self.activation = self.get_actf()
        self.hidden = [self.trial.suggest_int(f"{self.prefix}_hidden_{i}", *self.hidden_lims)
                       for i in range(self.n_layers)]
        self.pooling_layer = self.get_pooling_layer()

        model_parameters = {
            "pre_fc_params": FCNNParams(self.trial, **self.pre_fc_params, prefix="pre_fc").get(),
            "post_fc_params": FCNNParams(self.trial, **self.post_fc_params, prefix="post_fc").get(),
            "hidden_conv": self.hidden,
            "conv_dropout": self.dropout,
            "conv_actf": self.activation,
            "conv_layer": self.conv_layer,
            "conv_parameters": self.conv_parameters,
            "graph_pooling": global_mean_pool
        }

        return model_parameters


def objective(trial: optuna.Trial):
    model_parameters = GCNNParams(
        trial,
        pre_fc_params={
            "dim_lims": (1, 5),
            "n_layers_lims": (1, 5),
            "actf_variants": None,
            "dropout_lims": (0, 0.8),
            "bn_variants": (True, False)
        },
        post_fc_params={
            "dim_lims": (1, 5),
            "n_layers_lims": (1, 5),
            "actf_variants": None,
            "dropout_lims": (0, 0.8),
            "bn_variants": (True, False)
        },
        n_conv_lims=(1, 3), dropout_lims=(0, 1),
        actf_variants=None, hidden_lims=(32, 512, 64),
        conv_layer_variants={
            "MFConv": MFConv,
            # "GATConv": GATConv,
            # "TransformerConv": TransformerConv,
            # "SAGEConv": SAGEConv,
            # "GCNConv": GCNConv,  # edge_weight
            # "ARMAConv": ARMAConv,  # edge_weight
            # "SSGConv": SSGConv,  # edge_weight
        }, pooling_layer_variants={
            "global_mean_pool": global_mean_pool,
            "global_max_pool": global_max_pool,
        },
    ).get()

    general_parameters = GeneralParams(
        trial,
        optimizer_variants={"Adam": Adam},
        lr_lims=(1e-5, 1e-1),
        featurizer_variants={
            "ConvMolFeaturizer": ConvMolFeaturizer(),
            "CanonicalAtomFeaturizer": CanonicalAtomFeaturizer(),
            "AttentiveFPAtomFeaturizer": AttentiveFPAtomFeaturizer(),
            "PAGTNAtomFeaturizer": PAGTNAtomFeaturizer(),
        },
    ).get()

    data = featurize_sdf(path_to_sdf=path_to_sdf, mol_featurizer=general_parameters["featurizer"], seed=42)
    folds, test_data = train_test_valid_split(data, n_folds=cv_folds, test_ratio=0.1, batch_size=batch_size)

    model_parameters["node_features"] = test_data.dataset[0].x.shape[-1]
    model_parameters["targets"] = targets
    model_parameters["optimizer"] = general_parameters["optimizer"]
    model_parameters["optimizer_parameters"] = general_parameters["optimizer_parameters"]

    model = GCNN(**model_parameters)

    time_mark = str(datetime.now()).replace(" ", "_").replace("-", "_").replace(":", "_").split(".")[0]
    output_folder = os.path.join(general_output_path, f"GCNN_{mode}_{time_mark}")

    trainer = GCNNTrainer(
        model=model,
        train_valid_data=folds,
        test_data=test_data,
        output_folder=output_folder,
        epochs=epochs,
        es_patience=es_patience,
        targets=targets,
        seed=seed,
    )

    trainer.train_cv_models()

    trial.set_user_attr(key="metrics", value=trainer.results_dict["general"])
    trial.set_user_attr(key="model_parameters", value={key: str(model_parameters[key]) for key in model_parameters})

    errors = [trainer.results_dict["general"][key] for key in trainer.results_dict["general"] if "MSE" in key]
    return max(errors)


def callback(study: optuna.Study, trial):
    study.trials_dataframe().to_csv(path_or_buf=os.path.join(general_output_path, f"trials.csv"), index=False)


if __name__ == "__main__":
    start = datetime.now()
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, callbacks=[callback])
    end = datetime.now()

    result = {
        "trials": len(study.trials),
        "started": str(start).split(".")[0],
        "finished": str(end).split(".")[0],
        "duration": str(end - start).split(".")[0],
        **study.best_trial.user_attrs["metrics"],
        "model_parameters": study.best_trial.user_attrs["model_parameters"],
    }

    time_mark = str(start).replace(" ", "_").replace("-", "_").replace(":", "_").split(".")[0]

    study.trials_dataframe().to_csv(path_or_buf=os.path.join(general_output_path, f"trials.csv"), index=False)

    with open(os.path.join(general_output_path, f"result.json"), "w") as jf:
        json.dump(result, jf)
