import json
import os
from datetime import datetime
from inspect import isfunction, isclass, signature

import optuna
import torch
import torch.nn as nn
import torch.utils.data
from sklearn.metrics import mean_squared_error
from torch.optim import Adam, AdamW, RMSprop
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MFConv, global_mean_pool, global_max_pool

from Source.data import get_num_node_features, get_num_metal_features, get_num_targets, \
    get_batch_size
from Source.metal_ligand_concat import mean_unifunc, max_unifunc
from Source.models.GCNN_bimodal import MolGraphHeteroNet
from Source.featurizers.featurizers import featurize_sdf_with_metal, ConvMolFeaturizer, SkipatomFeaturizer
from Source.trainer import MolGraphHeteroNetTrainer

train_sdf = "Data/OptunaMgCdLa/train_trans.sdf"
test_sdfs = ["Data/OptunaMgCdLa/Mg_val.sdf", "Data/OptunaMgCdLa/Cd_val.sdf"]
output_path = "Output"
output_mark = f"Optuna_MgCd"
# valuenames = ["logK_Cu"]

n_trials = 150
batch_size = 128
epochs = 1000
es_patience = 100


class DictParams:
    def __init__(self, params):
        self.params = params

    def get_optimizer_parameters(self):
        optimizer_parameters = {}
        # if self.params["optimizer_name"] == "Adam":
        #     optimizer_parameters = {
        #         "lr": self.params["lr"],
        #         "betas": (self.params["betas_0"],
        #                   self.params["betas_1"]),
        #     }
        return optimizer_parameters

    def get_hidden_dims(self):
        hidden_conv = [self.params["conv_features"]] + \
                      [self.params[f"conv_layer{i}"]
                       for i in range(self.params["n_conv"] - 1)]
        hidden_linear = [hidden_conv[-1]] + \
                        [self.params[f"linear_layer{i}"]
                         for i in range(self.params["n_linear"] - 1)]

        return hidden_conv, hidden_linear

    def get(self):
        hidden_conv, hidden_linear = self.get_hidden_dims()
        optimizer_parameters = self.get_optimizer_parameters()

        model_parameters = {
            "n_conv": self.params["n_conv"],
            "n_linear": self.params["n_linear"],

            "hidden_conv": hidden_conv,
            "hidden_linear": hidden_linear,

            "conv_dropout": self.params["conv_dropout"],
            "linear_dropout": self.params["linear_dropout"],

            "conv_layer": self.params["conv_layer_name"],
            "pooling_layer": self.params["pooling_layer_name"],
            "conv_actf": self.params["conv_actf_name"],
            "linear_actf": self.params["linear_actf_name"],
            "linear_bn": self.params["linear_bn"],
            "optimizer": self.params["optimizer_name"],
            "optimizer_parameters": optimizer_parameters,
        }

        return model_parameters


class DictHeteroParams(DictParams):
    def get_hidden_dims(self):
        hidden_metal = [self.params["metal_features"]] + \
                       [self.params[f"metal_layer{i}"]
                        for i in range(self.params["n_metal"] - 1)]

        if self.params["metal_ligand_unifunc_name"] in ["concat"]:
            hidden_conv = [self.params["conv_features"]] + \
                          [self.params[f"conv_layer{i}"]
                           for i in range(self.params["n_conv"] - 1)]
            hidden_linear = [hidden_metal[-1] + hidden_conv[-1]] + \
                            [self.params[f"linear_layer{i}"]
                             for i in range(self.params["n_linear"] - 1)]
        elif self.params["metal_ligand_unifunc_name"] in ["max", "sum", "mean"]:
            hidden_conv = [self.params["conv_features"]] + \
                          [self.params[f"conv_layer{i}"]
                           for i in range(self.params["n_conv"] - 2)] + \
                          [hidden_metal[-1]]
            hidden_linear = [hidden_metal[-1]] + \
                            [self.params[f"linear_layer{i}"]
                             for i in range(self.params["n_linear"] - 1)]
        else:
            raise ValueError(f"Unknown metal_ligand_unifunc_name: '{self.params['metal_ligand_unifunc_name']}'")

        return hidden_metal, hidden_conv, hidden_linear

    def get(self):
        hidden_metal, hidden_conv, hidden_linear = self.get_hidden_dims()
        optimizer_parameters = self.get_optimizer_parameters()

        model_parameters = {
            "n_metal": self.params["n_metal"],
            "n_conv": self.params["n_conv"],
            "n_linear": self.params["n_linear"],

            "hidden_metal": hidden_metal,
            "hidden_conv": hidden_conv,
            "hidden_linear": hidden_linear,

            "metal_dropout": self.params["metal_dropout"],
            "conv_dropout": self.params["conv_dropout"],
            "linear_dropout": self.params["linear_dropout"],

            "metal_ligand_unifunc": self.params["metal_ligand_unifunc_name"],
            "conv_layer": self.params["conv_layer_name"],
            "pooling_layer": self.params["pooling_layer_name"],
            "conv_actf": self.params["conv_actf_name"],
            "linear_actf": self.params["linear_actf_name"],
            "linear_bn": self.params["linear_bn"],
            "optimizer": self.params["optimizer_name"],
            "optimizer_parameters": optimizer_parameters,
        }

        return model_parameters


class MolGraphNetParams:
    def __init__(self, trial: optuna.Trial, conv_features,
                 n_conv_lims=(2, 10), n_linear_lims=(2, 10),
                 conv_dropout_lims=(0, 0.8), linear_dropout_lims=(0, 0.8),
                 conv_layer_variants=None, pooling_layer_variants=None,
                 conv_actf_variants=None, linear_actf_variants=None,
                 optimizer_variants=None, hidden_variants=(64, 128, 256, 512), linear_bn_variants=None):

        self.trial = trial
        self.conv_features = conv_features
        self.n_conv_lims = n_conv_lims
        self.n_linear_lims = n_linear_lims
        self.conv_dropout_lims = conv_dropout_lims
        self.linear_dropout_lims = linear_dropout_lims
        self.hidden_variants = hidden_variants

        if conv_actf_variants is None:
            conv_actf_variants = {
                "nn.ReLU()": nn.ReLU(),
                "nn.LeakyReLU()": nn.LeakyReLU(),
            }
        self.conv_actf_variants = conv_actf_variants

        if linear_actf_variants is None:
            linear_actf_variants = {
                "nn.ReLU()": nn.ReLU(),
                "nn.LeakyReLU()": nn.LeakyReLU(),
            }
        self.linear_actf_variants = linear_actf_variants

        if optimizer_variants is None:
            optimizer_variants = {
                "Adam": Adam,
            }
        self.optimizer_variants = optimizer_variants

        if pooling_layer_variants is None:
            pooling_layer_variants = {
                "global_mean_pool": global_mean_pool,
            }
        self.pooling_layer_variants = pooling_layer_variants

        if conv_layer_variants is None:
            conv_layer_variants = {
                "MFConv": MFConv,
            }
        self.conv_layer_variants = conv_layer_variants

        if linear_bn_variants is None:
            linear_bn_variants = [False]
        self.linear_bn_variants = linear_bn_variants

        self.n_conv = None
        self.n_linear = None

        self.hidden_conv = None
        self.hidden_linear = None

        self.conv_dropout = None
        self.linear_dropout = None

        self.linear_actf = None
        self.linear_actf_name = None
        self.conv_actf = None
        self.conv_actf_name = None
        self.conv_layer = None
        self.conv_layer_name = None
        self.pooling_layer = None
        self.pooling_layer_name = None
        self.optimizer = None
        self.optimizer_name = None

        self.linear_bn = None
        self.optimizer_parameters = None

    def get_hidden_dims(self):

        hidden_conv = [self.conv_features] + \
                      [self.trial.suggest_categorical(f"conv_layer{i}", self.hidden_variants)
                       for i in range(self.n_conv)]

        hidden_linear = [hidden_conv[-1]] + \
                        [self.trial.suggest_categorical(f"linear_layer{i}", self.hidden_variants)
                         for i in range(self.n_linear)]

        return hidden_conv, hidden_linear

    def get_conv_layer(self):
        self.conv_layer_name = self.trial.suggest_categorical(
            "conv_layer_name",
            list(self.conv_layer_variants.keys())
        )
        return self.conv_layer_variants[self.conv_layer_name]

    def get_pooling_layer(self):
        self.pooling_layer_name = self.trial.suggest_categorical(
            "pooling_layer_name",
            list(self.pooling_layer_variants.keys())
        )
        pooling_layer = self.pooling_layer_variants[self.pooling_layer_name]
        if isfunction(pooling_layer):
            return pooling_layer
        if isclass(pooling_layer) and "in_channels" in signature(pooling_layer).parameters:
            params = {"in_channels": self.hidden_conv[-1]}
            return pooling_layer(**params)

    def get_conv_actf(self):
        self.conv_actf_name = self.trial.suggest_categorical(
            "conv_actf_name",
            list(self.conv_actf_variants.keys())
        )
        return self.conv_actf_variants[self.conv_actf_name]

    def get_linear_actf(self):
        self.linear_actf_name = self.trial.suggest_categorical(
            "linear_actf_name",
            list(self.linear_actf_variants.keys())
        )
        return self.linear_actf_variants[self.linear_actf_name]

    def get_optimizer(self):
        self.optimizer_name = self.trial.suggest_categorical(
            "optimizer_name",
            list(self.optimizer_variants.keys())
        )
        return self.optimizer_variants[self.optimizer_name]

    def get_optimizer_parameters(self):
        optimizer_parameters = {}
        # if self.optimizer_name == "Adam":
        #     optimizer_parameters = {
        #         "lr": self.trial.suggest_float("lr", 1e-4, 1e-3),
        #         "betas": (self.trial.suggest_float("betas_0", 0.001, 0.999),
        #                   self.trial.suggest_float("betas_1", 0.001, 0.999))
        #     }
        return optimizer_parameters

    def get(self):
        self.n_conv = self.trial.suggest_int("n_conv", *self.n_conv_lims)
        self.n_linear = self.trial.suggest_int("n_linear", *self.n_linear_lims)

        self.conv_dropout = self.trial.suggest_float("conv_dropout", *self.conv_dropout_lims)
        self.linear_dropout = self.trial.suggest_float("linear_dropout", *self.linear_dropout_lims)
        self.linear_bn = self.trial.suggest_categorical("linear_bn", self.linear_bn_variants)
        self.conv_layer = self.get_conv_layer()
        self.conv_actf = self.get_conv_actf()
        self.linear_actf = self.get_linear_actf()

        self.hidden_conv, self.hidden_linear = self.get_hidden_dims()
        self.pooling_layer = self.get_pooling_layer()

        self.optimizer = self.get_optimizer()
        self.optimizer_parameters = self.get_optimizer_parameters()

        model_parameters = {
            "hidden_conv": self.hidden_conv,
            "hidden_linear": self.hidden_linear,
            "conv_layer": self.conv_layer,
            "pooling_layer": self.pooling_layer,
            "conv_dropout": self.conv_dropout,
            "linear_dropout": self.linear_dropout,
            "linear_bn": self.linear_bn,
            "conv_actf": self.conv_actf,
            "linear_actf": self.linear_actf,
            "optimizer": self.optimizer,
            "optimizer_parameters": self.optimizer_parameters,
        }

        return model_parameters


class MolGraphHeteroNetParams(MolGraphNetParams):
    def __init__(self, trial: optuna.Trial, metal_features, conv_features,
                 n_metal_lims=(2, 10), metal_dropout_lims=(0, 0.8), metal_bn_variants=(True, False),
                 metal_ligand_unifunc_variants=None, **kwargs):
        super(MolGraphHeteroNetParams, self).__init__(trial, conv_features, **kwargs)

        self.metal_features = metal_features
        self.n_metal_lims = n_metal_lims
        self.metal_dropout_lims = metal_dropout_lims
        self.metal_bn_variants = metal_bn_variants

        if metal_ligand_unifunc_variants is None:
            metal_ligand_unifunc_variants = {
                "sum": lambda x1, x2: x1 + x2,
                "max": lambda x1, x2: torch.max(x1, x2),
                "concat": lambda x1, x2: torch.cat((x1, x2), dim=1),
            }
        self.metal_ligand_unifunc_variants = metal_ligand_unifunc_variants

        self.n_metal = None
        self.hidden_metal = None
        self.metal_dropout = None
        self.metal_bn = None
        self.metal_ligand_unifunc = None
        self.metal_ligand_unifunc_name = None

    def get_hidden_dims(self):

        hidden_metal = [self.metal_features] + \
                       [self.trial.suggest_categorical(f"metal_layer{i}", self.hidden_variants)
                        for i in range(self.n_metal)]

        if self.metal_ligand_unifunc_name in ["concat"]:
            hidden_conv = [self.conv_features] + \
                          [self.trial.suggest_categorical(f"conv_layer{i}", self.hidden_variants)
                           for i in range(self.n_conv)]
            hidden_linear = [hidden_metal[-1] + hidden_conv[-1]] + \
                            [self.trial.suggest_categorical(f"linear_layer{i}", self.hidden_variants)
                             for i in range(self.n_linear)]
        elif self.metal_ligand_unifunc_name in ["max", "sum", "mean"]:
            hidden_conv = [self.conv_features] + \
                          [self.trial.suggest_categorical(f"conv_layer{i}", self.hidden_variants)
                           for i in range(self.n_conv - 1)] + \
                          [hidden_metal[-1]]
            hidden_linear = [hidden_metal[-1]] + \
                            [self.trial.suggest_categorical(f"linear_layer{i}", self.hidden_variants)
                             for i in range(self.n_linear)]
        else:
            raise ValueError(f"Unknown metal_ligand_unifunc_name: '{self.metal_ligand_unifunc_name}'")

        return hidden_metal, hidden_conv, hidden_linear

    def get_metal_ligand_unifunc(self):
        self.metal_ligand_unifunc_name = self.trial.suggest_categorical(
            "metal_ligand_unifunc_name",
            list(self.metal_ligand_unifunc_variants.keys())
        )
        return self.metal_ligand_unifunc_variants[self.metal_ligand_unifunc_name]

    def get(self):
        self.metal_ligand_unifunc = self.get_metal_ligand_unifunc()

        self.n_conv = self.trial.suggest_int("n_conv", *self.n_conv_lims)
        self.n_metal = self.trial.suggest_int("n_metal", *self.n_metal_lims)
        self.n_linear = self.trial.suggest_int("n_linear", *self.n_linear_lims)

        self.conv_dropout = self.trial.suggest_float("conv_dropout", *self.conv_dropout_lims)
        self.metal_dropout = self.trial.suggest_float("metal_dropout", *self.metal_dropout_lims)
        self.linear_dropout = self.trial.suggest_float("linear_dropout", *self.linear_dropout_lims)
        self.metal_bn = self.trial.suggest_categorical("metal_bn", self.metal_bn_variants)
        self.linear_bn = self.trial.suggest_categorical("linear_bn", self.linear_bn_variants)
        self.conv_layer = self.get_conv_layer()
        self.conv_actf = self.get_conv_actf()
        self.linear_actf = self.get_linear_actf()

        self.hidden_metal, self.hidden_conv, self.hidden_linear = self.get_hidden_dims()
        self.pooling_layer = self.get_pooling_layer()

        self.optimizer = self.get_optimizer()
        self.optimizer_parameters = self.get_optimizer_parameters()

        model_parameters = {
            "metal_ligand_unifunc": self.metal_ligand_unifunc,
            "hidden_conv": self.hidden_conv,
            "hidden_metal": self.hidden_metal,
            "hidden_linear": self.hidden_linear,
            "conv_layer": self.conv_layer,
            "pooling_layer": self.pooling_layer,
            "conv_dropout": self.conv_dropout,
            "metal_dropout": self.metal_dropout,
            "linear_dropout": self.linear_dropout,
            "linear_bn": self.linear_bn,
            "metal_bn": self.linear_bn,
            "conv_actf": self.conv_actf,
            "linear_actf": self.linear_actf,
            "optimizer": self.optimizer,
            "optimizer_parameters": self.optimizer_parameters,
        }

        return model_parameters


# def objective(trial: optuna.Trial):
#     conv_features = trial.suggest_categorical("conv_features", [75])
#
#     featurized_train = featurize_sdf(path_to_sdf, valuenames)
#     folds, test_data = train_test_valid_split(featurized_train, n_split,
#                                               batch_size=64, subsample_size=False,
#                                               test_ratio=0.1, return_test=True)
#
#     model_parameters = MolGraphNetParams(
#         trial,
#         conv_features=conv_features,
#         n_conv_lims=(2, 6), n_linear_lims=(2, 5),
#         conv_dropout_lims=(0, 0.5), linear_dropout_lims=(0, 0.5),
#         conv_layer_variants={
#             # "GCNConv": GCNConv,
#             # "TAGConv": TAGConv,
#             # "ARMAConv": ARMAConv,
#             # "SGConv": SGConv,
#             # "FeaStConv": FeaStConv,
#             # "ClusterGCNConv": ClusterGCNConv,
#             # "GENConv": GENConv,
#             # "SuperGATConv": SuperGATConv,
#             # "EGConv": EGConv,
#             # "SAGEConv": SAGEConv,
#             # "GraphConv": GraphConv,
#             # "ResGatedGraphConv": ResGatedGraphConv,
#             # "GATConv": GATConv,
#             # "GATv2Conv": GATv2Conv,
#             # "TransformerConv": TransformerConv,
#             "MFConv": MFConv,
#             # "PointTransformerConv": PointTransformerConv,
#             # "LEConv": LEConv,
#             # "FiLMConv": FiLMConv,
#             # "HypergraphConv": HypergraphConv
#         },
#         pooling_layer_variants={
#             "global_mean_pool": global_mean_pool,
#             "global_max_pool": global_max_pool,
#             # "TopKPooling": TopKPooling,
#             # "SAGPooling": SAGPooling,
#             # "EdgePooling": EdgePooling,
#             # "ASAPooling": ASAPooling,
#             # "PANPooling": PANPooling,
#         },
#         conv_actf_variants={
#             "nn.ELU()": nn.ELU(),
#             "nn.LeakyReLU()": nn.LeakyReLU(),
#             "nn.LogSigmoid()": nn.LogSigmoid(),
#             # "nn.MultiheadAttention()": nn.MultiheadAttention(),
#             "nn.PReLU()": nn.PReLU(),
#             "nn.ReLU()": nn.ReLU(),
#             "nn.ReLU6()": nn.ReLU6(),
#             "nn.RReLU()": nn.RReLU(),
#             "nn.SELU()": nn.SELU(),
#             "nn.CELU()": nn.CELU(),
#             "nn.GELU()": nn.GELU(),
#             "nn.SiLU()": nn.SiLU(),
#             "nn.Mish()": nn.Mish(),
#             "nn.Softplus()": nn.Softplus(),
#             "nn.Softshrink()": nn.Softshrink(),
#             "nn.Softsign()": nn.Softsign(),
#             "nn.Tanh()": nn.Tanh(),
#             "nn.Tanhshrink()": nn.Tanhshrink(),
#             # "nn.GLU()": nn.GLU(),
#         },
#         linear_actf_variants={
#             "nn.ELU()": nn.ELU(),
#             "nn.LeakyReLU()": nn.LeakyReLU(),
#             "nn.LogSigmoid()": nn.LogSigmoid(),
#             # "nn.MultiheadAttention()": nn.MultiheadAttention(),
#             "nn.PReLU()": nn.PReLU(),
#             "nn.ReLU()": nn.ReLU(),
#             "nn.ReLU6()": nn.ReLU6(),
#             "nn.RReLU()": nn.RReLU(),
#             "nn.SELU()": nn.SELU(),
#             "nn.CELU()": nn.CELU(),
#             "nn.GELU()": nn.GELU(),
#             "nn.SiLU()": nn.SiLU(),
#             "nn.Mish()": nn.Mish(),
#             "nn.Softplus()": nn.Softplus(),
#             "nn.Softshrink()": nn.Softshrink(),
#             "nn.Softsign()": nn.Softsign(),
#             "nn.Tanh()": nn.Tanh(),
#             "nn.Tanhshrink()": nn.Tanhshrink(),
#             # "nn.GLU()": nn.GLU(),
#         },
#         optimizer_variants={
#             "Adam": Adam,
#         },
#         hidden_variants=(64, 128, 256), linear_bn_variants=[False],
#     ).get()
#
#     trainer = ModelTrainer(
#         model=MolGraphNet(
#             dataset=folds[0][0],
#             mode="regression",
#             **model_parameters,
#         ),
#         train_valid_data=folds,
#         test_data=test_data,
#         output_folder=output_path,
#         out_folder_mark=output_mark,
#         epochs=1000,
#         es_patience=100,
#     )
#
#     metrics = trainer.train_cv_models()
#     trial.set_user_attr(key="metrics", value=metrics)
#
#     return metrics["valid_mean_squared_error"]


def objective(trial: optuna.Trial):
    metal_features = trial.suggest_categorical("metal_features", [200])
    conv_features = trial.suggest_categorical("conv_features", [75])

    train_data = DataLoader(featurize_sdf_with_metal(path_to_sdf=train_sdf,
                                                     mol_featurizer=ConvMolFeaturizer(),
                                                     metal_featurizer=SkipatomFeaturizer("Source/featurizers/skipatom_vectors_dim200.torch")), batch_size=batch_size)

    metal_val_sets = [featurize_sdf_with_metal(path_to_sdf=test_sdf,
                                               mol_featurizer=ConvMolFeaturizer(),
                                               metal_featurizer=SkipatomFeaturizer("Source/featurizers/skipatom_vectors_dim200.torch"))
                      for test_sdf in test_sdfs]
    general_val_set = []
    for val_set in metal_val_sets: general_val_set += val_set

    val_data = DataLoader(general_val_set, batch_size=batch_size)
    test_batches = [Batch.from_data_list(val_set) for val_set in metal_val_sets]

    model_parameters = MolGraphHeteroNetParams(
        trial,
        metal_features=metal_features, conv_features=conv_features,
        n_metal_lims=(2, 6), n_conv_lims=(1, 3), n_linear_lims=(2, 5),
        metal_dropout_lims=(0, 0), conv_dropout_lims=(0, 1), linear_dropout_lims=(0, 0),
        metal_bn_variants=[True, False], linear_bn_variants=[True, False],
        conv_layer_variants={
            # "GCNConv": GCNConv,
            # "TAGConv": TAGConv,
            # "ARMAConv": ARMAConv,
            # "SGConv": SGConv,
            # "FeaStConv": FeaStConv,
            # "ClusterGCNConv": ClusterGCNConv,
            # "GENConv": GENConv,
            # "SuperGATConv": SuperGATConv,
            # "EGConv": EGConv,
            # "SAGEConv": SAGEConv,
            # "GraphConv": GraphConv,
            # "ResGatedGraphConv": ResGatedGraphConv,
            # "GATConv": GATConv,
            # "GATv2Conv": GATv2Conv,
            # "TransformerConv": TransformerConv,
            "MFConv": MFConv,
            # "PointTransformerConv": PointTransformerConv,
            # "LEConv": LEConv,
            # "FiLMConv": FiLMConv,
            # "HypergraphConv": HypergraphConv
        },
        pooling_layer_variants={
            "global_mean_pool": global_mean_pool,
            "global_max_pool": global_max_pool,
            # "TopKPooling": TopKPooling,
            # "SAGPooling": SAGPooling,
            # "EdgePooling": EdgePooling,
            # "ASAPooling": ASAPooling,
            # "PANPooling": PANPooling,
        },
        conv_actf_variants={
            "nn.LeakyReLU()": nn.LeakyReLU(),
            "nn.PReLU()": nn.PReLU(),
            "nn.Tanhshrink()": nn.Tanhshrink(),
        },
        linear_actf_variants={
            "nn.LeakyReLU()": nn.LeakyReLU(),
            "nn.PReLU()": nn.PReLU(),
            "nn.Tanhshrink()": nn.Tanhshrink(),
        },
        metal_ligand_unifunc_variants={
            "mean": mean_unifunc,
            "max": max_unifunc,
            # "concat": concat_unifunc,
        },
        optimizer_variants={
            "Adam": Adam,
            "AdamW": AdamW,
            "RMSprop": RMSprop,
        },
        hidden_variants=(64, 128, 256),
    ).get()
    model_parameters["node_features"] = get_num_node_features(train_data)
    model_parameters["metal_features"] = get_num_metal_features(train_data)
    model_parameters["num_targets"] = get_num_targets(train_data)
    model_parameters["batch_size"] = get_batch_size(train_data)
    model_parameters["mode"] = "regression"

    trainer = MolGraphHeteroNetTrainer(
        model=MolGraphHeteroNet(**model_parameters),
        train_valid_data=((train_data, val_data),),
        output_folder=output_path,
        out_folder_mark=output_mark,
        epochs=epochs,
        es_patience=es_patience,
        verbose=True,
    )

    model, metrics = trainer.train_cv_models()
    trial.set_user_attr(key="metrics", value=metrics)
    trial.set_user_attr(key="model_parameters", value=model_parameters)

    max_error = max([mean_squared_error(batch.y, model(x=batch.x,
                                                       edge_index=batch.edge_index,
                                                       metal_x=batch.metal_x,
                                                       batch=batch.batch)) for batch in test_batches])

    return max_error


def callback(study: optuna.Study, trial):
    time_mark = str(start).replace(" ", "_").replace("-", "_").replace(":", "_").split(".")[0]
    study.trials_dataframe().to_csv(path_or_buf=os.path.join(output_path, f"{output_mark}_{time_mark}.csv"),
                                    index=False)


if __name__ == "__main__":
    start = datetime.now()
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=None, callbacks=[callback])
    end = datetime.now()

    best_params = DictHeteroParams(study.best_trial.params).get()
    result = {
        "trials": len(study.trials),
        "started": str(start).split(".")[0],
        "finished": str(end).split(".")[0],
        "duration": str(end - start).split(".")[0],
        **study.best_trial.user_attrs["metrics"],
        "model_parameters": best_params,
    }

    time_mark = str(start).replace(" ", "_").replace("-", "_").replace(":", "_").split(".")[0]

    study.trials_dataframe().to_csv(path_or_buf=os.path.join(output_path, f"{output_mark}_{time_mark}.csv"),
                                    index=False)
    with open(os.path.join(output_path, f"{output_mark}_{time_mark}.json"), "w") as jf:
        json.dump(result, jf)
