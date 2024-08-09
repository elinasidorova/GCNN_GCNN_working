import logging #
from argparse import ArgumentParser
from functools import partial
import pandas as pd

import mlflow
import optuna
import torch
import torch.nn as nn
from dgllife.utils import CanonicalAtomFeaturizer, AttentiveFPAtomFeaturizer, PAGTNAtomFeaturizer
from sklearn.metrics import r2_score, mean_absolute_error
from torch_geometric.nn import MFConv, GATConv, TransformerConv, SAGEConv, global_mean_pool, global_max_pool
from torch_geometric.loader import DataLoader #

from Source.data import root_mean_squared_error
from Source.models.GCNN.model import GCNN
from Source.models.GCNN.optimize_hparams import GCNNParams, GeneralParams
from Source.models.GCNN.trainer import GCNNTrainer
from config import ROOT_DIR
from Source.models.GCNN.featurizers import ConvMolFeaturizer
from Source.models.GCNN.featurizers import featurize_df
from Source.data import train_test_valid_split
from Source.models.GCNN_FCNN.featurizers import featurize_sdf_with_solvent #
from Source.data import balanced_train_valid_split #
from Source.models.GCNN.featurizers import featurize_sdf #

# ACTIVATION_VARIANTS = {
#     "LeakyReLU": nn.LeakyReLU(),
#     "PReLU": nn.PReLU(),
#     "Tanhshrink": nn.Tanhshrink(),
# }
#
# CONVOLUTION_VARIANTS = {
#     "MFConv": MFConv,
#     "GATConv": GATConv,
#     "TransformerConv": TransformerConv,
#     "SAGEConv": SAGEConv,
# }
#
# POOLING_VARIANTS = {
#     "global_mean_pool": global_mean_pool,
#     "global_max_pool": global_max_pool,
# }
# DIM_LIMS = (32, 1024)
#
# OPTIMIZER_VARIANTS = {
#     "Adam": torch.optim.Adam,
#     "AdamW": torch.optim.AdamW,
#     "RMSprop": torch.optim.RMSprop,
#     "SGD": torch.optim.SGD,
# }
#
# LR_LIMS = (1e-2, 1e-2)
#
# FEATURIZER_VARIANTS = {
#     "CanonicalAtomFeaturizer": CanonicalAtomFeaturizer(),
#     "AttentiveFPAtomFeaturizer": AttentiveFPAtomFeaturizer(),
#     "PAGTNAtomFeaturizer": PAGTNAtomFeaturizer(),
#     "ConvMolFeaturizer": ConvMolFeaturizer(),
# }
#
# BATCH_SIZE_LIMS = (8, 64)

parser = ArgumentParser()
#parser.add_argument('--train-data', type=str, help='Path to the data file')  # required=True,
parser.add_argument('--train-sdf', type=str, help='Path to the data file')
parser.add_argument('--test-sdf', type=str, help='Path to the data file')
parser.add_argument('--experiment-name', type=str, help='The name of the experiment')  # required=True,
parser.add_argument('--max-samples', type=int, default=None, help='Use only a several samples for training')
parser.add_argument('--folds', type=int, default=1, help='Number of folds for cross-validation')
parser.add_argument('--seed', type=int, default=0, help='Seed for random number generators')
parser.add_argument('--epochs', type=int, default=1000, help='Maximum number of epochs to train')
parser.add_argument('--es-patience', type=int, default=100, help='Number of epochs to wait before early stopping')
parser.add_argument('--mode', type=str, default="regression",
                    help='Mode of the target - regression, binary_classification or multiclass_classification')
parser.add_argument('--n-trials', type=int, default=None, help='Number of optuna trials')
parser.add_argument('--timeout', type=int, default=None, help='Time limit for optuna study (in seconds)')
args = parser.parse_args()

targets = ({
               "name": "Solubility",
               "mode": "regression",
               "dim": 1,
               "metrics": {
                   "R2": (r2_score, {}),
                   "RMSE": (root_mean_squared_error, {}),
                   "MAE": (mean_absolute_error, {})
               },
               "loss": nn.MSELoss(),
           },)

mlflow.set_experiment(args.experiment_name)


def objective(trial: optuna.Trial):
    general_params = GeneralParams(
        trial,
        optimizer_variants={
            "Adam": torch.optim.Adam,
            "AdamW": torch.optim.AdamW,
            "RMSprop": torch.optim.RMSprop,
            "SGD": torch.optim.SGD,
        },
        lr_lims=(1e-2, 1e-2),
        featurizer_variants={
            "CanonicalAtomFeaturizer": CanonicalAtomFeaturizer(),
            "AttentiveFPAtomFeaturizer": AttentiveFPAtomFeaturizer(),
            "PAGTNAtomFeaturizer": PAGTNAtomFeaturizer(),
            "ConvMolFeaturizer": ConvMolFeaturizer(),
        },
        batch_size_lims=(8, 64),
    ).get()

    # model_parameters = GCNNParams(
    #     trial,
    #     pre_fc_params={
    #         "dim_lims": (32, 1024),
    #         "n_layers_lims": (1, 4),
    #         "actf_variants": {
    #             "LeakyReLU": nn.LeakyReLU(),
    #             "PReLU": nn.PReLU(),
    #             "Tanhshrink": nn.Tanhshrink(),
    #         },
    #         "dropout_lims": (0, 0),
    #         "bn_variants": (True, False),
    #     },
    #     post_fc_params={
    #         "dim_lims": (32, 1024),
    #         "n_layers_lims": (1, 5),
    #         "actf_variants": {
    #             "LeakyReLU": nn.LeakyReLU(),
    #             "PReLU": nn.PReLU(),
    #             "Tanhshrink": nn.Tanhshrink(),
    #         },
    #         "dropout_lims": (0, 0),
    #         "bn_variants": (True, False),
    #     },
    #     n_conv_lims=(1, 3),
    #     dropout_lims=(0.1, 0.5),
    #     actf_variants={
    #             "LeakyReLU": nn.LeakyReLU(),
    #             "PReLU": nn.PReLU(),
    #             "Tanhshrink": nn.Tanhshrink(),
    #         },
    #     dim_lims=(32, 1024),
    #     conv_layer_variants={
    #         "MFConv": MFConv,
    #         "GATConv": GATConv,
    #         "TransformerConv": TransformerConv,
    #         "SAGEConv": SAGEConv,
    #     },
    #     pooling_layer_variants={
    #         "global_mean_pool": global_mean_pool,
    #         "global_max_pool": global_max_pool,
    #     },
    # ).get()

    model_parameters = GCNNParams(
        trial,
        pre_fc_params={
            "dim_lims": (32, 1024),
            "n_layers_lims": (1, 4),
            "actf_variants": {
                "LeakyReLU": nn.LeakyReLU(),
                "PReLU": nn.PReLU(),
                "Tanhshrink": nn.Tanhshrink(),
                },
            "dropout_lims": (0, 0),
            "bn_variants": (True, False),
        },
        post_fc_params={
            "dim_lims": (32, 1024),
            "n_layers_lims": (1, 5),
            "actf_variants": {
                "LeakyReLU": nn.LeakyReLU(),
                "PReLU": nn.PReLU(),
                "Tanhshrink": nn.Tanhshrink(),
                },
            "dropout_lims": (0, 0),
            "bn_variants": (True, False),
        },
        n_conv_lims=(1, 3),
        dropout_lims=(0.1, 0.5),
        actf_variants={
            "LeakyReLU": nn.LeakyReLU(),
            "PReLU": nn.PReLU(),
            "Tanhshrink": nn.Tanhshrink(),
        },
        dim_lims=(32, 1024),
        conv_layer_variants={
            "MFConv": MFConv,
            "GATConv": GATConv,
            "TransformerConv": TransformerConv,
            "SAGEConv": SAGEConv,
        },
        pooling_layer_variants={
            "global_mean_pool": global_mean_pool,
            "global_max_pool": global_max_pool,
        },
    ).get()


    test_graphs = featurize_sdf(
        path_to_sdf=str(ROOT_DIR / args.train_sdf),
        targets=["Solubility"],
        mol_featurizer=general_params["featurizer"])
    train_graphs = featurize_sdf(
        path_to_sdf=str(ROOT_DIR / args.test_sdf),
        targets=["Solubility"],
        mol_featurizer=general_params["featurizer"])
    logging.info("Splitting...")
    folds = balanced_train_valid_split([train_graphs], n_folds=args.folds,
                                       batch_size=general_params["batch_size"],
                                       shuffle_every_epoch=True,
                                       seed=args.seed)

    test_loader = DataLoader(test_graphs, batch_size=general_params["batch_size"])

    #коммент - вариант, аналогичный изначальному в этом файле (работа с csv а не с sdf, ошибка та же)

    # total_df = pd.read_csv(str(ROOT_DIR / args.train_data), nrows=args.max_samples)
    #
    # featurize = partial(featurize_df, mol_featurizer=general_params["featurizer"])
    # train_dataset = featurize(df=total_df)
    # folds, test_loader = train_test_valid_split(train_dataset,
    #                                             n_folds=args.folds,
    #                                             test_ratio=0.1,
    #                                             batch_size=general_params["batch_size"],
    #                                             seed=args.seed)

    model = GCNN(
        node_features=next(iter(test_loader)).x.shape[-1],
        targets=targets,
        **model_parameters,
        optimizer=torch.optim.Adam,
        optimizer_parameters=None,
    )
    # model = GCNN(
    #     node_features=train_dataset[0].x.shape[-1],
    #     targets=targets,
    #     **model_parameters,
    #     optimizer=general_params["optimizer"],
    #     optimizer_parameters=general_params["optimizer_parameters"],
    # )

    trainer = GCNNTrainer(
        model=model,
        train_valid_data=folds,
        test_data=test_loader,
        output_folder=str(ROOT_DIR / "Output" / args.experiment_name),
        epochs=args.epochs,
        es_patience=args.es_patience,
        targets=targets,
        seed=args.seed,
    )



    with mlflow.start_run(run_name=f"{trial.number}"):
        mlflow.log_params(trial.params)
        mlflow.set_tags({"trial": trial.number, "datetime_start": trial.datetime_start})

        trainer.train_cv_models()

        mlflow.log_metrics(trainer.results_dict["general"])
    trial.set_user_attr(key="metrics", value=trainer.results_dict["general"])
    return trainer.results_dict["general"]["logS_valid_RMSE"]



study = optuna.create_study(
    study_name=args.experiment_name,
    storage=f"sqlite:///{ROOT_DIR / 'Output' / args.experiment_name}.db",
    load_if_exists=True,
    direction="minimize"
)

study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout, catch=(ValueError,))
