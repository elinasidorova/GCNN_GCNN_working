import os
from argparse import ArgumentParser
from functools import partial

import mlflow
import optuna
import pandas as pd
import torch
import torch.nn as nn
from dgllife.utils import CanonicalAtomFeaturizer, AttentiveFPAtomFeaturizer, PAGTNAtomFeaturizer
from sklearn.metrics import r2_score, mean_absolute_error
from torch.nn import MSELoss
from torch_geometric.nn import MFConv, GATConv, TransformerConv, SAGEConv, global_mean_pool, global_max_pool

from Source.data import root_mean_squared_error, train_test_valid_split
from Source.models.GCNN.featurizers import featurize_df
from Source.models.GCNN.model import GCNN
from Source.models.GCNN.optimize_hparams import GCNNParams, GeneralParams
from Source.models.GCNN.trainer import GCNNTrainer
from config import ROOT_DIR

ACTIVATION_VARIANTS = {
    "LeakyReLU": nn.LeakyReLU(),
    "PReLU": nn.PReLU(),
    "Tanhshrink": nn.Tanhshrink(),
}

CONVOLUTION_VARIANTS = {
    "MFConv": MFConv,
    "GATConv": GATConv,
    "TransformerConv": TransformerConv,
    "SAGEConv": SAGEConv,
}

POOLING_VARIANTS = {
    "global_mean_pool": global_mean_pool,
    "global_max_pool": global_max_pool,
}
DIM_LIMS = (32, 1024)

parser = ArgumentParser()
parser.add_argument('--train-data', type=str, help='Path to the data file')  # required=True,
# parser.add_argument('--test-data', type=str, help='Path to the data file')
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
total_df = pd.read_csv(str(ROOT_DIR / args.train_data), nrows=args.max_samples)
mlflow.set_experiment(args.experiment_name)
mlflow.set_tracking_uri("http://127.0.0.1:8890")


def objective(trial: optuna.Trial):
    general_params = GeneralParams(
        trial,
        optimizer_variants={
            "Adam": torch.optim.Adam,
            "AdamW": torch.optim.AdamW,
            "RMSprop": torch.optim.RMSprop,
            "SGD": torch.optim.SGD
        },
        lr_lims=(1e-2, 1e-2),
        featurizer_variants={
            "CanonicalAtomFeaturizer": CanonicalAtomFeaturizer(),
            "AttentiveFPAtomFeaturizer": AttentiveFPAtomFeaturizer(),
            "PAGTNAtomFeaturizer": PAGTNAtomFeaturizer(),
        },
        batch_size_lims=(8, 64),
    ).get()

    model_parameters = GCNNParams(
        trial,
        pre_fc_params={
            "dim_lims": DIM_LIMS,
            "n_layers_lims": (1, 4),
            "actf_variants": ACTIVATION_VARIANTS,
            "dropout_lims": (0, 0),
            "bn_variants": (True, False),
        },
        post_fc_params={
            "dim_lims": DIM_LIMS,
            "n_layers_lims": (1, 5),
            "actf_variants": ACTIVATION_VARIANTS,
            "dropout_lims": (0, 0),
            "bn_variants": (True, False),
        },
        n_conv_lims=(1, 3),
        dropout_lims=(0.1, 0.5),
        actf_variants=ACTIVATION_VARIANTS,
        dim_lims=DIM_LIMS,
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

    featurize = partial(featurize_df, mol_featurizer=general_params["featurizer"])
    train_dataset = featurize(df=total_df)
    folds, test_loader = train_test_valid_split(train_dataset,
                                                n_folds=args.folds,
                                                test_ratio=0.1,
                                                batch_size=general_params["batch_size"],
                                                seed=args.seed)
    # if args.test_data is not None:
    #     test_dataset = featurize(df=pd.read_csv(str(ROOT_DIR / args.test_data)), )
    #     test_loader = DataLoader(test_dataset)
    # else:
    #     test_loader = None

    model = GCNN(
        node_features=train_dataset[0].x.shape[-1],
        targets=targets,
        **model_parameters,
        optimizer=general_params["optimizer"],
        optimizer_parameters=general_params["optimizer_parameters"],
    )

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
        mlflow.log_input(mlflow.data.from_pandas(total_df, source=args.train_data), context="total_data")

        trainer.train_cv_models()

        mlflow.log_metrics(trainer.results_dict["general"])
    trial.set_user_attr(key="metrics", value=trainer.results_dict["general"])
    return trainer.results_dict["general"]["logK_valid_RMSE"]


study = optuna.create_study(
    study_name=args.experiment_name,
    storage=f"sqlite:///{ROOT_DIR / 'Output' / args.experiment_name}.db",
    load_if_exists=True,
    direction="minimize"
)

study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout, catch=(ValueError,))
