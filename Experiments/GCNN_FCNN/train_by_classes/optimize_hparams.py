from argparse import ArgumentParser
from functools import partial

import mlflow
import optuna
import pandas as pd
import torch
from dgllife.utils import AttentiveFPAtomFeaturizer, PAGTNAtomFeaturizer
from dgllife.utils import CanonicalAtomFeaturizer
from pytorch_lightning.loggers import MLFlowLogger
from sklearn.metrics import r2_score, mean_absolute_error
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, MFConv, global_max_pool, SAGEConv, GATConv, TransformerConv

from Source.data import balanced_train_valid_split, root_mean_squared_error
from Source.models.GCNN.trainer import GCNNTrainer
from Source.models.GCNN_FCNN.featurizers import SkipatomFeaturizer, featurize_df
from Source.models.GCNN_FCNN.model import GCNN_FCNN
from Source.models.GCNN_FCNN.optimize_hparams import GCNNFCNNParams
from Source.models.GCNN_FCNN.optimize_hparams import GeneralParams
from Source.models.global_poolings import MaxPooling, ConcatPooling, SumPooling, CrossAttentionPooling
from config import ROOT_DIR

parser = ArgumentParser()
parser.add_argument('--train-data', type=str, help='Path to the data file')  # required=True,
parser.add_argument('--test-data', type=str, help='Path to the data file')
parser.add_argument('--conditions', type=str, help='List of condition columns to consider')
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

ACTIVATION_VARIANTS = {
    "LeakyReLU": nn.LeakyReLU(),
    "PReLU": nn.PReLU(),
    "Tanhshrink": nn.Tanhshrink(),
}
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
        lr_lims=(1e-4, 1e-1),
        featurizer_variants={
            "CanonicalAtomFeaturizer": CanonicalAtomFeaturizer(),
            "AttentiveFPAtomFeaturizer": AttentiveFPAtomFeaturizer(),
            "PAGTNAtomFeaturizer": PAGTNAtomFeaturizer(),
        },
        metal_featurizer_variants={"SkipatomFeaturizer": SkipatomFeaturizer()},
        batch_size_lims=(8, 64),
    ).get()

    model_parameters = GCNNFCNNParams(
        trial,
        metal_fc_params={
            "dim_lims": (32, 512),
            "n_layers_lims": (1, 4),
            "actf_variants": ACTIVATION_VARIANTS,
            "dropout_lims": (0, 0),
            "bn_variants": (True, False),
        },
        gcnn_params={
            "pre_fc_params": {
                "dim_lims": (32, 512),
                "n_layers_lims": (1, 4),
                "actf_variants": ACTIVATION_VARIANTS,
                "dropout_lims": (0, 0),
                "bn_variants": (True, False),
            },
            "post_fc_params": {
                "dim_lims": (32, 512),
                "n_layers_lims": (1, 4),
                "actf_variants": ACTIVATION_VARIANTS,
                "dropout_lims": (0, 0),
                "bn_variants": (True, False),
            },
            "n_conv_lims": (1, 3),
            "dropout_lims": (0, 0.8),
            "actf_variants": ACTIVATION_VARIANTS,
            "dim_lims": (32, 512),
            "conv_layer_variants": {
                "MFConv": MFConv,
                "GATConv": GATConv,
                "TransformerConv": TransformerConv,
                "SAGEConv": SAGEConv,
            },
            "pooling_layer_variants": {
                "global_mean_pool": global_mean_pool,
                "global_max_pool": global_max_pool,
            },
        },
        post_fc_params={
            "dim_lims": (32, 512),
            "n_layers_lims": (1, 4),
            "actf_variants": ACTIVATION_VARIANTS,
            "dropout_lims": (0, 0),
            "bn_variants": (True, False),
        },
        global_pooling_variants={
            "ConcatPooling": ConcatPooling,
            "SumPooling": SumPooling,
            "MaxPooling": MaxPooling,
            "CrossAttentionPooling": CrossAttentionPooling,
        },
    ).get()

    featurize = partial(
        featurize_df,
        mol_featurizer=general_params["featurizer"],
        metal_featurizer=general_params["metal_featurizer"],
        conditions=args.conditions.split(" "),
    )

    train_datasets = [
        featurize(df=metal_df.reset_index())
        for _, metal_df in total_df.groupby("metal")
    ]
    folds = balanced_train_valid_split(train_datasets, n_folds=args.folds,
                                       batch_size=general_params["batch_size"],
                                       shuffle_every_epoch=True,
                                       seed=args.seed)
    if args.test_data is not None:
        test_dataset = featurize(df=pd.read_csv(str(ROOT_DIR / args.test_data)), )
        test_loader = DataLoader(test_dataset)
    else:
        test_loader = None

    model = GCNN_FCNN(
        metal_features=train_datasets[0][0].metal_x.shape[-1],
        node_features=train_datasets[0][0].x.shape[-1],
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
        mlflow.log_input(mlflow.data.from_pandas(total_df, source=args.train_data), context="train_data")

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
