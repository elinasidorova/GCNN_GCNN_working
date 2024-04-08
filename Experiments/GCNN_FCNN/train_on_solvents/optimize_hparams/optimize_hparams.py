import logging
from argparse import ArgumentParser
from functools import partial

import mlflow
import optuna
import torch
from dgllife.utils import CanonicalAtomFeaturizer
from sklearn.metrics import r2_score, mean_absolute_error
from torch import nn
from torch_geometric.loader import DataLoader

from Experiments.GCNN_FCNN.train_on_solvents.optimize_hparams.search_space import OPTIMIZER_VARIANTS, LR_LIMS, \
    FEATURIZER_VARIANTS, METAL_FEATURIZER_VARIANTS, BATCH_SIZE_LIMS, METAL_FC_PARAMS, GCNN_PARAMS, POST_FC_PARAMS, \
    GLOBAL_POOLING_VARIANTS
from Source.data import balanced_train_valid_split, root_mean_squared_error
from Source.models.GCNN.featurizers import DGLFeaturizer
from Source.models.GCNN.trainer import GCNNTrainer
from Source.models.GCNN_FCNN.featurizers import featurize_sdf_with_solvent
from Source.models.GCNN_FCNN.model import GCNN_FCNN
from Source.models.GCNN_FCNN.optimize_hparams import GCNNFCNNParams
from Source.models.GCNN_FCNN.optimize_hparams import GeneralParams
from config import ROOT_DIR

parser = ArgumentParser()
parser.add_argument('--train-sdf', type=str, help='Path to the data file')  # required=True,
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
               "name": "logS",
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
        optimizer_variants=OPTIMIZER_VARIANTS,
        lr_lims=LR_LIMS,
        featurizer_variants=FEATURIZER_VARIANTS,
        metal_featurizer_variants=METAL_FEATURIZER_VARIANTS,
        batch_size_lims=BATCH_SIZE_LIMS,
    ).get()

    model_parameters = GCNNFCNNParams(
        trial,
        metal_fc_params=METAL_FC_PARAMS,
        gcnn_params=GCNN_PARAMS,
        post_fc_params=POST_FC_PARAMS,
        global_pooling_variants=GLOBAL_POOLING_VARIANTS,
    ).get()

    featurize = partial(
        featurize_sdf_with_solvent,
        mol_featurizer=general_params["featurizer"],
    )

    train_val_dataset = featurize(path_to_sdf=str(ROOT_DIR / args.train_sdf))
    logging.info("Splitting...")
    folds = balanced_train_valid_split(
        datasets=[train_val_dataset],
        n_folds=args.folds,
        batch_size=general_params["batch_size"],
        shuffle_every_epoch=True,
        seed=args.seed
    )
    if args.test_sdf is not None:
        test_dataset = featurize(path_to_sdf=str(ROOT_DIR / args.test_sdf))
        test_loader = DataLoader(test_dataset)
    else:
        test_loader = None

    model = GCNN_FCNN(
        metal_features=train_val_dataset[0].x_fully_connected.shape[-1],
        node_features=train_val_dataset[0].x.shape[-1],
        targets=targets,
        **model_parameters,
        optimizer=torch.optim.Adam,
        optimizer_parameters=None,
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
    with mlflow.start_run(run_name=f"trial-{trial.number}"):
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
