import json
import os
from argparse import ArgumentParser
from datetime import datetime

import optuna
import torch
import torch.nn as nn
from dgllife.utils import CanonicalAtomFeaturizer, AttentiveFPAtomFeaturizer, PAGTNAtomFeaturizer
from sklearn.metrics import r2_score, mean_absolute_error
from torch.nn import MSELoss
from torch_geometric.nn import MFConv, GATConv, TransformerConv, SAGEConv, global_mean_pool, global_max_pool

from Source.data import root_mean_squared_error, train_test_valid_split
from Source.models.GCNN.featurizers import featurize_csv
from Source.models.GCNN.model import GCNN
from Source.models.GCNN.optimize_hparams import GCNNParams, GeneralParams
from Source.models.GCNN.trainer import GCNNTrainer

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

parser = ArgumentParser()
parser.add_argument('--n-trials', type=int, default=100, help='Number of optuna trials')
parser.add_argument('--timeout', type=int, default=None, help='Time limit (in seconds) for optuna optimization')
parser.add_argument('--data', type=str, required=True, help='Path to the data file')
parser.add_argument('--target-name', type=str, required=True, help='Name of column with targets')
parser.add_argument('--output-folder', type=str, required=True, help='Output folder')
parser.add_argument('--max-samples', type=int, default=None, help='Use only a several samples for training')
parser.add_argument('--test-ratio', type=float, default=0.2, help='Ratio of data to be used for testing')
parser.add_argument('--folds', type=int, default=1, help='Number of folds for cross-validation')
parser.add_argument('--seed', type=int, default=0, help='Seed for random number generators')
parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
parser.add_argument('--epochs', type=int, default=1000, help='Maximum number of epochs to train')
parser.add_argument('--es-patience', type=int, default=100, help='Number of epochs to wait before early stopping')
parser.add_argument('--mode', type=str, default="regression",
                    help='Mode of the target - regression, binary_classification or multiclass_classification')
parser.add_argument('--learning-rate', default=1e-3, type=float, help='Learning rate')
parser.add_argument('--save-models', action='store_true',
                    help='If set, models from each trial will be saved to output folder')

args = parser.parse_args()

os.makedirs(args.output_folder, exist_ok=True)
targets = ({
               'name': args.target_name,
               'mode': args.mode,
               'dim': 1,
               'metrics': {
                   'R2': (r2_score, {}),
                   'RMSE': (root_mean_squared_error, {}),
                   'MAE': (mean_absolute_error, {})
               },
               'loss': MSELoss()
           },)


def objective(trial: optuna.Trial):
    model_parameters = GCNNParams(
        trial,
        pre_fc_params={
            "dim_lims": (32, 512),
            "n_layers_lims": (1, 4),
            "actf_variants": ACTIVATION_VARIANTS,
            "dropout_lims": (0, 0),
            "bn_variants": (True,)
        },
        post_fc_params={
            "dim_lims": (32, 512),
            "n_layers_lims": (1, 4),
            "actf_variants": ACTIVATION_VARIANTS,
            "dropout_lims": (0, 0),
            "bn_variants": (True,)
        },
        n_conv_lims=(1, 3), dropout_lims=(0, 1),
        actf_variants=ACTIVATION_VARIANTS, dim_lims=(32, 512),
        conv_layer_variants=CONVOLUTION_VARIANTS,
        pooling_layer_variants=POOLING_VARIANTS,
    ).get()

    general_parameters = GeneralParams(
        trial,
        optimizer_variants={
            "SGD": torch.optim.SGD,
            'Adam': torch.optim.Adam,
            'AdamW': torch.optim.AdamW,
            "RMSprop": torch.optim.RMSprop,
            "Adagrad": torch.optim.Adagrad,
            "Adadelta": torch.optim.Adadelta,
            "Adamax": torch.optim.Adamax,
        },
        lr_lims=(5e-5, 5e-4),
        featurizer_variants={
            "CanonicalAtomFeaturizer": CanonicalAtomFeaturizer(),
            "AttentiveFPAtomFeaturizer": AttentiveFPAtomFeaturizer(),
            "PAGTNAtomFeaturizer": PAGTNAtomFeaturizer(),
        },
    ).get()

    full_dataset = featurize_csv(path_to_csv=args.data, mol_featurizer=general_parameters["featurizer"],
                                 targets=("mu",), seed=args.seed, max_samples=args.max_samples)
    folds, test_data = train_test_valid_split(full_dataset, n_folds=args.folds, test_ratio=args.test_ratio,
                                              batch_size=64, seed=args.seed)

    model_parameters["node_features"] = test_data.dataset[0].x.shape[-1]
    model_parameters["targets"] = targets
    model_parameters["optimizer"] = general_parameters["optimizer"]
    model_parameters["optimizer_parameters"] = general_parameters["optimizer_parameters"]

    model = GCNN(**model_parameters)

    time_mark = str(datetime.now()).replace(" ", "_").replace("-", "_").replace(":", "_").split(".")[0]
    output_folder = os.path.join(args.output_folder, f"{trial.number}-trial_GCNN_{args.mode}_{time_mark}")

    trainer = GCNNTrainer(
        model=model,
        train_valid_data=folds,
        test_data=test_data,
        output_folder=output_folder,
        epochs=args.epochs,
        es_patience=args.es_patience,
        targets=targets,
        seed=args.seed,
        save_to_folder=args.save_models,
    )

    trainer.train_cv_models()

    trial.set_user_attr(key="metrics", value=trainer.results_dict["general"])
    trial.set_user_attr(key="model_parameters", value={key: str(model_parameters[key]) for key in model_parameters})

    errors = [trainer.results_dict["general"][key] for key in trainer.results_dict["general"] if "MSE" in key]
    return max(errors)


def callback(study: optuna.Study, trial):
    study.trials_dataframe().to_csv(path_or_buf=os.path.join(args.output_folder, f"trials.csv"), index=False)


if __name__ == "__main__":
    start = datetime.now()
    study = optuna.create_study(
        direction="minimize",
        storage=f"sqlite:///{args.output_folder}/optuna_study.db"
    )
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout, callbacks=[callback])
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

    study.trials_dataframe().to_csv(path_or_buf=os.path.join(args.output_folder, f"trials.csv"), index=False)

    with open(os.path.join(args.output_folder, f"result.json"), "w") as jf:
        json.dump(result, jf)
