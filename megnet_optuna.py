import copy
import json
import os
import random
from datetime import datetime

import optuna
import torch
import torch.nn as nn
import torch.utils.data
from rdkit import Chem
from sklearn.metrics import mean_squared_error
from torch.optim import Adam, AdamW, RMSprop
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from Source.models.megnet_model import MEGNet
from Source.featurizers.featurizers import ConvMolFeaturizer, SkipatomFeaturizer
from Source.trainer import MegnetTrainer

train_sdf = "Data/OptunaMgCdLa/train_trans.sdf"
val_sdfs = ["Data/OptunaMgCdLa/La_val.sdf"]
test_sdfs = ["Data/OptunaMgCdLa/La_test.sdf"]
output_path = "Output"
output_mark = f"MegnetOptunaLa"

n_trials = 150
batch_size = 128
epochs = 1000
es_patience = 100
seed = 31


def featurize_sdf_for_megnet(path_to_sdf,
                             mol_featurizer=ConvMolFeaturizer(),
                             metal_featurizer=SkipatomFeaturizer(),
                             seed=42):
    """
    Extract molecules from .sdf file and featurize them

    Parameters
    ----------
    path_to_sdf : str
        path to .sdf file with data
        single molecule in .sdf file can contain properties like "logK_{metal}"
        each of these properties will be transformed into a different training sample
    mol_featurizer : featurizer, optional
        instance of the class used for extracting features of organic molecule
    metal_featurizer : featurizer, optional
        instance of the class used for extracting metal features

    Returns
    -------
    features : list of torch_geometric.data objects
        list of graphs corresponding to individual molecules from .sdf file
    """
    suppl = Chem.SDMolSupplier(path_to_sdf)
    mols = [x for x in suppl if x is not None]
    mol_graphs = [mol_featurizer._featurize(m) for m in mols]

    all_data = []
    for mol_ind in range(len(mols)):
        if len(mol_graphs[mol_ind].edge_index.unique()) < mol_graphs[mol_ind].x.shape[0]: continue
        targets = [prop for prop in mols[mol_ind].GetPropNames() if prop.startswith("logK_")]
        for target in targets:
            graph = copy.deepcopy(mol_graphs[mol_ind])
            element_symbol = target.split("_")[-1]
            graph.u = metal_featurizer._featurize(element_symbol)
            graph.y = torch.tensor([float(mols[mol_ind].GetProp(target))])
            graph.edge_attr = torch.tensor([[1. for _ in range(19)] for _ in range(graph.edge_index.shape[1])])
            all_data += [graph]
    random.Random(seed).shuffle(all_data)

    return all_data


class MegnetParams:
    def __init__(self, trial: optuna.Trial, n_megnet_lims, batch_norm_variants,
                 use_pre_dense_general_variants, pre_dense_hidden_lims, megnet_hidden_lims, post_dense_hidden_lims,
                 n_post_dense_lims, actf_variants, optimizer_variants):
        self.trial = trial
        self.n_megnet_lims = n_megnet_lims
        self.batch_norm_variants = batch_norm_variants
        self.use_pre_dense_general_variants = use_pre_dense_general_variants
        self.pre_dense_hidden_lims = pre_dense_hidden_lims
        self.megnet_hidden_lims = megnet_hidden_lims
        self.post_dense_hidden_lims = post_dense_hidden_lims
        self.n_post_dense_lims = n_post_dense_lims
        self.actf_variants = actf_variants
        self.optimizer_variants = optimizer_variants

        self.linear_bn = None
        self.optimizer_parameters = None

        self.optimizer = None
        self.actf = None
        self.n_megnet = None
        self.use_pre_dense_general = None
        self.megnet_hidden = None
        self.pre_dense_hidden = None
        self.batch_norm = None

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
        self.n_megnet = self.trial.suggest_int("n_megnet", *self.n_megnet_lims)
        self.batch_norm = self.trial.suggest_categorical("batch_norm", self.batch_norm_variants)
        self.use_pre_dense_general = self.trial.suggest_categorical("use_pre_dense_general",
                                                                    self.use_pre_dense_general_variants)
        self.pre_dense_hidden = self.trial.suggest_int("pre_dense_hidden", *self.pre_dense_hidden_lims)
        self.megnet_hidden = self.trial.suggest_int("megnet_hidden", *self.megnet_hidden_lims)

        post_dense_hidden_0 = self.trial.suggest_int("post_dense_hidden_0", *self.post_dense_hidden_lims)
        n_post_dense = self.trial.suggest_int("n_post_dense", *self.n_post_dense_lims)

        self.actf_name = self.trial.suggest_categorical("actf", self.actf_variants.keys())
        self.actf = self.actf_variants[self.actf_name]

        self.optimizer = self.get_optimizer()
        self.optimizer_parameters = self.get_optimizer_parameters()

        model_parameters = {
            "pre_dense_edge_hidden": (),
            "pre_dense_node_hidden": (self.pre_dense_hidden,),
            "pre_dense_general_hidden": (self.pre_dense_hidden,) if self.use_pre_dense_general else (),
            "megnet_dense_hidden": (self.megnet_hidden,) * self.n_megnet,
            "megnet_edge_conv_hidden": (self.megnet_hidden,),
            "megnet_node_conv_hidden": (self.megnet_hidden,),
            "megnet_general_conv_hidden": (self.megnet_hidden,),
            "post_dense_hidden": tuple(post_dense_hidden_0 // (2 ** i) for i in range(n_post_dense)),
            "pool": "global_mean_pool",
            "pool_order": "early",
            "batch_norm": self.batch_norm,
            "batch_track_stats": True,
            "actf": self.actf,
            "dropout_rate": 0.0,
            "optimizer": self.optimizer,
            "optimizer_parameters": self.optimizer_parameters,
        }

        return model_parameters


def objective(trial: optuna.Trial):
    train_data = DataLoader(featurize_sdf_for_megnet(path_to_sdf=train_sdf,
                                                     mol_featurizer=ConvMolFeaturizer(),
                                                     metal_featurizer=SkipatomFeaturizer("Source/featurizers/skipatom_vectors_dim200.torch")), batch_size=batch_size)

    metal_val_sets = [featurize_sdf_for_megnet(path_to_sdf=val_sdf,
                                               mol_featurizer=ConvMolFeaturizer(),
                                               metal_featurizer=SkipatomFeaturizer("Source/featurizers/skipatom_vectors_dim200.torch"))
                      for val_sdf in val_sdfs]
    general_val_set = []
    for val_set in metal_val_sets: general_val_set += val_set
    val_data = DataLoader(general_val_set, batch_size=batch_size)
    val_batches = [Batch.from_data_list(val_set) for val_set in metal_val_sets]

    metal_test_sets = [featurize_sdf_for_megnet(path_to_sdf=test_sdf,
                                                mol_featurizer=ConvMolFeaturizer(),
                                                metal_featurizer=SkipatomFeaturizer("Source/featurizers/skipatom_vectors_dim200.torch"))
                       for test_sdf in test_sdfs]
    general_test_set = []
    for test_set in metal_test_sets: general_test_set += test_set
    test_data = DataLoader(general_test_set, batch_size=batch_size)

    model_parameters = MegnetParams(trial,
                                    n_megnet_lims=(3, 8),
                                    batch_norm_variants=(True, False),
                                    use_pre_dense_general_variants=(True, False),
                                    pre_dense_hidden_lims=(32, 256, 32),
                                    megnet_hidden_lims=(32, 256, 32),
                                    post_dense_hidden_lims=(32, 256, 32),
                                    n_post_dense_lims=(1, 4),
                                    actf_variants={"nn.LeakyReLU()": nn.LeakyReLU(),
                                                   "nn.PReLU()": nn.PReLU(),
                                                   "nn.Tanhshrink()": nn.Tanhshrink(), },
                                    optimizer_variants={"Adam": Adam,
                                                        "AdamW": AdamW,
                                                        "RMSprop": RMSprop, }).get()
    model_parameters["data"] = train_data.dataset
    model_parameters["batch_size"] = batch_size

    trainer = MegnetTrainer(
        model=MEGNet(**model_parameters),
        train_valid_data=((train_data, val_data),),
        test_data=test_data,
        output_folder=output_path,
        out_folder_mark=output_mark,
        epochs=epochs,
        es_patience=es_patience,
        verbose=True,
        seed=seed,
    )

    model, metrics = trainer.train_cv_models()
    trial.set_user_attr(key="metrics", value=metrics)
    trial.set_user_attr(key="model_parameters", value=model_parameters)

    max_error = max([mean_squared_error(batch.y, model(batch)) for batch in val_batches])

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

    model_parameters = {key: str(value) for key, value in study.best_trial.user_attrs["model_parameters"].items()}
    result = {
        "trials": len(study.trials),
        "started": str(start).split(".")[0],
        "finished": str(end).split(".")[0],
        "duration": str(end - start).split(".")[0],
        **study.best_trial.user_attrs["metrics"],
        "model_parameters": model_parameters,
    }

    time_mark = str(start).replace(" ", "_").replace("-", "_").replace(":", "_").split(".")[0]

    study.trials_dataframe().to_csv(path_or_buf=os.path.join(output_path, f"{output_mark}_{time_mark}.csv"),
                                    index=False)
    with open(os.path.join(output_path, f"{output_mark}_{time_mark}.json"), "w") as jf:
        json.dump(result, jf)
