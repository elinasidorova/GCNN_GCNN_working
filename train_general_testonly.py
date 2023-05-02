import random
import sys

from loguru import logger
from sklearn.model_selection import train_test_split, KFold
from torch import nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MFConv, global_mean_pool

from Source.data import get_num_node_features, get_num_metal_features, get_num_targets, get_batch_size
from Source.metal_ligand_concat import max_unifunc
from Source.models.GCNN_bimodal import MolGraphHeteroNet
from Source.featurizers.featurizers import featurize_sdf_with_metal, ConvMolFeaturizer, SkipatomFeaturizer
from Source.trainer import MolGraphHeteroNetTrainer


def create_folds(metals_list, n_folds, batch_size, shuffle_every_epoch, seed=17):
    train = [[] for _ in range(n_folds)]
    val = [[] for _ in range(n_folds)]
    for metal in metals_list:
        logger.info(f"Reading {metal} complexes for train set...")
        mol_list = featurize_sdf_with_metal(path_to_sdf=f"Data/OneMetal/{metal}.sdf",
                                            mol_featurizer=ConvMolFeaturizer(),
                                            metal_featurizer=SkipatomFeaturizer("Source/featurizers/skipatom_vectors_dim200.torch"))
        mol_list = (mol_list * n_folds)[:n_folds] if len(mol_list) < n_folds else mol_list
        mol_ids = list(range(len(mol_list)))
        for fold_ind, (train_index, valid_index) in enumerate(KFold(n_splits=n_folds).split(mol_ids)):
            train[fold_ind] += [val for i, val in enumerate(mol_list) if i in train_index]
            val[fold_ind] += [val for i, val in enumerate(mol_list) if i in valid_index]
            random.Random(seed).shuffle(train[fold_ind])
            random.Random(seed).shuffle(val[fold_ind])

    train_loaders = [DataLoader(train[fold_ind], batch_size=batch_size, shuffle=shuffle_every_epoch)
                     for fold_ind in range(n_folds)]
    valid_loaders = [DataLoader(val[fold_ind], batch_size=batch_size, shuffle=shuffle_every_epoch)
                     for fold_ind in range(n_folds)]
    folds = [(t_loader, v_loader) for t_loader, v_loader in zip(train_loaders, valid_loaders)]
    return folds


def create_train_val(metals_list, batch_size, shuffle_every_epoch, seed=17):
    train = []
    val = []
    for metal in metals_list:
        logger.info(f"Reading {metal} complexes for train set...")
        mol_list = featurize_sdf_with_metal(path_to_sdf=f"Data/OneMetal/{metal}.sdf",
                                            mol_featurizer=ConvMolFeaturizer(),
                                            metal_featurizer=SkipatomFeaturizer("Source/featurizers/skipatom_vectors_dim200.torch"))
        random.Random(seed).shuffle(mol_list)
        mol_ids = list(range(len(mol_list)))
        train_index, valid_index = train_test_split(mol_ids, test_size=0.2, random_state=seed, shuffle=False)
        train += [mol for i, mol in enumerate(mol_list) if i in train_index]
        val += [mol for i, mol in enumerate(mol_list) if i in valid_index]

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=shuffle_every_epoch)
    valid_loader = DataLoader(val, batch_size=batch_size, shuffle=shuffle_every_epoch)
    return train_loader, valid_loader


def train_testonly(test_metal, output_path, output_mark, batch_size=64, n_split=10, epochs=1000,
                   es_patience=100, model_parameters=None, shuffle_every_epoch=False, seed=17):
    model_parameters = {} if model_parameters is None else model_parameters

    if n_split == 1:
        train_loader, valid_loader = create_train_val(metals_list=[m for m in all_metals if m != test_metal],
                                                      batch_size=batch_size,
                                                      shuffle_every_epoch=True, seed=seed)
        folds = ((train_loader, valid_loader),)
    else:
        folds = create_folds(
            metals_list=[m for m in all_metals if m != test_metal],
            n_folds=n_split,
            batch_size=batch_size,
            shuffle_every_epoch=shuffle_every_epoch,
            seed=seed)
    test_loader = DataLoader(
        featurize_sdf_with_metal(path_to_sdf=f"Data/OneMetal/{test_metal}.sdf",
                                 mol_featurizer=ConvMolFeaturizer(),
                                 metal_featurizer=SkipatomFeaturizer("Source/featurizers/skipatom_vectors_dim200.torch")),
        batch_size=batch_size)

    model = MolGraphHeteroNet(
        node_features=get_num_node_features(folds[0]),
        metal_features=get_num_metal_features(folds[0]),
        num_targets=get_num_targets(folds[0]),
        batch_size=get_batch_size(folds[0]),
        **model_parameters,
    )
    trainer = MolGraphHeteroNetTrainer(
        model=model,
        train_valid_data=folds,
        test_data=test_loader,
        output_folder=output_path,
        out_folder_mark=output_mark,
        epochs=epochs,
        es_patience=es_patience,
        seed=seed)
    trainer.train_cv_models()


test_metal = sys.argv[1]
seed = 23

transition_metals = ["Mg", "Al", "Ca", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Y", "Ag", "Cd", "Hg", "Pb", "Bi"]
Ln_metals = ["La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]
Ac_metals = ["Th", "Am", "Cm", "Bk", "Cf"]  # "U", "Pu",
all_metals = transition_metals + Ln_metals + Ac_metals

model_parameters = {
    "hidden_metal": [200, 256, 128, 128, 64, 64],
    "hidden_conv": [75, 128, 128, 64],
    "hidden_linear": [64, 256],
    "metal_dropout": 0.25108912274809364,
    "conv_dropout": 0.27936243337975536,
    "linear_dropout": 0.06698879155641034,
    "metal_ligand_unifunc": max_unifunc,
    "conv_layer": MFConv,
    "pooling_layer": global_mean_pool,
    "conv_actf": nn.LeakyReLU(),
    "linear_actf": nn.LeakyReLU(),
    'linear_bn': False,
    'metal_bn': False,
    "optimizer": Adam,
    "optimizer_parameters": {},
    'mode': 'regression'
}

train_testonly(
    test_metal=test_metal,
    output_path="Output",
    output_mark=f"GeneralModel/General_{test_metal}_seed{seed}_testonly_10folds",
    batch_size=32,
    n_split=10,
    epochs=1000,
    es_patience=100,
    model_parameters=model_parameters,
    seed=seed,
    shuffle_every_epoch=True)
