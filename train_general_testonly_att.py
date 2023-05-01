import random
import sys

from loguru import logger
from sklearn.model_selection import train_test_split, KFold
from torch.nn import LeakyReLU, PReLU
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MFConv, global_max_pool

from Source.data import get_num_node_features, get_num_metal_features, get_num_targets, get_batch_size
from Source.metal_ligand_concat import max_unifunc
from Source.models.model_with_attention import MolGraphHeteroNet
from Source.featurizers.mol_featurizer import featurize_sdf_with_metal, ConvMolFeaturizer, SkipatomFeaturizer
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


def train_testonly(test_metal, output_path, output_mark, n_folds=1, batch_size=64, epochs=1000,
                   es_patience=100, model_parameters=None, shuffle_every_epoch=False, seed=17):
    model_parameters = {} if model_parameters is None else model_parameters

    if n_folds == 1:
        folds = (create_train_val(
            metals_list=[m for m in all_metals if m != test_metal],
            batch_size=batch_size,
            shuffle_every_epoch=shuffle_every_epoch,
            seed=seed),)
    else:
        folds = create_folds(metals_list=[m for m in all_metals if m != test_metal],
                             n_folds=n_folds,
                             batch_size=batch_size,
                             shuffle_every_epoch=shuffle_every_epoch,
                             seed=seed)
    test_loader = DataLoader(
        featurize_sdf_with_metal(path_to_sdf=f"Data/OneMetal/{test_metal}.sdf",
                                 mol_featurizer=ConvMolFeaturizer(),
                                 metal_featurizer=SkipatomFeaturizer("Source/featurizers/skipatom_vectors_dim200.torch")),
        batch_size=batch_size)

    model = MolGraphHeteroNet(
        node_features=get_num_node_features(folds[0][0]),
        metal_features=get_num_metal_features(folds[0][0]),
        num_targets=get_num_targets(folds[0][0]),
        batch_size=get_batch_size(folds[0][0]),
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
attention_name = "GlobalAddAttention"  # sys.argv[2]
seed = 23

transition_metals = ["Mg", "Al", "Ca", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Y", "Ag", "Cd", "Hg", "Pb", "Bi"]
Ln_metals = ["La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]
Ac_metals = ["Th", "Am", "Cm", "Bk", "Cf"]  # "U", "Pu",
all_metals = transition_metals + Ln_metals + Ac_metals

model_parameters = {
    'metal_ligand_unifunc': max_unifunc,
    'hidden_conv': [256],
    'hidden_metal': [64, 64, 64, 256],
    'hidden_linear': [256, 64, 64, 128],
    'conv_layer': MFConv,
    'pooling_layer': global_max_pool,
    'conv_dropout': 0.4593056421053323,
    'metal_dropout': 0.0,
    'linear_dropout': 0.0,
    'linear_bn': False,
    'metal_bn': False,
    'conv_actf': PReLU(num_parameters=1),
    'linear_actf': LeakyReLU(negative_slope=0.01),
    'use_attention': True,
    'attention_name': attention_name,
    'attention_parameters': {'num_heads': 4},
    'attention_key_hidden': (),
    'attention_query_hidden': (),
    'attention_key_actf': LeakyReLU(negative_slope=0.01),
    'attention_query_actf': LeakyReLU(negative_slope=0.01),
    'attention_key_bn': False,
    'attention_query_bn': False,
    'optimizer': Adam,
    'optimizer_parameters': {},
    'mode': 'regression',
}

train_testonly(
    test_metal=test_metal,
    output_path="Output",
    output_mark=f"GeneralModelAtt/{test_metal}_{attention_name}_seed{seed}_testonly_1fold",
    n_folds=1,
    batch_size=10,
    epochs=1000,
    es_patience=100,
    model_parameters=model_parameters,
    seed=seed,
    shuffle_every_epoch=True)
