import copy
import random
import sys

import torch
from loguru import logger
from rdkit import Chem
from sklearn.model_selection import KFold, train_test_split
from torch import nn
from torch.optim import AdamW
from torch_geometric.loader import DataLoader

from Source.models.megnet_model import MEGNet
from Source.featurizers.featurizers import ConvMolFeaturizer, SkipatomFeaturizer
from Source.trainer import MegnetTrainer


def create_train_val(metals_list, batch_size, shuffle_every_epoch, seed=17):
    train = []
    val = []
    for metal in metals_list:
        logger.info(f"Reading {metal} complexes for train set...")
        mol_list = featurize_sdf_for_megnet(path_to_sdf=f"Data/OneMetal/{metal}.sdf",
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


def create_folds(metals_list, n_folds, batch_size, shuffle_every_epoch, seed=17):
    train = [[] for _ in range(n_folds)]
    val = [[] for _ in range(n_folds)]
    for metal in metals_list:
        logger.info(f"Reading {metal} complexes for train set...")
        mol_list = featurize_sdf_for_megnet(path_to_sdf=f"Data/OneMetal/{metal}.sdf",
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


transition_metals = ["Mg", "Al", "Ca", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Y", "Ag", "Cd", "Hg", "Pb", "Bi"]
Ln_metals = ["La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]
Ac_metals = ["Th", "Am", "Cm", "Bk", "Cf"]  # "U", "Pu",
all_metals = transition_metals + Ln_metals + Ac_metals

seed = 42
test_metal = sys.argv[1]
output_path = "Output/MegnetTrans"
output_mark = f"{test_metal}_seed42_testonly_fold1"
shuffle_every_epoch = True
# n_split = 1
batch_size = 64
epochs = 1000
es_patience = 100

train_loader, val_loader = create_train_val(metals_list=[m for m in all_metals if m != test_metal],
                                            batch_size=batch_size,
                                            shuffle_every_epoch=shuffle_every_epoch,
                                            seed=seed)
test_loader = DataLoader(
    featurize_sdf_for_megnet(path_to_sdf=f"Data/OneMetal/{test_metal}.sdf",
                             mol_featurizer=ConvMolFeaturizer(),
                             metal_featurizer=SkipatomFeaturizer("Source/featurizers/skipatom_vectors_dim200.torch")),
    batch_size=batch_size)

model_parameters = {
    "pre_dense_edge_hidden": (),
    "pre_dense_node_hidden": (224,),
    "pre_dense_general_hidden": (),
    "megnet_dense_hidden": (256, 256, 256, 256),
    "megnet_edge_conv_hidden": (256,),
    "megnet_node_conv_hidden": (256,),
    "megnet_general_conv_hidden": (256,),
    "post_dense_hidden": (256, 128, 64, 32),
    "pool": "global_mean_pool",
    "pool_order": "early",
    "batch_norm": False,
    "batch_track_stats": True,
    "actf": nn.LeakyReLU(),
    "dropout_rate": 0.0,
    "optimizer": AdamW,
    "optimizer_parameters": None,
}

model = MEGNet(
    train_loader.dataset,
    batch_size,
    **model_parameters)

trainer = MegnetTrainer(
    model=model,
    train_valid_data=((train_loader, val_loader),),
    test_data=test_loader,
    output_folder=output_path,
    out_folder_mark=output_mark,
    epochs=epochs,
    es_patience=es_patience,
    seed=seed)
trainer.train_cv_models()
