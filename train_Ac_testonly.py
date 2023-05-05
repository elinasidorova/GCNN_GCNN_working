import sys

import torch
from torch.nn import LeakyReLU, Softplus, LogSigmoid
from torch_geometric.nn import global_mean_pool, MFConv

from Source.metal_ligand_concat import concat_unifunc, max_unifunc

from Source.trainer import MolGraphHeteroNetTrainer
from Source.models.GCNN_FCNN.model_oldversion import GCNNBimodal
from Source.data import train_test_valid_split, get_num_node_features, get_batch_size, get_num_targets, \
    get_num_metal_features
from Source.featurizers.featurizers import featurize_sdf_with_metal, SkipatomFeaturizer, ConvMolFeaturizer
from torch_geometric.loader import DataLoader


def train_testonly(train_sdf, test_sdf, output_path, output_mark, seed, batch_size=10, n_split=10, epochs=1000,
                   es_patience=100, model_parameters=None):
    model_parameters = {} if model_parameters is None else model_parameters
    featurized_train = featurize_sdf_with_metal(path_to_sdf=train_sdf,
                                                mol_featurizer=ConvMolFeaturizer(),
                                                metal_featurizer=SkipatomFeaturizer("Source/featurizers/skipatom_vectors_dim200.torch"))

    folds = train_test_valid_split(featurized_train, n_split, test_ratio=0.1, batch_size=batch_size,
                                   subsample_size=False, return_test=False)

    featurized_test = featurize_sdf_with_metal(path_to_sdf=test_sdf,
                                               mol_featurizer=ConvMolFeaturizer(),
                                               metal_featurizer=SkipatomFeaturizer("Source/featurizers/skipatom_vectors_dim200.torch"))
    test_data = DataLoader(featurized_test, batch_size=batch_size)

    model = GCNNBimodal(
        node_features=get_num_node_features(folds[0]),
        metal_features=get_num_metal_features(folds[0]),
        num_targets=get_num_targets(folds[0]),
        batch_size=get_batch_size(folds[0]),
        **model_parameters,
    )

    trainer = MolGraphHeteroNetTrainer(
        model=model,
        train_valid_data=folds,
        test_data=test_data,
        output_folder=output_path,
        out_folder_mark=output_mark,
        epochs=epochs,
        es_patience=es_patience,
        seed=seed,
    )

    trainer.train_cv_models()


Ac = sys.argv[1]
seed = int(sys.argv[2])
i = int(sys.argv[3])

model_parameters = [{
    "hidden_metal": [200, 64, 128],
    "hidden_conv": [75, 256, 64],
    "hidden_linear": [192, 128],
    "metal_dropout": 0.0177117228723456,
    "conv_dropout": 0.2959383376253603,
    "linear_dropout": 0.1459422508196252,
    "metal_ligand_unifunc": concat_unifunc,
    "conv_layer": MFConv,
    "pooling_layer": global_mean_pool,
    "conv_actf": Softplus(),
    "linear_actf": LogSigmoid(),
    "linear_bn": True,
    "optimizer": torch.optim.Adam,
    "optimizer_parameters": None,
    "mode": "regression",
}, {
    'hidden_metal': [200, 256, 128, 128, 64, 64],
    'hidden_conv': [75, 128, 128, 64],
    'hidden_linear': [64, 256],
    'metal_dropout': 0.2510891227480936,
    'conv_dropout': 0.2793624333797553,
    'linear_dropout': 0.0669887915564103,
    'metal_ligand_unifunc': max_unifunc,
    'conv_layer': MFConv,
    'pooling_layer': global_mean_pool,
    'conv_actf': LeakyReLU(),
    'linear_actf': LeakyReLU(),
    'linear_bn': True,
    'optimizer': torch.optim.Adam,
    'optimizer_parameters': None,
    "mode": "regression",
}]

train_testonly(
    train_sdf=f"Data/LnAc_testonly/LnAc_{Ac}_testonly_train.sdf",
    test_sdf=f"Data/LnAc_testonly/LnAc_{Ac}_testonly_test.sdf",
    output_path="Output",
    output_mark=f"LnAc_testonly/seed{seed}/{Ac}/LnAc_{Ac}_{i}_seed{seed}_testonly_10folds",
    seed=seed,
    batch_size=32,
    n_split=10,
    epochs=1000,
    es_patience=100,
    model_parameters=model_parameters[i],
)
