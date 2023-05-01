import sys

import torch
from torch.nn import LeakyReLU, Softplus, LogSigmoid
from torch_geometric.nn import global_mean_pool, global_max_pool, MFConv

from Source.metal_ligand_concat import concat_unifunc, max_unifunc

sys.path.append("Source")

from Source.trainer import MolGraphHeteroNetTrainer
from Source.models.GCNN_bimodal import MolGraphHeteroNet
from Source.data import train_test_valid_split, get_num_node_features, get_num_metal_features, get_num_targets, \
    get_batch_size
from Source.featurizers.mol_featurizer import featurize_sdf_with_metal, SkipatomFeaturizer, ConvMolFeaturizer
from torch_geometric.loader import DataLoader

n_split = 10
batch_size = 64
train_sdf = "Data/TransPb_train.sdf"
test_sdf = "Data/TransPb_test.sdf"
output_path = "Output"

featurized_train = featurize_sdf_with_metal(path_to_sdf=train_sdf,
                                            mol_featurizer=ConvMolFeaturizer(),
                                            metal_featurizer=SkipatomFeaturizer("Source/featurizers/skipatom_vectors_dim200.torch"))

folds = train_test_valid_split(featurized_train, n_split, test_ratio=0.1, batch_size=batch_size,
                               subsample_size=False, return_test=False)

featurized_test = featurize_sdf_with_metal(path_to_sdf=test_sdf,
                                           mol_featurizer=ConvMolFeaturizer(),
                                           metal_featurizer=SkipatomFeaturizer("Source/featurizers/skipatom_vectors_dim200.torch"))
test_data = DataLoader(featurized_test, batch_size=batch_size)

model_parameters = [{
    "hidden_metal": [200, 128, 256, 64],
    "hidden_conv": [75, 256, 64],
    "hidden_linear": [128, 64, 64],
    "metal_dropout": 0.21535065146018023,
    "conv_dropout": 0.46671155052604313,
    "linear_dropout": 0.01285924860803213,
    "metal_ligand_unifunc": concat_unifunc,
    "conv_layer": MFConv,
    "pooling_layer": global_max_pool,
    "conv_actf": LeakyReLU(),
    "linear_actf": Softplus(),
    "linear_bn": False,
    "optimizer": torch.optim.Adam,
    "optimizer_parameters": None,
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
}, {
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
}]

seed = int(sys.argv[1])
i = int(sys.argv[2])

model = MolGraphHeteroNet(
    node_features=get_num_node_features(folds[0][0]),
    metal_features=get_num_metal_features(folds[0][0]),
    num_targets=get_num_targets(folds[0][0]),
    batch_size=get_batch_size(folds[0][0]),
    **model_parameters[i],
    mode="regression",
)

trainer = MolGraphHeteroNetTrainer(
    model=model,
    train_valid_data=folds,
    test_data=test_data,
    output_folder=output_path,
    out_folder_mark=f"TransPb_seed{seed}_{i}",
    epochs=1000,
    es_patience=100,
    seed=seed,
)

trainer.train_cv_models()
