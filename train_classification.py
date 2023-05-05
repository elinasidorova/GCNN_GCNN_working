import sys

import torch
from torch.nn import LeakyReLU, Softplus
from torch_geometric.nn import global_max_pool, MFConv

sys.path.append("Source")

from Source.trainer import ModelTrainer
from Source.models.GCNN_FCNN import MolGraphNet
from Source.data import train_test_valid_split, get_num_node_features, get_num_targets, \
    get_batch_size
from Source.featurizers.featurizers import featurize_sdf_with_metal, SkipatomFeaturizer, ConvMolFeaturizer
from torch_geometric.loader import DataLoader

n_split = 10
batch_size = 64
train_sdf = "Data/Cu_classification_train.sdf"
test_sdf = "Data/Cu_classification_test.sdf"
output_path = "Output"
output_mark = "With_test"

featurized_train = featurize_sdf_with_metal(path_to_sdf=train_sdf,
                                            mol_featurizer=ConvMolFeaturizer(),
                                            metal_featurizer=SkipatomFeaturizer("Source/featurizers/skipatom_vectors_dim200.torch"))

folds = train_test_valid_split(featurized_train, n_split, test_ratio=0.1, batch_size=batch_size, subsample_size=False, return_test=False)

featurized_test = featurize_sdf_with_metal(path_to_sdf=test_sdf,
                                           mol_featurizer=ConvMolFeaturizer(),
                                           metal_featurizer=SkipatomFeaturizer("Source/featurizers/skipatom_vectors_dim200.torch"))
test_data = DataLoader(featurized_test, batch_size=batch_size)

model = MolGraphNet(
    node_features=get_num_node_features(folds[0][0]),
    # metal_features=get_num_metal_features(folds[0][0]),
    num_targets=get_num_targets(folds[0][0]),
    batch_size=get_batch_size(folds[0][0]),
    conv_layer=MFConv,
    pooling_layer=global_max_pool,
    # metal_ligand_unifunc=concat_unifunc,
    # hidden_metal=[200, 128, 256, 64],
    hidden_conv=[75, 256, 64],
    hidden_linear=[64, 64, 64],
    # metal_dropout=0.21535065146018023,
    conv_dropout=0.46671155052604313,
    linear_dropout=0.01285924860803213,
    linear_bn=False,
    conv_actf=LeakyReLU(),
    linear_actf=Softplus(),
    optimizer=torch.optim.Adam,
    optimizer_parameters=None,
    mode="binary_classification",
)

trainer = ModelTrainer(
    model=model,
    train_valid_data=folds,
    test_data=test_data,
    output_folder=output_path,
    out_folder_mark=output_mark,
    epochs=1000,
    es_patience=100,
)

trainer.train_cv_models()
