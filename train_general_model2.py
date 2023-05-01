from skipatom import SkipAtomInducedModel
from torch import nn
from torch.optim import RMSprop
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MFConv, global_mean_pool

from Source.data import get_num_node_features, get_num_metal_features, get_num_targets, get_batch_size
from Source.metal_ligand_concat import mean_unifunc
from Source.model import MolGraphHeteroNet
from Source.mol_featurizer import featurize_sdf_with_metal, ConvMolFeaturizer, SkipatomFeaturizer
from Source.trainer import MolGraphHeteroNetTrainer

train_sdf = "Data/OptunaMgCdLa/train.sdf"
test_sdfs = ["Data/OptunaMgCdLa/Mg_val.sdf", "Data/OptunaMgCdLa/Cd_val.sdf", "Data/OptunaMgCdLa/La_val.sdf"]
output_path = "Output"
output_mark = f"General_MgCdLa"

batch_size = 128
epochs = 1000
es_patience = 100

train_data = DataLoader(featurize_sdf_with_metal(path_to_sdf=train_sdf,
                                                 mol_featurizer=ConvMolFeaturizer(),
                                                 metal_featurizer=SkipatomFeaturizer(models=[
                                                     SkipAtomInducedModel.load(
                                                         "skipatom_models/mp_2020_10_09.dim200.model",
                                                         "skipatom_models/mp_2020_10_09.training.data",
                                                         min_count=2e7, top_n=5
                                                     ),
                                                     SkipAtomInducedModel.load(
                                                         "skipatom_models/AmBkCfCm_2022_11_23.dim200.model",
                                                         "skipatom_models/AmBkCfCm_2022_11_23.training.data",
                                                         min_count=2e7, top_n=5
                                                     ),
                                                 ])), batch_size=batch_size)

metal_val_sets = [featurize_sdf_with_metal(path_to_sdf=test_sdf,
                                           mol_featurizer=ConvMolFeaturizer(),
                                           metal_featurizer=SkipatomFeaturizer(models=[
                                               SkipAtomInducedModel.load(
                                                   "skipatom_models/mp_2020_10_09.dim200.model",
                                                   "skipatom_models/mp_2020_10_09.training.data",
                                                   min_count=2e7, top_n=5
                                               ),
                                               SkipAtomInducedModel.load(
                                                   "skipatom_models/AmBkCfCm_2022_11_23.dim200.model",
                                                   "skipatom_models/AmBkCfCm_2022_11_23.training.data",
                                                   min_count=2e7, top_n=5
                                               ),
                                           ]))
                  for test_sdf in test_sdfs]
general_val_set = []
for val_set in metal_val_sets: general_val_set += val_set

val_data = DataLoader(general_val_set, batch_size=batch_size)
test_batches = [Batch.from_data_list(val_set) for val_set in metal_val_sets]

model_parameters = {
    "node_features": get_num_node_features(train_data),
    "metal_features": get_num_metal_features(train_data),
    "num_targets": get_num_targets(train_data),
    "batch_size": get_batch_size(train_data),
    "metal_ligand_unifunc": mean_unifunc,
    "hidden_conv": [75, 128, 256],
    "hidden_metal": [200, 128, 128, 256],
    "hidden_linear": [256, 128, 64],
    "conv_layer": MFConv,
    "pooling_layer": global_mean_pool,
    "conv_dropout": 0.417478267028345,
    "metal_dropout": 0.0,
    "linear_dropout": 0.0,
    "linear_bn": False,
    "metal_bn": False,
    "conv_actf": nn.Tanhshrink(),
    "linear_actf": nn.LeakyReLU(),
    "optimizer": RMSprop,
    "optimizer_parameters": {},
    "mode": "regression"}

trainer = MolGraphHeteroNetTrainer(
    model=MolGraphHeteroNet(**model_parameters),
    train_valid_data=((train_data, val_data),),
    output_folder=output_path,
    out_folder_mark=output_mark,
    epochs=epochs,
    es_patience=es_patience,
    verbose=True,
)

trainer.train_cv_models()
