import torch
from dgllife.utils import AttentiveFPAtomFeaturizer, PAGTNAtomFeaturizer
from dgllife.utils import CanonicalAtomFeaturizer
from torch import nn
from torch_geometric.nn import global_mean_pool, MFConv, global_max_pool, SAGEConv, GATConv, TransformerConv

from Source.models.GCNN_FCNN.featurizers import SkipatomFeaturizer
from Source.models.global_poolings import MaxPooling, ConcatPooling, SumPooling, CrossAttentionPooling

# is used below in METAL_FC_PARAMS, GCNN_PARAMS, POST_FC_PARAMS
ACTIVATION_VARIANTS = {
    "LeakyReLU": nn.LeakyReLU(),
    "PReLU": nn.PReLU(),
    "Tanhshrink": nn.Tanhshrink(),
}
DIM_LIMS = (64, 1024)

# general parameters
LR_LIMS = (1e-2, 1e-2)
BATCH_SIZE_LIMS = (8, 64)
OPTIMIZER_VARIANTS = {
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
    "RMSprop": torch.optim.RMSprop,
    "SGD": torch.optim.SGD
}
FEATURIZER_VARIANTS = {
    "CanonicalAtomFeaturizer": CanonicalAtomFeaturizer(),
    "AttentiveFPAtomFeaturizer": AttentiveFPAtomFeaturizer(),
    "PAGTNAtomFeaturizer": PAGTNAtomFeaturizer(),
}
METAL_FEATURIZER_VARIANTS = {"SkipatomFeaturizer": SkipatomFeaturizer()}

# model parameters
METAL_FC_PARAMS = {
    "dim_lims": DIM_LIMS,
    "n_layers_lims": (1, 4),
    "actf_variants": ACTIVATION_VARIANTS,
    "dropout_lims": (0, 0),
    "bn_variants": (True, False),
}
GCNN_PARAMS = {
    "pre_fc_params": {
        "dim_lims": DIM_LIMS,
        "n_layers_lims": (1, 2),
        "actf_variants": ACTIVATION_VARIANTS,
        "dropout_lims": (0, 0),
        "bn_variants": (True, False),
    },
    "post_fc_params": {
        "dim_lims": DIM_LIMS,
        "n_layers_lims": (1, 2),
        "actf_variants": ACTIVATION_VARIANTS,
        "dropout_lims": (0, 0),
        "bn_variants": (True, False),
    },
    "n_conv_lims": (1, 3),
    "dropout_lims": (0.1, 0.5),
    "actf_variants": ACTIVATION_VARIANTS,
    "dim_lims": DIM_LIMS,
    "conv_layer_variants": {
        "MFConv": MFConv,
        "GATConv": GATConv,
        "TransformerConv": TransformerConv,
        "SAGEConv": SAGEConv,
    },
    "pooling_layer_variants": {
        "global_mean_pool": global_mean_pool,
        "global_max_pool": global_max_pool,
    },
}
POST_FC_PARAMS = {
    "dim_lims": DIM_LIMS,
    "n_layers_lims": (1, 4),
    "actf_variants": ACTIVATION_VARIANTS,
    "dropout_lims": (0, 0),
    "bn_variants": (True, False),
}
GLOBAL_POOLING_VARIANTS = {
    "ConcatPooling": ConcatPooling,
    "SumPooling": SumPooling,
    "MaxPooling": MaxPooling,
    "CrossAttentionPooling": CrossAttentionPooling,
}
