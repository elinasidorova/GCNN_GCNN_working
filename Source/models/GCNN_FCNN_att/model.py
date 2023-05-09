from inspect import signature

import torch.nn as nn
import torch.nn.functional as F
import torch.optim.optimizer
from pytorch_lightning import LightningModule
from torch import sqrt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import global_mean_pool as gap, Sequential
from torch_geometric.nn.conv import MFConv
from torch_geometric.utils import add_self_loops

from Source.models.GCNN_FCNN.model import GCNN_FCNN
from Source.models.GCNN_FCNN_att.attentions import GlobalAddAttention


class GCNNBimodalAtt(GCNN_FCNN):
    def __init__(self, attention_parameters=None,
                 attention_key_fc_params=None, attention_query_fc_params=None,
                 **kwargs):
        super().__init__(**kwargs)

        attention_parameters = attention_parameters or {"num_heads": 1}
        attention_key_fc_params = attention_key_fc_params or {}
        attention_query_fc_params = attention_query_fc_params or {}

        self.attention = GlobalAddAttention(attention_parameters=attention_parameters,
                                            key_fc_params=attention_key_fc_params,
                                            query_fc_params=attention_query_fc_params)

    def forward(self, graph):
        x = self.graph_sequential(graph)
        metal_x = self.metal_fc_sequential(graph.metal_x)
        general = self.global_pooling(x, metal_x)
        general = self.post_fc_sequential(general)
        if self.use_out_sequential:
            general = self.out_sequential(general)
        return general
