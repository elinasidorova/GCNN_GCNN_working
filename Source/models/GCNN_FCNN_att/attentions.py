import warnings

import torch.nn as nn
import torch.optim.optimizer
from torch.nn.utils.rnn import pad_sequence

from Source.models.FCNN.model import FCNN


class BaseAttention(nn.Module):
    def __init__(self):
        super(BaseAttention, self).__init__()

    def forward(self, graph):
        return graph


class GlobalAddAttention(BaseAttention):
    def __init__(self, attention_parameters=None,
                 key_fc_params=None, query_fc_params=None):
        super(GlobalAddAttention, self).__init__()

        key_fc_params = key_fc_params or {}
        query_fc_params = query_fc_params or {}
        attention_parameters = attention_parameters or {"num_heads": 1}

        for p in [key_fc_params, query_fc_params]:
            if "num_targets" in p:
                warnings.warn("'num_targets' doesn't affect anything", DeprecationWarning)
            if "use_out_sequential" in p:
                warnings.warn("'use_out_sequential' parameter forcibly set to False", DeprecationWarning)
            p["use_out_sequential"] = False
            p["num_targets"] = 1

        self.key_sequential = FCNN(**key_fc_params)
        self.query_sequential = FCNN(**query_fc_params)
        self.attention = torch.nn.MultiheadAttention(**attention_parameters)

    def forward(self, graph):
        x, metal_x, batch = graph.x, graph.metal_x, graph.batch
        _, counts = torch.unique(batch, return_counts=True)

        # mask.shape == (num_graphs, max(nodes_per_graph) + 1)
        mask = pad_sequence([torch.ones(s) for s in counts + 1], batch_first=True).to(bool).to(x.device)
        metal_x_mask = torch.zeros_like(mask).to(bool).to(x.device)
        metal_x_mask[:, 0] = True
        x_mask = mask * (~metal_x_mask)

        # value.shape == (max(nodes_per_graph) + 1, num_graphs, node_features)
        value = pad_sequence([torch.cat((metal_x[i].unsqueeze(0), x[batch == i]), dim=0) for i in batch.unique()])
        key = self.key_sequential(value)
        query = self.query_sequential(value)

        # out.shape == (max(nodes_per_graph) + 1, num_graphs, node_features)
        out, _ = self.attention(query, key, value, key_padding_mask=~mask)

        graph.x = out[x_mask.T, :]
        graph.metal_x = out[metal_x_mask.T, :]

        return graph
