import warnings
from inspect import signature

import torch.optim.optimizer

from Source.models.FCNN.model import FCNN
from Source.models.MEGNet.model import MEGNet
from Source.models.base_model import BaseModel
from Source.models.global_poolings import ConcatPooling


class MEGNetFCNN(BaseModel):
    def __init__(self, metal_features, node_features, edge_features, global_features, targets,
                 metal_fc_params=None, megnet_params=None, post_fc_params=None, global_pooling=ConcatPooling,
                 use_out_sequential=True,
                 optimizer=torch.optim.Adam, optimizer_parameters=None):
        super(MEGNetFCNN, self).__init__(targets, use_out_sequential, optimizer, optimizer_parameters)
        param_values = locals()
        self.config = {name: param_values[name] for name in signature(self.__init__).parameters.keys()}

        # preparing params for model blocks
        for p in [metal_fc_params, megnet_params, post_fc_params]:
            if "targets" in p and len(p["targets"]) > 0:
                warnings.warn(
                    "Not recommended to set 'targets' in model blocks as far as it doesn't affect anything",
                    DeprecationWarning)
            if "use_out_sequential" in p:
                warnings.warn("'use_out_sequential' parameter forcibly set to False", DeprecationWarning)
            p["use_out_sequential"] = False
            p["targets"] = ()

        metal_fc_params["input_features"] = metal_features
        self.metal_fc_sequential = FCNN(**metal_fc_params)

        megnet_params["node_features"] = node_features
        megnet_params["edge_features"] = edge_features
        megnet_params["global_features"] = global_features
        self.megnet_sequential = MEGNet(**megnet_params)

        self.global_pooling = global_pooling(input_dims=(self.megnet_sequential.output_dim,
                                                         self.metal_fc_sequential.output_dim))

        post_fc_params["input_features"] = self.global_pooling.output_dim
        self.post_fc_sequential = FCNN(**post_fc_params)

        self.last_common_dim = (self.global_pooling.output_dim, self.post_fc_sequential.output_dim)[-1]
        self.configure_out_layer()

    def forward(self, graph):
        x = self.megnet_sequential(graph)
        metal_x = self.metal_fc_sequential(graph.metal_x)
        general = self.global_pooling(x, metal_x)
        general = self.post_fc_sequential(general)
        if self.use_out_sequential:
            general = {target: sequential(general) for target, sequential in self.out_sequentials.items()}
        return general
