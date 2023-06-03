import json
import warnings
from inspect import signature

import torch.nn as nn
import torch.nn.functional as F
import torch.optim.optimizer
from pytorch_lightning import LightningModule
from torch import sqrt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import global_mean_pool, Sequential
from torch_geometric.nn.conv import MFConv
from torch_geometric.utils import add_self_loops

from Source.models.FCNN.model import FCNN


class GCNN(LightningModule):
    def __init__(self, node_features, num_targets,
                 pre_fc_params=None, hidden_conv=(64,), conv_dropout=0, conv_actf=nn.LeakyReLU(), post_fc_params=None,
                 conv_layer=MFConv, conv_parameters=None, graph_pooling=global_mean_pool,
                 use_out_sequential=True,
                 optimizer=torch.optim.Adam, optimizer_parameters=None, mode="regression"):
        super(GCNN, self).__init__()
        param_values = locals()
        self.config = {name: param_values[name] for name in signature(self.__init__).parameters.keys()}

        self.node_features = node_features
        self.num_targets = num_targets

        self.conv_layer = conv_layer
        self.conv_parameters = conv_parameters or {}
        self.hidden_conv = hidden_conv
        self.conv_dropout = conv_dropout
        self.conv_actf = conv_actf

        # preparing params for fully connected blocks
        for p in [pre_fc_params, post_fc_params]:
            if "num_targets" in p:
                warnings.warn(
                    "Not recommended to set 'num_targets' in model blocks as far as it doesn't affect anything",
                    DeprecationWarning)
            if "use_out_sequential" in p:
                warnings.warn("'use_out_sequential' parameter forcibly set to False", DeprecationWarning)
            p["use_out_sequential"] = False
            p["num_targets"] = 1
        if "use_bn" in pre_fc_params and pre_fc_params["use_bn"]:
            warnings.warn("Can't use batch normalization in FCNN on separate nodes")
            pre_fc_params["use_bn"] = False

        self.use_out_sequential = use_out_sequential

        self.optimizer = optimizer
        self.optimizer_parameters = optimizer_parameters or {}
        self.mode = mode

        pre_fc_params["input_features"] = node_features
        self.pre_fc_sequential = FCNN(**pre_fc_params)
        self.conv_sequential = self.make_conv_blocks(hidden_dims=(self.pre_fc_sequential.output_dim, *hidden_conv),
                                                     actf=conv_actf,
                                                     layer=conv_layer,
                                                     layer_parameters=conv_parameters,
                                                     dropout=conv_dropout)
        self.graph_pooling = graph_pooling
        post_fc_params["input_features"] = (self.pre_fc_sequential.output_dim, *hidden_conv)[-1]
        self.post_fc_sequential = FCNN(**post_fc_params)

        if self.use_out_sequential:
            self.output_dim = num_targets
            if self.mode == "regression":
                self.out_sequential = nn.Sequential(
                    nn.Linear(self.post_fc_sequential.output_dim, num_targets))
                self.loss = lambda *args, **kwargs: sqrt(F.mse_loss(*args, **kwargs))
            elif self.mode == "binary_classification":
                self.out_sequential = nn.Sequential(
                    nn.Linear(self.post_fc_sequential.output_dim, num_targets), nn.Sigmoid())
                self.loss = F.binary_cross_entropy
            elif self.mode == "multiclass_classification":
                self.out_sequential = nn.Sequential(
                    nn.Linear(self.post_fc_sequential.output_dim, num_targets), nn.Softmax())
                self.loss = F.cross_entropy
            else:
                raise ValueError(
                    "Invalid mode value, only 'regression', 'binary_classification' or 'multiclass_classification' are allowed")
        else:
            self.out_sequential = nn.Sequential()
            self.output_dim = self.post_fc_sequential.output_dim

        self.valid_losses = []
        self.train_losses = []

    @staticmethod
    def make_conv_blocks(hidden_dims, actf, layer, layer_parameters=None, dropout=0.0):
        layer_parameters = layer_parameters or {}

        def conv_block(in_f, out_f):
            layers = [(layer(in_f, out_f, **layer_parameters), 'x, edge_index -> x'),
                      nn.Dropout(dropout),
                      actf]
            return Sequential("x, edge_index", layers)

        conv_layers = [(conv_block(hidden_dims[i], hidden_dims[i + 1]), 'x, edge_index -> x')
                       for i in range(len(hidden_dims) - 1)]
        return Sequential("x, edge_index", conv_layers)

    def forward(self, graph):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.pre_fc_sequential(x)
        x = self.conv_sequential(x, edge_index)
        x = self.graph_pooling(x, batch=batch)
        x = self.post_fc_sequential(x)
        if self.use_out_sequential:
            x = self.out_sequential(x)
        return x

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), **self.optimizer_parameters)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, factor=0.2, patience=20, verbose=True),
                "monitor": "val_loss",
                "frequency": 1  # should be set to "trainer.check_val_every_n_epoch"
            },
        }

    def training_step(self, train_batch, *args, **kwargs):
        loss = self.loss(self.forward(train_batch), train_batch.y)
        self.log('train_loss', loss, batch_size=train_batch.batch.max() + 1, prog_bar=True)
        return loss

    def validation_step(self, val_batch, *args, **kwargs):
        loss = self.loss(self.forward(val_batch), val_batch.y)
        self.log('val_loss', loss, batch_size=val_batch.batch.max() + 1)
        return loss

    def get_model_structure(self):
        def make_jsonable(x):
            try:
                json.dumps(x)
                return x
            except (TypeError, OverflowError):
                if isinstance(x, dict):
                    return {key: make_jsonable(value) for key, value in x.items()}
                return str(x)

        return make_jsonable(self.config)
