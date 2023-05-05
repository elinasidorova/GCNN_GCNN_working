import json
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


class GCNN(LightningModule):
    def __init__(self, node_features, num_targets,
                 hidden_pre_fc=(64,), pre_fc_dropout=0, pre_fc_actf=nn.LeakyReLU(),
                 hidden_conv=(64,), conv_dropout=0, conv_actf=nn.LeakyReLU(),
                 hidden_post_fc=(64,), post_fc_dropout=0, post_fc_bn=False, post_fc_actf=nn.LeakyReLU(),
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

        self.hidden_post_fc = hidden_post_fc
        self.post_fc_bn = post_fc_bn
        self.post_fc_dropout = post_fc_dropout
        self.post_fc_actf = post_fc_actf

        self.use_out_sequential = use_out_sequential

        self.optimizer = optimizer
        self.optimizer_parameters = optimizer_parameters or {}
        self.mode = mode

        self.pre_fc_sequential = self.make_fc_blocks(hidden_dims=(node_features, *hidden_pre_fc),
                                                     actf=pre_fc_actf,
                                                     batch_norm=False,
                                                     dropout=pre_fc_dropout)
        self.conv_sequential = self.make_conv_blocks(hidden_dims=((node_features, *hidden_pre_fc)[-1], *hidden_conv),
                                                     actf=conv_actf,
                                                     layer=conv_layer,
                                                     layer_parameters=conv_parameters,
                                                     dropout=conv_dropout)
        self.graph_pooling = graph_pooling
        self.post_fc_sequential = self.make_fc_blocks(
            hidden_dims=((node_features, *hidden_pre_fc, *hidden_conv)[-1], *hidden_post_fc),
            actf=post_fc_actf,
            batch_norm=post_fc_bn,
            dropout=post_fc_dropout)

        if self.use_out_sequential:
            self.output_dim = num_targets
            if self.mode == "regression":
                self.out_sequential = nn.Sequential(
                    nn.Linear((node_features, *hidden_pre_fc, *hidden_conv, *hidden_post_fc, *hidden_post_fc)[-1],
                              num_targets))
                self.loss = lambda *args, **kwargs: sqrt(F.mse_loss(*args, **kwargs))
            elif self.mode == "binary_classification":
                self.out_sequential = nn.Sequential(
                    nn.Linear((node_features, *hidden_pre_fc, *hidden_conv, *hidden_post_fc, *hidden_post_fc)[-1],
                              num_targets), nn.Sigmoid())
                self.loss = F.binary_cross_entropy
            elif self.mode == "multy_classification":
                self.out_sequential = nn.Sequential(
                    nn.Linear((node_features, *hidden_pre_fc, *hidden_conv, *hidden_post_fc, *hidden_post_fc)[-1],
                              num_targets), nn.Softmax())
                self.loss = F.cross_entropy
            else:
                raise ValueError(
                    "Invalid mode value, only 'regression', 'binary_classification' or 'multy_classification' are allowed")
        else:
            self.out_sequential = nn.Sequential()
            self.output_dim = (node_features, *hidden_pre_fc, *hidden_conv, *hidden_post_fc)[-1]

        self.valid_losses = []
        self.train_losses = []

    @staticmethod
    def make_fc_blocks(hidden_dims, actf, batch_norm=False, dropout=0.0):
        def fc_layer(in_f, out_f):
            layers = [nn.Linear(in_f, out_f), nn.Dropout(dropout), actf]
            if batch_norm: layers.insert(1, nn.BatchNorm1d(out_f))
            return nn.Sequential(*layers)

        lin_layers = [fc_layer(hidden_dims[i], hidden_dims[i + 1]) for i, val in enumerate(hidden_dims[:-1])]
        return nn.Sequential(*lin_layers)

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
        logits = self.forward(train_batch)
        loss = self.loss(train_batch.y, logits.reshape(*train_batch.y.shape))
        self.log('train_loss', loss, batch_size=train_batch.batch.max() + 1, prog_bar=True)
        return loss

    def validation_step(self, val_batch, *args, **kwargs):
        logits = self.forward(val_batch)
        loss = self.loss(val_batch.y, logits.reshape(*val_batch.y.shape))
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
