import json
from inspect import signature

import torch.nn as nn
import torch.optim.optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn.conv import MFConv, GCNConv, GraphConv
from torch_geometric.nn import global_max_pool, global_mean_pool, BatchNorm, Sequential
from torch_geometric.utils import add_self_loops
from torch_geometric.nn.pool import TopKPooling
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import sqrt
import warnings

from Source.models.global_poolings import ConcatPooling


class MolGraphNet(LightningModule):
    def __init__(self, node_features, num_targets, batch_size, conv_layer=MFConv, pooling_layer=gap, hidden_conv=None,
                 hidden_fc=None,
                 conv_dropout=0, fc_dropout=0, fc_bn=False, conv_actf=nn.LeakyReLU(),
                 fc_actf=nn.LeakyReLU(), optimizer=torch.optim.Adam, optimizer_parameters=None, mode="regression"):
        super().__init__()
        self.config = {
            "node_features": node_features,
            "num_targets": num_targets,
            "batch_size": batch_size,
            "conv_layer": conv_layer,
            "pooling_layer": pooling_layer,
            "hidden_conv": hidden_conv,
            "hidden_fc": hidden_fc,
            "conv_dropout": conv_dropout,
            "fc_dropout": fc_dropout,
            "fc_bn": fc_bn,
            "conv_actf": conv_actf,
            "fc_actf": fc_actf,
            "optimizer": optimizer,
            "optimizer_parameters": optimizer_parameters,
            "mode": mode
        }

        if hidden_conv is None:
            hidden_conv = [node_features, 64, 64]

        if hidden_fc is None:
            hidden_fc = [hidden_conv[-1], 256, 256]

        if optimizer_parameters is None:
            optimizer_parameters = {}
        self.node_features = node_features
        self.num_targets = num_targets
        self.batch_size = batch_size
        self.conv_layer = conv_layer
        self.graph_pooling = pooling_layer
        self.hidden_conv = hidden_conv
        self.hidden_fc = hidden_fc
        self.n_conv = len(hidden_conv) - 1
        self.n_fc = len(hidden_fc) - 1
        self.conv_dropout = conv_dropout
        self.fc_dropout = fc_dropout
        self.fc_bn = fc_bn
        self.conv_actf = conv_actf
        self.fc_actf = fc_actf
        self.optimizer = optimizer
        self.optimizer_parameters = optimizer_parameters
        self.mode = mode

        if self.mode == "regression":
            self.loss = self.mse_loss
        elif self.mode == "binary_classification":
            self.loss = self.bce_loss
        elif self.mode == "multy_classification":
            self.loss = self.cross_entropy_loss
        else:
            raise ValueError(
                "Invalid mode value, only 'regression', 'binary_classification' or 'multy_classification' are allowed")

        self.conv_sequential = self.make_conv_blocks()
        self.lin_sequential = self.make_lin_blocks()

        self.valid_losses = []
        self.train_losses = []

    def make_conv_blocks(self):
        conv_layers = []
        for i, val in enumerate(self.hidden_conv[:-1]):
            conv_layers.append(self.conv_block(self.hidden_conv[i], self.hidden_conv[i + 1]))
        return Sequential("x, edge_index",
                          [(c_l, 'x, edge_index -> x') for c_l in conv_layers])

    def conv_block(self, in_f, out_f, *args, **kwargs):
        return Sequential(
            "x, edge_index",
            [(self.conv_layer(in_f, out_f, *args, **kwargs), 'x, edge_index -> x'),
             nn.Dropout(self.conv_dropout),
             self.conv_actf]
        )

    def make_lin_blocks(self):
        lin_layers = []

        for i, val in enumerate(self.hidden_fc[:-1]):
            lin_layers.append(self.lin_block(self.hidden_fc[i], self.hidden_fc[i + 1]))
        lin_layers += [nn.Linear(self.hidden_fc[-1], self.num_targets)]

        return nn.Sequential(*lin_layers)

    def lin_block(self, in_f, out_f, *args, **kwargs):
        layers = [
            nn.Linear(in_f, out_f, *args, **kwargs),
            nn.Dropout(self.fc_dropout),
            self.fc_actf,
        ]
        if self.fc_bn:
            layers.insert(1, nn.BatchNorm1d(out_f))
        return nn.Sequential(*layers)

    def forward(self, x, edge_index, batch=None, num_classes=None):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.conv_sequential(x, edge_index)
        x = self.graph_pooling(x, batch)
        x = self.lin_sequential(x)
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

    def mse_loss(self, true, pred):
        true.resize_(pred.shape)
        return sqrt(F.mse_loss(pred, true.type(torch.float32)))

    def cross_entropy_loss(self, true, pred):
        return F.cross_entropy(pred, true)

    def bce_loss(self, true, pred):
        return F.binary_cross_entropy(torch.sigmoid(pred) + 1e-18, true.reshape(-1, 1) + 1e-18)

    def training_step(self, train_batch, *args, **kwargs):
        logits = self.forward(train_batch.x, train_batch.edge_index, batch=train_batch.batch)
        loss = self.loss(train_batch.y, logits)
        self.log('train_loss', loss, batch_size=self.batch_size)
        return loss

    def validation_step(self, val_batch, *args, **kwargs):
        logits = self.forward(val_batch.x, val_batch.edge_index, batch=val_batch.batch)
        loss = self.loss(val_batch.y, logits)
        self.log('val_loss', loss, batch_size=self.batch_size)
        return loss

    def validation_epoch_end(self, outputs):  # TODO check why val_epoch_end calls one more time than train_epoch_end
        avg_loss = torch.stack([x for x in outputs]).mean()
        self.valid_losses.append(float(avg_loss))

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.train_losses.append(float(avg_loss))

    def get_architecture(self):
        return {"conv_layer": self.conv_layer,
                "hidden_conv": self.hidden_conv,
                "hidden_fc": self.hidden_fc,
                "n_conv": self.n_conv,
                "n_fc": self.n_fc,
                "conv_dropout": self.conv_dropout,
                "fc_dropout": self.fc_dropout,
                "fc_bn": self.fc_bn,
                "conv_actf": self.conv_actf,
                "fc_actf": self.fc_actf,
                "optimizer": self.optimizer,
                }

    def freeze_layers(self, number_of_layers: int) -> None:
        n_c = self.n_conv
        n_l = self.n_fc
        parameters = list(self.named_parameters())
        if number_of_layers <= n_c:
            for i in range(number_of_layers):
                for name, tensor in parameters:
                    if name.startswith(f"conv_sequential.module_{i}"):
                        tensor.requires_grad = False

        elif n_c + n_l >= number_of_layers > n_c:
            for i in range(n_c):
                for name, tensor in parameters:
                    if name.startswith(f"conv_sequential.module_{i}"):
                        tensor.requires_grad = False
            for i in range(number_of_layers - n_c):
                for name, tensor in parameters:
                    if name.startswith(f"lin_sequential.{i}"):
                        tensor.requires_grad = False
        else:
            warnings.warn("Number of layers in model lower than requested freeze number")
            for name, tensor in parameters:
                tensor.requires_grad = False

    def get_model_structure(self):
        return {"node_features": self.node_features,
                "num_targets": self.num_targets,
                "conv_layer": str(self.conv_layer),
                "hidden_conv": self.hidden_conv,
                "hidden_fc": self.hidden_fc,
                "n_conv": self.n_conv,
                "n_fc": self.n_fc}


class MolGraphHeteroNet(LightningModule):
    def __init__(self, node_features, metal_features, num_targets, batch_size,
                 hidden_metal_fc=(64,), metal_fc_dropout=0, metal_fc_bn=False, metal_fc_actf=nn.LeakyReLU(),
                 hidden_pre_fc=(64,), pre_fc_dropout=0, pre_fc_actf=nn.LeakyReLU(),
                 hidden_conv=(64,), conv_dropout=0, conv_actf=nn.LeakyReLU(),
                 hidden_post_fc=(64,), post_fc_dropout=0, post_fc_bn=False, post_fc_actf=nn.LeakyReLU(),
                 conv_layer=MFConv, conv_parameters=None, graph_pooling=global_mean_pool, global_pooling=ConcatPooling,
                 optimizer=torch.optim.Adam, optimizer_parameters=None, mode="regression"):
        super(MolGraphHeteroNet, self).__init__()

        self.node_features = node_features
        self.metal_features = metal_features
        self.num_targets = num_targets
        self.batch_size = batch_size

        self.conv_layer = conv_layer
        self.conv_parameters = conv_parameters or {}
        self.hidden_conv = hidden_conv
        self.conv_dropout = conv_dropout
        self.conv_actf = conv_actf

        self.hidden_fc_metal = hidden_metal_fc
        self.metal_fc_bn = metal_fc_bn
        self.metal_fc_dropout = metal_fc_dropout
        self.metal_fc_actf = metal_fc_actf


        self.hidden_post_fc = hidden_post_fc
        self.post_fc_bn = post_fc_bn
        self.post_fc_dropout = post_fc_dropout
        self.post_fc_actf = post_fc_actf

        self.optimizer = optimizer
        self.optimizer_parameters = optimizer_parameters or {}
        self.mode = mode

        self.pre_fc_sequential = self.make_fc_blocks(hidden_dims=(node_features, *hidden_pre_fc),
                                                     actf=pre_fc_actf,
                                                     batch_norm=False,
                                                     dropout=pre_fc_dropout)
        self.conv_sequential = self.make_conv_blocks(hidden_dims=(hidden_pre_fc[-1], *hidden_conv),
                                                     actf=conv_actf,
                                                     layer=conv_layer,
                                                     layer_parameters=conv_parameters,
                                                     dropout=conv_dropout)
        self.graph_pooling = graph_pooling
        self.metal_fc_sequential = self.make_fc_blocks(hidden_dims=(metal_features, *hidden_metal_fc),
                                                       actf=metal_fc_actf,
                                                       batch_norm=metal_fc_bn,
                                                       dropout=metal_fc_dropout)
        self.global_pooling = global_pooling(input_dims=(hidden_conv[-1], hidden_metal_fc[-1]))
        self.post_fc_sequential = self.make_fc_blocks(hidden_dims=(self.global_pooling.output_dim, *hidden_post_fc),
                                                      actf=post_fc_actf,
                                                      batch_norm=post_fc_bn,
                                                      dropout=post_fc_dropout)

        if self.mode == "regression":
            self.out_sequential = nn.Sequential(nn.Linear(hidden_post_fc[-1], num_targets))
            self.loss = lambda *args, **kwargs: sqrt(F.mse_loss(*args, **kwargs))
        elif self.mode == "binary_classification":
            self.out_sequential = nn.Sequential(nn.Linear(hidden_post_fc[-1], num_targets), nn.Sigmoid())
            self.loss = F.binary_cross_entropy
        elif self.mode == "multy_classification":
            self.out_sequential = nn.Sequential(nn.Linear(hidden_post_fc[-1], num_targets), nn.Softmax())
            self.loss = F.cross_entropy
        else:
            raise ValueError(
                "Invalid mode value, only 'regression', 'binary_classification' or 'multy_classification' are allowed")


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
        x, edge_index, metal_x, batch = graph.x, graph.edge_index, graph.metal_x, graph.batch
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.pre_fc_sequential(x)
        x = self.conv_sequential(x, edge_index)
        x = self.graph_pooling(x, batch=batch)
        metal_x = self.metal_fc_sequential(metal_x)
        general = self.global_pooling(x, metal_x)
        general = self.post_fc_sequential(general)
        return general

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
        logits = self.forward(train_batch.x, train_batch.edge_index, train_batch.metal_x, batch=train_batch.batch)
        loss = self.loss(train_batch.y, logits.reshape(*train_batch.y.shape))
        self.log('train_loss', loss, batch_size=self.batch_size)
        return loss

    def validation_step(self, val_batch, *args, **kwargs):
        logits = self.forward(val_batch.x, val_batch.edge_index, val_batch.metal_x, batch=val_batch.batch)
        loss = self.loss(val_batch.y, logits.reshape(*val_batch.y.shape))
        self.log('val_loss', loss, batch_size=self.batch_size)
        return loss

    def get_model_structure(self):
        def is_jsonable(x):
            try:
                json.dumps(x)
                return True
            except (TypeError, OverflowError):
                return False
        return {key: value if is_jsonable(value) else str(value) for key, value in self.config.items()}