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

from Source.models.FCNN.FCNN import FCNN
from Source.models.GCNN.GCNN import GCNN
from Source.models.global_poolings import ConcatPooling


class GCNNBimodal(LightningModule):
    def __init__(self, num_targets, batch_size,
                 metal_fc_params=None, gcnn_params=None, post_fc_params=None, global_pooling=ConcatPooling,
                 use_out_sequential = True,
                 optimizer=torch.optim.Adam, optimizer_parameters=None, mode="regression"):
        super(GCNNBimodal, self).__init__()
        param_values = locals()
        self.config = {name: param_values[name] for name in signature(self.__init__).parameters.keys()}

        self.num_targets = num_targets
        self.batch_size = batch_size
        self.use_out_sequential = use_out_sequential

        self.optimizer = optimizer
        self.optimizer_parameters = optimizer_parameters or {}
        self.mode = mode

        gcnn_params["use_out_sequential"] = False
        metal_fc_params["use_out_sequential"] = False
        post_fc_params["use_out_sequential"] = False

        self.graph_sequential = GCNN(**gcnn_params)
        self.metal_fc_sequential = FCNN(**metal_fc_params)

        self.global_pooling = global_pooling(input_dims=(self.graph_sequential.output_dim,
                                                         self.metal_fc_sequential.output_dim))

        self.post_fc_sequential = FCNN(**post_fc_params)

        if self.use_out_sequential:
            self.output_dim = num_targets
            if self.mode == "regression":
                self.out_sequential = nn.Sequential(
                    nn.Linear((self.global_pooling.output_dim, self.post_fc_sequential.output_dim)[-1], num_targets))
                self.loss = lambda *args, **kwargs: sqrt(F.mse_loss(*args, **kwargs))
            elif self.mode == "binary_classification":
                self.out_sequential = nn.Sequential(
                    nn.Linear((self.global_pooling.output_dim, self.post_fc_sequential.output_dim)[-1], num_targets),
                    nn.Sigmoid())
                self.loss = F.binary_cross_entropy
            elif self.mode == "multy_classification":
                self.out_sequential = nn.Sequential(
                    nn.Linear((self.global_pooling.output_dim, self.post_fc_sequential.output_dim)[-1], num_targets),
                    nn.Softmax())
                self.loss = F.cross_entropy
            else:
                raise ValueError(
                    "Invalid mode value, only 'regression', 'binary_classification' or 'multy_classification' are allowed")
        else:
            self.output_dim = self.post_fc_sequential.output_dim

        self.valid_losses = []
        self.train_losses = []

    def forward(self, graph):
        x = self.gcnn_sequential(graph)
        metal_x = self.metal_fc_sequential(graph.metal_x)
        general = self.global_pooling(x, metal_x)
        general = self.post_fc_sequential(general)
        if self.use_out_sequential:
            general = self.out_sequential(general)
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
        logits = self.forward(train_batch)
        loss = self.loss(train_batch.y, logits.reshape(*train_batch.y.shape))
        self.log('train_loss', loss, batch_size=self.batch_size)
        return loss

    def validation_step(self, val_batch, *args, **kwargs):
        logits = self.forward(val_batch)
        loss = self.loss(val_batch.y, logits.reshape(*val_batch.y.shape))
        self.log('val_loss', loss, batch_size=self.batch_size)
        return loss

    def get_model_structure(self):
        def make_jsonable(x):
            try:
                json.dumps(x)
                return x
            except (TypeError, OverflowError):
                if type(x) == dict:
                    return {key: make_jsonable(value) for key, value in x}
                return str(x)

        return make_jsonable(self.config)
