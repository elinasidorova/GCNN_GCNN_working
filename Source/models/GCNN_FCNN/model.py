import json
import warnings
from inspect import signature

import torch.nn as nn
import torch.nn.functional as F
import torch.optim.optimizer
from pytorch_lightning import LightningModule
from torch import sqrt
from torch.optim.lr_scheduler import ReduceLROnPlateau

from Source.models.FCNN.model import FCNN
from Source.models.GCNN.model import GCNN
from Source.models.global_poolings import ConcatPooling


class GCNN_FCNN(LightningModule):
    def __init__(self, metal_features, node_features, num_targets,
                 metal_fc_params=None, gcnn_params=None, post_fc_params=None, global_pooling=ConcatPooling,
                 use_out_sequential=True,
                 optimizer=torch.optim.Adam, optimizer_parameters=None, mode="regression"):
        super(GCNN_FCNN, self).__init__()
        param_values = locals()
        self.config = {name: param_values[name] for name in signature(self.__init__).parameters.keys()}

        self.num_targets = num_targets
        self.use_out_sequential = use_out_sequential

        self.optimizer = optimizer
        self.optimizer_parameters = optimizer_parameters or {}
        self.mode = mode

        # preparing params for model blocks
        for p in [metal_fc_params, gcnn_params, post_fc_params]:
            if "num_targets" in p:
                warnings.warn(
                    "Not recommended to set 'num_targets' in model blocks as far as it doesn't affect anything",
                    DeprecationWarning)
            if "use_out_sequential" in p:
                warnings.warn("'use_out_sequential' parameter forcibly set to False", DeprecationWarning)
            p["use_out_sequential"] = False
            p["num_targets"] = 1

        metal_fc_params["input_features"] = metal_features
        gcnn_params["node_features"] = node_features
        self.graph_sequential = GCNN(**gcnn_params)
        self.metal_fc_sequential = FCNN(**metal_fc_params)

        self.global_pooling = global_pooling(input_dims=(self.graph_sequential.output_dim,
                                                         self.metal_fc_sequential.output_dim))

        post_fc_params["input_features"] = self.global_pooling.output_dim

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
        x = self.graph_sequential(graph)
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
