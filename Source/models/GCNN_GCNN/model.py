import json
import warnings
from inspect import signature

import torch.nn as nn
import torch.nn.functional as F
import torch.optim.optimizer
from pytorch_lightning import LightningModule
from torch import sqrt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data

from Source.models.FCNN.model import FCNN
from Source.models.GCNN.model import GCNN
from Source.models.global_poolings import ConcatPooling


class GCNNGCNN(LightningModule):
    def __init__(self, metal_node_features, mol_node_features, num_targets,
                 mol_gcnn_params=None, metal_gcnn_params=None, global_pooling=ConcatPooling, post_fc_params=None,
                 use_out_sequential=True,
                 optimizer=torch.optim.Adam, optimizer_parameters=None, mode="regression"):
        super(GCNNGCNN, self).__init__()
        param_values = locals()
        self.config = {name: param_values[name] for name in signature(self.__init__).parameters.keys()}

        self.num_targets = num_targets
        self.use_out_sequential = use_out_sequential

        self.optimizer = optimizer
        self.optimizer_parameters = optimizer_parameters or {}
        self.mode = mode

        # preparing params for model blocks
        for p in [mol_gcnn_params, metal_gcnn_params, post_fc_params]:
            if "num_targets" in p:
                warnings.warn(
                    "Not recommended to set 'num_targets' in model blocks as far as it doesn't affect anything",
                    DeprecationWarning)
            if "use_out_sequential" in p:
                warnings.warn("'use_out_sequential' parameter forcibly set to False", DeprecationWarning)
            p["use_out_sequential"] = False
            p["num_targets"] = 1

        mol_gcnn_params["input_features"] = mol_node_features
        metal_gcnn_params["node_features"] = metal_node_features
        self.mol_gcnn_sequential = GCNN(**mol_gcnn_params)
        self.metal_gcnn_sequential = GCNN(**metal_gcnn_params)

        self.global_pooling = global_pooling(input_dims=(self.mol_gcnn_sequential.output_dim,
                                                         self.metal_gcnn_sequential.output_dim))

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
            elif self.mode == "multiclass_classification":
                self.out_sequential = nn.Sequential(
                    nn.Linear((self.global_pooling.output_dim, self.post_fc_sequential.output_dim)[-1], num_targets),
                    nn.Softmax())
                self.loss = F.cross_entropy
            else:
                raise ValueError(
                    "Invalid mode value, only 'regression', 'binary_classification' or 'multiclass_classification' are allowed")
        else:
            self.output_dim = self.post_fc_sequential.output_dim

        self.valid_losses = []
        self.train_losses = []

    def forward(self, graph):
        mol_graph = Data(x=graph.x_mol, edge_index=graph.edge_index_mol,
                         edge_attr=graph.edge_attr_mol, u=graph.u_mol,
                         batch=graph.batch)
        metal_graph = Data(x=graph.x_metal, edge_index=graph.edge_index_metal,
                           edge_attr=graph.edge_attr_metal, u=graph.u_metal,
                           batch=graph.batch)
        mol_x = self.mol_gcnn_sequential(mol_graph)
        metal_x = self.metal_gcnn_sequential(metal_graph)
        general = self.global_pooling(mol_x, metal_x)
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
