import json
import warnings
from inspect import signature

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.optimizer
from pytorch_lightning import LightningModule
from torch import sqrt
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import Set2Set, MetaLayer
from torch_scatter import scatter_mean, scatter

from Source.models.FCNN.model import FCNN


# Megnet
class Megnet_EdgeModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super(Megnet_EdgeModel, self).__init__()
        self.fcnn = FCNN(**kwargs)

    def forward(self, src, dest, edge_attr, u, batch):
        comb = torch.cat([src, dest, edge_attr, u[batch]], dim=1)
        out = self.fcnn(comb)
        return out


class Megnet_NodeModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super(Megnet_NodeModel, self).__init__()
        self.fcnn = FCNN(**kwargs)

    def forward(self, x, edge_index, edge_attr, u, batch):
        v_e = scatter_mean(edge_attr, edge_index[0, :], dim=0)
        comb = torch.cat([x, v_e, u[batch]], dim=1)
        out = self.fcnn(comb)
        return out


class Megnet_GlobalModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super(Megnet_GlobalModel, self).__init__()
        self.fcnn = FCNN(**kwargs)

    def forward(self, x, edge_index, edge_attr, u, batch):
        u_e = scatter_mean(edge_attr, edge_index[0, :], dim=0)
        u_e = scatter_mean(u_e, batch, dim=0)
        u_v = scatter_mean(x, batch, dim=0)
        comb = torch.cat([u_e, u_v, u], dim=1)
        out = self.fcnn(comb)
        return out


class MEGNet(LightningModule):
    def __init__(
            self,
            edge_features, node_features, global_features, num_targets,
            n_megnet_blocks=1,
            pre_fc_edge_params=None, pre_fc_node_params=None, pre_fc_general_params=None,
            megnet_fc_edge_params=None, megnet_fc_node_params=None, megnet_fc_general_params=None,
            megnet_conv_edge_params=None, megnet_conv_node_params=None, megnet_conv_general_params=None,
            post_fc_params=None, pool="mean",
            use_out_sequential=True,
            optimizer=Adam, optimizer_parameters=None, mode="regression",
    ):
        super(MEGNet, self).__init__()
        param_values = locals()
        self.config = {name: param_values[name] for name in signature(self.__init__).parameters.keys()}

        pre_fc_edge_params = pre_fc_edge_params or {}
        pre_fc_node_params = pre_fc_node_params or {}
        pre_fc_general_params = pre_fc_general_params or {}
        megnet_fc_edge_params = megnet_fc_edge_params or {}
        megnet_fc_node_params = megnet_fc_node_params or {}
        megnet_fc_general_params = megnet_fc_general_params or {}
        megnet_conv_edge_params = megnet_conv_edge_params or {}
        megnet_conv_node_params = megnet_conv_node_params or {}
        megnet_conv_general_params = megnet_conv_general_params or {}
        post_fc_params = post_fc_params or {}

        for p in [pre_fc_edge_params, pre_fc_node_params, pre_fc_general_params,
                  megnet_fc_edge_params, megnet_fc_node_params, megnet_fc_general_params,
                  megnet_conv_edge_params, megnet_conv_node_params, megnet_conv_general_params,
                  post_fc_params]:
            if "num_targets" in p:
                warnings.warn(
                    "Not recommended to set 'num_targets' in model blocks as far as it doesn't affect anything",
                    DeprecationWarning)
            if "use_out_sequential" in p:
                warnings.warn("'use_out_sequential' parameter forcibly set to False", DeprecationWarning)
            p["use_out_sequential"] = False
            p["num_targets"] = 1

        post_fc_params["input_features"] = None

        self.pool = pool
        self.num_targets = num_targets

        self.mode = mode
        self.optimizer = optimizer
        self.optimizer_parameters = optimizer_parameters or {}

        self.use_out_sequential = use_out_sequential

        pre_fc_edge_params["input_features"] = edge_features
        pre_fc_node_params["input_features"] = node_features
        pre_fc_general_params["input_features"] = global_features

        self.pre_fc_edge_sequential = FCNN(**pre_fc_edge_params)
        self.pre_fc_node_sequential = FCNN(**pre_fc_node_params)
        self.pre_fc_general_sequential = FCNN(**pre_fc_general_params)

        megnet_fc_edge_params["input_features"] = self.pre_fc_edge_sequential.output_dim
        megnet_fc_node_params["input_features"] = self.pre_fc_node_sequential.output_dim
        megnet_fc_general_params["input_features"] = self.pre_fc_general_sequential.output_dim

        for fc, conv in zip([megnet_fc_edge_params, megnet_fc_node_params, megnet_fc_general_params],
                            [megnet_conv_edge_params, megnet_conv_node_params, megnet_conv_general_params]):
            if fc["input_features"] != conv["hidden"][-1]:
                warnings.warn(
                    f"Skip-connection requires matching of FC input_features and conv output_dim, but {fc['input_features']} != {conv['hidden'][-1]}. Adding {fc['input_features']} as new output_dim.")
                conv["hidden"] = (*conv["hidden"], fc["input_features"])
            if fc["input_features"] != fc["hidden"][-1]:
                warnings.warn(
                    f"MEGNet requires matching of FC input_features and FC output_dim, but {fc['input_features']} != {fc['hidden'][-1]}. Adding {fc['input_features']} as new output_dim.")
                fc["hidden"] = (*fc["hidden"], fc["input_features"])

        ##Set up GNN layers
        self.megnet_fc_edge_list = torch.nn.ModuleList([FCNN(**megnet_fc_edge_params) for _ in range(n_megnet_blocks)])
        self.megnet_fc_node_list = torch.nn.ModuleList([FCNN(**megnet_fc_node_params) for _ in range(n_megnet_blocks)])
        self.megnet_fc_general_list = torch.nn.ModuleList([
            FCNN(**megnet_fc_general_params) for _ in range(n_megnet_blocks)
        ])

        e = self.megnet_fc_edge_list[0].output_dim
        n = self.megnet_fc_node_list[0].output_dim
        g = self.megnet_fc_general_list[0].output_dim

        megnet_conv_edge_params["input_features"] = 2 * n + e + g
        megnet_conv_node_params["input_features"] = n + e + g
        megnet_conv_general_params["input_features"] = n + e + g

        self.conv_list = torch.nn.ModuleList([
            MetaLayer(
                Megnet_EdgeModel(**megnet_conv_edge_params),
                Megnet_NodeModel(**megnet_conv_node_params),
                Megnet_GlobalModel(**megnet_conv_general_params)) for _ in range(n_megnet_blocks)
        ])

        # Setup set2set pooling
        if self.pool == "set2set":
            self.set2set_x = Set2Set(n, processing_steps=3)
            self.set2set_e = Set2Set(e, processing_steps=3)

        # Set up post-GNN fc layers
        post_fc_params["input_features"] = (2 * e + 2 * n + g) if self.pool == "set2set" else (e + n + g)
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
            elif self.mode == "multy_classification":
                self.out_sequential = nn.Sequential(
                    nn.Linear(self.post_fc_sequential.output_dim, num_targets), nn.Softmax())
                self.loss = F.cross_entropy
            else:
                raise ValueError(
                    "Invalid mode value, only 'regression', 'binary_classification' or 'multy_classification' are allowed")
        else:
            self.out_sequential = nn.Sequential()
            self.output_dim = self.post_fc_sequential.output_dim

        self.valid_losses = []
        self.train_losses = []

    def forward(self, graph):
        # Pre-MEGNet fc layers
        edge_attr = self.pre_fc_edge_sequential(graph.edge_attr)
        x = self.pre_fc_node_sequential(graph.x)
        u = self.pre_fc_general_sequential(graph.u)

        # MEGNet layers
        for edge_fc, node_fc, general_fc, conv in zip(self.megnet_fc_edge_list, self.megnet_fc_node_list,
                                                      self.megnet_fc_general_list, self.conv_list):
            processed_edge_attr = edge_fc(edge_attr)
            processed_x = node_fc(x)
            processed_u = general_fc(u)
            x_out, e_out, u_out = conv(processed_x, graph.edge_index, processed_edge_attr, processed_u, graph.batch)
            edge_attr = torch.add(e_out, edge_attr)
            x = torch.add(x_out, x)
            u = torch.add(u_out, u)

        # set2set pooling
        if self.pool == "set2set":
            x_pool = self.set2set_x(x, graph.batch)
            e = scatter(edge_attr, graph.edge_index[0, :], dim=0, reduce="mean")
            e_pool = self.set2set_e(e, graph.batch)
        else:
            x_pool = scatter(x, graph.batch, dim=0, reduce=self.pool)
            e_pool = scatter(edge_attr, graph.edge_index[0, :], dim=0, reduce=self.pool)
            e_pool = scatter(e_pool, graph.batch, dim=0, reduce=self.pool)
        out = torch.cat([x_pool, e_pool, u], dim=1)

        # Post-MEGNet fc layers
        out = self.post_fc_sequential(out)
        out = self.out_sequential(out)

        return out

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
