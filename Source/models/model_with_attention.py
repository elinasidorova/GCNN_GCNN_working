import warnings
from inspect import signature

import torch.nn as nn
import torch.nn.functional as F
import torch.optim.optimizer
from pytorch_lightning import LightningModule
from torch import sqrt
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import global_mean_pool as gap, Sequential
from torch_geometric.nn.conv import MFConv
from torch_geometric.utils import add_self_loops


class BaseAttention(nn.Module):
    def __init__(self):
        super(BaseAttention, self).__init__()

    def forward(self, x, metal_x, batch):
        return x, metal_x


class GlobalAddAttention(BaseAttention):
    def __init__(self, attention_parameters=None,
                 key_sequential=nn.Sequential(), query_sequential=nn.Sequential()):
        super(GlobalAddAttention, self).__init__()

        attention_parameters = attention_parameters or {"num_heads": 1}
        self.attention = torch.nn.MultiheadAttention(**attention_parameters)
        self.key_sequential = key_sequential
        self.query_sequential = query_sequential

    def forward(self, x, metal_x, batch):
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

        x_out = out[x_mask.T, :]
        metal_x_out = out[metal_x_mask.T, :]

        return x_out, metal_x_out


class GlobalConcatAttention(BaseAttention):
    def __init__(self, out_sequential, attention_parameters=None,
                 key_sequential=nn.Sequential(), query_sequential=nn.Sequential()):
        super(GlobalConcatAttention, self).__init__()

        attention_parameters = attention_parameters or {"num_heads": 1}
        self.attention = torch.nn.MultiheadAttention(**attention_parameters)
        self.key_sequential = key_sequential
        self.query_sequential = query_sequential
        self.out_sequential = out_sequential

    def forward(self, x, metal_x, batch):
        _, counts = torch.unique(batch, return_counts=True)
        # mask.shape == (num_graphs, max(nodes_per_graph))
        mask = pad_sequence([torch.ones(s) for s in counts], batch_first=True).to(bool).to(x.device)

        # value.shape = (max(nodes_per_graph), num_graphs, node_features * 2)
        value = pad_sequence([torch.cat((metal_x[batch][batch == i], x[batch == i]), dim=-1) for i in batch.unique()])
        key = self.key_sequential(value)
        query = self.query_sequential(value)

        # out.shape = (max(nodes_per_graph), num_graphs, node_features * 2)
        out, _ = self.attention(query, key, value, key_padding_mask=~mask)

        x_out = out[mask.T, :]
        x_out = self.out_sequential(x_out)

        return x_out, metal_x


class LocalConcatAttention(BaseAttention):
    def __init__(self, out_sequential, attention_parameters=None,
                 key_sequential=nn.Sequential(), query_sequential=nn.Sequential()):
        super(LocalConcatAttention, self).__init__()

        attention_parameters = attention_parameters or {"num_heads": 1}
        self.attention = torch.nn.MultiheadAttention(**attention_parameters)
        self.key_sequential = key_sequential
        self.query_sequential = query_sequential
        self.out_sequential = out_sequential

    def forward(self, x, metal_x, batch):
        # value.shape == (node_features * 2, nodes_in_batch, 1)
        value = torch.cat((x, metal_x[batch]), dim=-1).T.unsqueeze(-1)
        key = self.key_sequential(value)
        query = self.query_sequential(value)

        # out.shape == (node_features * 2, nodes_in_batch, 1)
        out, _ = self.attention(query, key, value)
        # x_out.shape == (nodes_in_batch, node_features)
        x_out = self.out_sequential(out.squeeze().T)

        return x_out, metal_x


class MolGraphHeteroNet(LightningModule):
    def __init__(self, node_features, metal_features, num_targets, batch_size,
                 hidden_metal=(64,), hidden_conv=(64,), hidden_linear=(64,),
                 metal_actf=nn.LeakyReLU(), conv_actf=nn.LeakyReLU(), linear_actf=nn.LeakyReLU(),
                 metal_dropout=0, conv_dropout=0, linear_dropout=0,
                 metal_bn=False, linear_bn=False,
                 metal_ligand_unifunc=None, conv_layer=MFConv, pooling_layer=gap,
                 use_attention=False, attention_name="GlobalAddAttention", attention_parameters=None,
                 attention_key_hidden=(64,), attention_query_hidden=(64,),
                 attention_key_actf=nn.ReLU(), attention_query_actf=nn.ReLU(),
                 attention_key_bn=False, attention_query_bn=False,
                 optimizer=torch.optim.Adam, optimizer_parameters=None, mode="regression"):
        super().__init__()

        assert mode in ["regression",
                        "binary_classification",
                        "multy_classification"], "Invalid mode value, only 'regression', 'binary_classification' or 'multy_classification' are allowed"

        self.config = {
            "node_features": node_features,
            "metal_features": metal_features,
            "num_targets": num_targets,
            "batch_size": batch_size,
            "hidden_metal": hidden_metal,
            "hidden_conv": hidden_conv,
            "hidden_linear": hidden_linear,
            "metal_actf": metal_actf,
            "conv_actf": conv_actf,
            "linear_actf": linear_actf,
            "metal_dropout": metal_dropout,
            "conv_dropout": conv_dropout,
            "linear_dropout": linear_dropout,
            "metal_bn": metal_bn,
            "linear_bn": linear_bn,
            "metal_ligand_unifunc": metal_ligand_unifunc,
            "conv_layer": conv_layer,
            "pooling_layer": pooling_layer,
            "use_attention": use_attention,
            "attention_name": attention_name,
            "attention_parameters": attention_parameters,
            "attention_key_hidden": attention_key_hidden,
            "attention_query_hidden": attention_query_hidden,
            "attention_key_actf": attention_key_actf,
            "attention_query_actf": attention_query_actf,
            "attention_key_bn": attention_key_bn,
            "attention_query_bn": attention_query_bn,
            "optimizer": optimizer,
            "optimizer_parameters": optimizer_parameters,
            "mode": mode,
        }

        optimizer_parameters = optimizer_parameters or {}
        attention_parameters = attention_parameters or {"num_heads": 1}

        self.metal_features = metal_features
        self.node_features = node_features
        self.batch_size = batch_size
        self.num_targets = num_targets

        self.hidden_metal = (self.metal_features, *hidden_metal)
        self.hidden_linear = hidden_linear
        self.hidden_conv = (self.node_features, *hidden_conv)

        self.metal_bn = metal_bn
        self.linear_bn = linear_bn
        self.metal_actf = metal_actf
        self.conv_actf = conv_actf
        self.linear_actf = linear_actf

        self.conv_dropout = conv_dropout
        self.linear_dropout = linear_dropout
        self.metal_dropout = metal_dropout

        self.conv_layer = conv_layer
        self.pooling_layer = pooling_layer
        self.metal_ligand_unifunc = metal_ligand_unifunc or (lambda x1, x2: torch.cat((x1, x2), dim=1))

        self.use_attention = use_attention
        self.attention_name = attention_name
        self.attention_parameters = attention_parameters
        self.attention_key_hidden = attention_key_hidden
        self.attention_query_hidden = attention_query_hidden
        self.attention_key_actf = attention_key_actf
        self.attention_query_actf = attention_query_actf
        self.attention_key_bn = attention_key_bn
        self.attention_query_bn = attention_query_bn

        self.n_conv = len(hidden_conv) - 1
        self.n_linear = len(hidden_linear) - 1
        self.optimizer = optimizer
        self.optimizer_parameters = optimizer_parameters
        self.mode = mode

        self.loss = {"regression": self.mse_loss,
                     "binary_classification": self.bce_loss,
                     "multy_classification": self.cross_entropy_loss}[self.mode]

        self.pre_metal_sequential = self.make_lin_blocks(hidden_dims=self.hidden_metal,
                                                         actf=self.metal_actf,
                                                         batch_norm=self.metal_bn,
                                                         dropout=self.metal_dropout)
        self.conv_blocks = self.make_conv_blocks()
        self.metal_blocks = self.make_metal_blocks()
        self.attention_blocks = self.make_attention_blocks() if self.use_attention else (BaseAttention(),) * len(
            self.conv_blocks)
        self.lin_sequential = self.make_lin_blocks(hidden_dims=self.hidden_linear,
                                                   actf=self.linear_actf,
                                                   batch_norm=self.linear_bn,
                                                   dropout=self.linear_dropout)
        self.out_layer = nn.Linear(self.hidden_linear[-1], self.num_targets)

        self.valid_losses = []
        self.train_losses = []

    @staticmethod
    def make_lin_blocks(hidden_dims, actf, batch_norm=False, dropout=0.0):
        def fc_layer(in_f, out_f):
            layers = [nn.Linear(in_f, out_f), nn.Dropout(dropout), actf]
            if batch_norm:
                layers.insert(1, nn.BatchNorm1d(out_f))
            return nn.Sequential(*layers)

        lin_layers = [fc_layer(hidden_dims[i], hidden_dims[i + 1]) for i, val in enumerate(hidden_dims[:-1])]
        return nn.Sequential(*lin_layers)

    def make_conv_blocks(self):
        def conv_block(in_f, out_f):
            layers = [(self.conv_layer(in_f, out_f), 'x, edge_index -> x'),
                      nn.Dropout(self.conv_dropout),
                      self.conv_actf]
            return Sequential("x, edge_index", layers)

        conv_layers = [conv_block(self.hidden_conv[i], self.hidden_conv[i + 1])
                       for i in range(len(self.hidden_conv) - 1)]
        return nn.ModuleList(conv_layers)

    def make_attention_blocks(self):
        if self.attention_name == "GlobalAddAttention":
            layers = [GlobalAddAttention(attention_parameters={**self.attention_parameters, "embed_dim": dim},
                                         key_sequential=self.make_lin_blocks(
                                             hidden_dims=(dim, *self.attention_key_hidden,
                                                          dim) if self.attention_key_hidden != () else (),
                                             actf=self.attention_key_actf,
                                             batch_norm=self.attention_key_bn),
                                         query_sequential=self.make_lin_blocks(
                                             hidden_dims=(dim, *self.attention_query_hidden,
                                                          dim) if self.attention_query_hidden != () else (),
                                             actf=self.attention_query_actf,
                                             batch_norm=self.attention_query_bn))
                      for dim in self.hidden_conv[1:]]
        elif self.attention_name == "GlobalConcatAttention":
            layers = [GlobalConcatAttention(out_sequential=self.make_lin_blocks(
                hidden_dims=(dim * 2, dim),
                actf=nn.LeakyReLU(),
                batch_norm=True),
                attention_parameters={**self.attention_parameters, "embed_dim": dim * 2},
                key_sequential=self.make_lin_blocks(
                    hidden_dims=(dim * 2, *self.attention_key_hidden,
                                 dim * 2) if self.attention_key_hidden != () else (),
                    actf=self.attention_key_actf,
                    batch_norm=self.attention_key_bn),
                query_sequential=self.make_lin_blocks(
                    hidden_dims=(dim * 2, *self.attention_query_hidden,
                                 dim * 2) if self.attention_query_hidden != () else (),
                    actf=self.attention_query_actf,
                    batch_norm=self.attention_query_bn))
                for dim in self.hidden_conv[1:]]
        elif self.attention_name == "LocalConcatAttention":
            layers = [LocalConcatAttention(out_sequential=self.make_lin_blocks(
                hidden_dims=(dim * 2, dim),
                actf=nn.LeakyReLU(),
                batch_norm=True),
                attention_parameters={**self.attention_parameters, "embed_dim": 1},
                key_sequential=self.make_lin_blocks(
                    hidden_dims=(1, *self.attention_key_hidden,
                                 1) if self.attention_key_hidden != () else (),
                    actf=self.attention_key_actf,
                    batch_norm=self.attention_key_bn),
                query_sequential=self.make_lin_blocks(
                    hidden_dims=(1, *self.attention_query_hidden,
                                 1) if self.attention_query_hidden != () else (),
                    actf=self.attention_query_actf,
                    batch_norm=self.attention_query_bn))
                for dim in self.hidden_conv[1:]]
        else:
            raise ValueError(
                "'attention_name' should be one of: 'GlobalAddAttention', 'GlobalConcatAttention', 'LocalConcatAttention'")
        return nn.ModuleList(layers)

    def make_metal_blocks(self):
        def fc_layer(in_f, out_f):
            layers = [nn.Linear(in_f, out_f), nn.Dropout(self.metal_dropout), self.linear_actf]
            if self.metal_bn:
                layers.insert(1, nn.BatchNorm1d(out_f))
            return nn.Sequential(*layers)

        dims = (self.hidden_metal[-1], *self.hidden_conv[1:])
        lin_layers = [fc_layer(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        return nn.ModuleList(lin_layers)

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

    @staticmethod
    def mse_loss(true, pred):
        true.resize_(pred.shape)
        return sqrt(F.mse_loss(pred, true.type(torch.float32)))

    @staticmethod
    def cross_entropy_loss(true, pred):
        return F.cross_entropy(pred, true)

    @staticmethod
    def bce_loss(true, pred):
        return F.binary_cross_entropy(torch.sigmoid(pred) + 1e-18, true.reshape(-1, 1) + 1e-18)

    def forward(self, x, edge_index, metal_x=None, batch=None):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        metal_x = self.pre_metal_sequential(metal_x)
        for conv_layer, metal_layer, attention_layer in zip(self.conv_blocks, self.metal_blocks, self.attention_blocks):
            x = conv_layer(x, edge_index)
            metal_x = metal_layer(metal_x)
            if self.use_attention:
                x, metal_x = attention_layer(x, metal_x, batch)

        if hasattr(self.pooling_layer, "forward") and "edge_index" in signature(self.pooling_layer.forward).parameters:
            x = self.pooling_layer(x, edge_index, batch=batch)[0]
        elif "edge_index" in signature(self.pooling_layer).parameters:
            x = self.pooling_layer(x, edge_index, batch=batch)
        else:
            x = self.pooling_layer(x, batch=batch)
        x = self.metal_ligand_unifunc(metal_x, x)
        x = self.lin_sequential(x)
        x = self.out_layer(x)
        return x

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

    def validation_epoch_end(self, outputs):  # TODO check why val_epoch_end calls one more time than train_epoch_end
        avg_loss = torch.stack([x for x in outputs]).mean()
        self.valid_losses.append(float(avg_loss))

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.train_losses.append(float(avg_loss))

    def get_architecture(self):
        return {"conv_layer": self.conv_layer,
                "hidden_conv": self.hidden_conv,
                "hidden_linear": self.hidden_linear,
                "hidden_metal": self.hidden_metal,
                "n_conv": self.n_conv,
                "n_linear": self.n_linear,
                "conv_dropout": self.conv_dropout,
                "linear_dropout": self.linear_dropout,
                "metal_dropout": self.metal_dropout,
                "linear_bn": self.linear_bn,
                "conv_actf": self.conv_actf,
                "linear_actf": self.linear_actf,
                "optimizer": self.optimizer,
                }

    def get_model_structure(self):
        return {
            "node_features": self.node_features,
            "metal_features": self.hidden_metal[0],
            "num_targets": self.num_targets,
            "batch_size": self.batch_size,
            "conv_layer": self.conv_layer.__name__,
            "metal_ligand_unifunc": self.metal_ligand_unifunc.__name__,
            "hidden_metal": self.hidden_metal,
            "hidden_conv": self.hidden_conv,
            "hidden_linear": self.hidden_linear,
            "metal_dropout": self.metal_dropout,
            "conv_dropout": self.conv_dropout,
            "linear_dropout": self.linear_dropout,
            "pooling_layer": self.pooling_layer.__name__ if hasattr(
                self.pooling_layer,
                "__name__",
            ) else type(self.pooling_layer).__name__,
            "conv_actf": str(self.conv_actf),
            "linear_actf": str(self.linear_actf),
            "linear_bn": self.linear_bn,
            "optimizer": self.optimizer.__name__,
        }
