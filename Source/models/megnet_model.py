import torch
import torch.nn.functional as F
import torch_geometric
from pytorch_lightning import LightningModule
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import Set2Set, MetaLayer
from torch_scatter import scatter_mean, scatter


# Megnet
class Megnet_EdgeModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims=(64, 64), actf=nn.ReLU(),
                 batch_norm=False, batch_track_stats=True, dropout_rate=0.0):
        super(Megnet_EdgeModel, self).__init__()
        self.actf = actf
        self.batch_track_stats = batch_track_stats
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate

        def get_layer(in_dim, out_dim):
            layer = [nn.Linear(in_dim, out_dim),
                     self.actf,
                     nn.Dropout(self.dropout_rate), ]
            if self.batch_norm:
                layer.insert(1, nn.BatchNorm1d(out_dim, track_running_stats=self.batch_track_stats))
            return nn.Sequential(*layer)

        dims = (input_dim, *hidden_dims)
        self.edge_mlp = nn.Sequential(*[get_layer(dims[i], dims[i + 1]) for i in range(len(hidden_dims))])

    def forward(self, src, dest, edge_attr, u, batch):
        comb = torch.cat([src, dest, edge_attr, u[batch]], dim=1)
        out = self.edge_mlp(comb)
        return out


class Megnet_NodeModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims=(64, 64), actf=nn.ReLU(),
                 batch_norm=False, batch_track_stats=True, dropout_rate=0.0):
        super(Megnet_NodeModel, self).__init__()
        self.actf = actf
        self.batch_track_stats = batch_track_stats
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate

        def get_layer(in_dim, out_dim):
            layer = [nn.Linear(in_dim, out_dim),
                     self.actf,
                     nn.Dropout(self.dropout_rate), ]
            if self.batch_norm:
                layer.insert(1, nn.BatchNorm1d(out_dim, track_running_stats=self.batch_track_stats))
            return nn.Sequential(*layer)

        dims = (input_dim, *hidden_dims)
        self.node_mlp = nn.Sequential(*[get_layer(dims[i], dims[i + 1]) for i in range(len(hidden_dims))])

    def forward(self, x, edge_index, edge_attr, u, batch):
        v_e = scatter_mean(edge_attr, edge_index[0, :], dim=0)
        comb = torch.cat([x, v_e, u[batch]], dim=1)
        out = self.node_mlp(comb)
        return out


class Megnet_GlobalModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims=(64, 64), actf=nn.ReLU(),
                 batch_norm=False, batch_track_stats=True, dropout_rate=0.0):
        super(Megnet_GlobalModel, self).__init__()
        self.actf = actf
        self.batch_track_stats = batch_track_stats
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate

        def get_layer(in_dim, out_dim):
            layer = [nn.Linear(in_dim, out_dim),
                     self.actf,
                     nn.Dropout(self.dropout_rate), ]
            if self.batch_norm:
                layer.insert(1, nn.BatchNorm1d(out_dim, track_running_stats=self.batch_track_stats))
            return nn.Sequential(*layer)

        dims = (input_dim, *hidden_dims)
        self.global_mlp = nn.Sequential(*[get_layer(dims[i], dims[i + 1]) for i in range(len(hidden_dims))])

    def forward(self, x, edge_index, edge_attr, u, batch):
        u_e = scatter_mean(edge_attr, edge_index[0, :], dim=0)
        u_e = scatter_mean(u_e, batch, dim=0)
        u_v = scatter_mean(x, batch, dim=0)
        comb = torch.cat([u_e, u_v, u], dim=1)
        out = self.global_mlp(comb)
        return out


class MEGNet(LightningModule):
    def __init__(
            self,
            data,
            batch_size,
            pre_dense_edge_hidden=(),
            pre_dense_node_hidden=(),
            pre_dense_general_hidden=(64,),
            megnet_dense_hidden=(64, 64, 64),
            megnet_edge_conv_hidden=(64,),
            megnet_node_conv_hidden=(64,),
            megnet_general_conv_hidden=(64,),
            post_dense_hidden=(64,),
            pool="global_mean_pool",
            pool_order="early",
            batch_norm=False,
            batch_track_stats=True,
            actf=nn.ReLU(),
            dropout_rate=0.0,
            optimizer=Adam,
            optimizer_parameters=None,
            mode="regression",
    ):
        super(MEGNet, self).__init__()

        self.config = {
            "data": data,
            "batch_size": batch_size,
            "pre_dense_edge_hidden": pre_dense_edge_hidden,
            "pre_dense_node_hidden": pre_dense_node_hidden,
            "pre_dense_general_hidden": pre_dense_general_hidden,
            "megnet_dense_hidden": megnet_dense_hidden,
            "megnet_edge_conv_hidden": megnet_edge_conv_hidden,
            "megnet_node_conv_hidden": megnet_node_conv_hidden,
            "megnet_general_conv_hidden": megnet_general_conv_hidden,
            "post_dense_hidden": post_dense_hidden,
            "pool": pool,
            "pool_order": pool_order,
            "batch_norm": batch_norm,
            "batch_track_stats": batch_track_stats,
            "actf": actf,
            "dropout_rate": dropout_rate,
            "optimizer": optimizer,
            "optimizer_parameters": optimizer_parameters,
            "mode": mode,
        }
        self.mode = mode
        self.optimizer = optimizer
        self.optimizer_parameters = optimizer_parameters or {}
        self.loss = self.mse_loss
        self.batch_size = batch_size
        self.batch_track_stats = batch_track_stats
        self.batch_norm = batch_norm
        self.pool = pool
        if pool == "global_mean_pool":
            self.pool_reduce = "mean"
        elif pool == "global_max_pool":
            self.pool_reduce = "max"
        elif pool == "global_sum_pool":
            self.pool_reduce = "sum"
        self.actf = actf
        self.pool_order = pool_order
        self.dropout_rate = dropout_rate

        ##Determine gc dimension dimension
        assert len(megnet_dense_hidden) > 0, "Need at least 1 GC layer"

        ##Determine output dimension length
        self.num_targets = data[0].y.shape[0]

        ##Set up pre-GNN dense layers (NOTE: in v0.1 this is always set to 1 layer)
        pre_lin_dims = (data[0].edge_attr.shape[-1], *pre_dense_edge_hidden)
        self.pre_lin_edge_layer = nn.Sequential(*[nn.Sequential(
            nn.Linear(pre_lin_dims[i], pre_lin_dims[i + 1]),
            actf
        ) for i in range(len(pre_lin_dims) - 1)])

        pre_lin_dims = (data[0].num_features, *pre_dense_node_hidden)
        self.pre_lin_node_layer = nn.Sequential(*[nn.Sequential(
            nn.Linear(pre_lin_dims[i], pre_lin_dims[i + 1]),
            actf
        ) for i in range(len(pre_lin_dims) - 1)])

        pre_lin_dims = (data[0].u.shape[-1], *pre_dense_general_hidden)
        self.pre_lin_general_layer = nn.Sequential(*[nn.Sequential(
            nn.Linear(pre_lin_dims[i], pre_lin_dims[i + 1]),
            actf
        ) for i in range(len(pre_lin_dims) - 1)])

        ##Set up GNN layers
        self.e_embed_list = torch.nn.ModuleList()
        self.x_embed_list = torch.nn.ModuleList()
        self.u_embed_list = torch.nn.ModuleList()
        self.conv_list = torch.nn.ModuleList()
        self.bn_list = torch.nn.ModuleList()

        e_dim = data[0].num_edge_features if len(pre_dense_edge_hidden) == 0 else pre_dense_edge_hidden[-1]
        x_dim = data[0].num_features if len(pre_dense_node_hidden) == 0 else pre_dense_node_hidden[-1]
        u_dim = data[0].u.shape[1] if len(pre_dense_general_hidden) == 0 else pre_dense_general_hidden[-1]

        e_embed_dims = (e_dim, *megnet_dense_hidden)
        x_embed_dims = (x_dim, *megnet_dense_hidden)
        u_embed_dims = (u_dim, *megnet_dense_hidden)

        for i in range(len(megnet_dense_hidden)):
            self.e_embed_list.append(nn.Sequential(
                nn.Linear(e_embed_dims[i], e_embed_dims[i + 1]),
                nn.ReLU(),
                nn.Linear(e_embed_dims[i + 1], e_embed_dims[i + 1]),
                nn.ReLU()
            ))
            self.x_embed_list.append(nn.Sequential(
                nn.Linear(x_embed_dims[i], x_embed_dims[i + 1]),
                nn.ReLU(),
                nn.Linear(x_embed_dims[i + 1], x_embed_dims[i + 1]),
                nn.ReLU()
            ))
            self.u_embed_list.append(nn.Sequential(
                nn.Linear(u_embed_dims[i], u_embed_dims[i + 1]),
                nn.ReLU(),
                nn.Linear(u_embed_dims[i + 1], u_embed_dims[i + 1]),
                nn.ReLU()
            ))
            self.conv_list.append(MetaLayer(
                Megnet_EdgeModel(
                    input_dim=megnet_dense_hidden[i] * 4,
                    hidden_dims=(*megnet_edge_conv_hidden, megnet_dense_hidden[i]),
                    actf=self.actf,
                    batch_norm=self.batch_norm,
                    batch_track_stats=self.batch_track_stats,
                    dropout_rate=self.dropout_rate),
                Megnet_NodeModel(
                    input_dim=megnet_dense_hidden[i] * 3,
                    hidden_dims=(*megnet_node_conv_hidden, megnet_dense_hidden[i]),
                    actf=self.actf,
                    batch_norm=self.batch_norm,
                    batch_track_stats=self.batch_track_stats,
                    dropout_rate=self.dropout_rate),
                Megnet_GlobalModel(
                    input_dim=megnet_dense_hidden[i] * 3,
                    hidden_dims=(*megnet_general_conv_hidden, megnet_dense_hidden[i]),
                    actf=self.actf,
                    batch_norm=self.batch_norm,
                    batch_track_stats=self.batch_track_stats,
                    dropout_rate=self.dropout_rate)
            ))

        ##Set up post-GNN dense layers (NOTE: in v0.1 there was a minimum of 2 dense layers, and fc_count(now post_fc_count) added to this number. In the current version, the minimum is zero)
        def get_layer(in_dim, out_dim):
            layer = [nn.Linear(in_dim, out_dim),
                     nn.Dropout(self.dropout_rate),
                     self.actf]
            if self.batch_norm:
                layer.insert(1, nn.BatchNorm1d(out_dim, track_running_stats=self.batch_track_stats))
            return nn.Sequential(*layer)

        ##Set2set pooling has doubled dimension
        if self.pool_order == "early" and self.pool == "set2set":
            input_dim = megnet_dense_hidden[-1] * 5
        elif self.pool_order == "early" and self.pool != "set2set":
            input_dim = megnet_dense_hidden[-1] * 3
        elif self.pool_order == "late":
            input_dim = megnet_dense_hidden[-1]
        else:
            raise ValueError("'self.pool_order' should be one of: 'early', 'late'")
        dims = (input_dim, *post_dense_hidden)
        self.post_lin_list = nn.Sequential(*[get_layer(dims[i], dims[i + 1]) for i in range(len(post_dense_hidden))])
        self.lin_out = nn.Linear(post_dense_hidden[-1] if len(post_dense_hidden) != 0 else input_dim, self.num_targets)

        ##Set up set2set pooling (if used)
        if self.pool_order == "early" and self.pool == "set2set":
            self.set2set_x = Set2Set(megnet_dense_hidden[-1], processing_steps=3)
            self.set2set_e = Set2Set(megnet_dense_hidden[-1], processing_steps=3)
        elif self.pool_order == "late" and self.pool == "set2set":
            self.set2set_x = Set2Set(self.num_targets, processing_steps=3, num_layers=1)
            # workaround for doubled dimension by set2set; if late pooling not reccomended to use set2set
            self.lin_out_2 = nn.Linear(self.num_targets * 2, self.num_targets)

        self.valid_losses = []
        self.train_losses = []

    def mse_loss(self, true, pred):
        true.resize_(pred.shape)
        return torch.sqrt(F.mse_loss(pred, true.type(torch.float32)))

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

    def forward(self, data):
        x = data.x
        edge_attr = data.edge_attr
        u = data.u

        ##Pre-GNN dense layers
        x = self.pre_lin_node_layer(x)
        edge_attr = self.pre_lin_edge_layer(edge_attr)
        u = self.pre_lin_general_layer(u)

        ##GNN layers
        for i in range(len(self.conv_list)):
            x = self.x_embed_list[i](x)
            edge_attr = self.e_embed_list[i](edge_attr)
            u = self.u_embed_list[i](u)
            x_out, e_out, u_out = self.conv_list[i](x, data.edge_index, edge_attr, u, data.batch)
            x = torch.add(x_out, x)
            edge_attr = torch.add(e_out, edge_attr)
            u = torch.add(u_out, u)

        ##Post-GNN dense layers
        if self.pool_order == "early":
            if self.pool == "set2set":
                x_pool = self.set2set_x(x, data.batch)
                e = scatter(edge_attr, data.edge_index[0, :], dim=0, reduce="mean")
                e_pool = self.set2set_e(e, data.batch)
                out = torch.cat([x_pool, e_pool, u], dim=1)
            else:
                x_pool = scatter(x, data.batch, dim=0, reduce=self.pool_reduce)
                e_pool = scatter(edge_attr, data.edge_index[0, :], dim=0, reduce=self.pool_reduce)
                e_pool = scatter(e_pool, data.batch, dim=0, reduce=self.pool_reduce)
                out = torch.cat([x_pool, e_pool, u], dim=1)
            out = self.post_lin_list(out)
            out = self.lin_out(out)

        ##currently only uses node features for late pooling
        elif self.pool_order == "late":
            out = x
            out = self.post_lin_list(out)
            out = self.lin_out(out)
            if self.pool == "set2set":
                out = self.set2set_x(out, data.batch)
                out = self.lin_out_2(out)
            else:
                out = getattr(torch_geometric.nn, self.pool)(out, data.batch)

        if out.shape[1] == 1:
            return out.view(-1)
        else:
            return out

    def training_step(self, train_batch, *args, **kwargs):
        logits = self.forward(train_batch)
        loss = self.loss(train_batch.y, logits)
        self.log('train_loss', loss, batch_size=self.batch_size, prog_bar=True)
        return loss

    def validation_step(self, val_batch, *args, **kwargs):
        logits = self.forward(val_batch)
        loss = self.loss(val_batch.y, logits)
        self.log('val_loss', loss, batch_size=self.batch_size)
        return loss

    def validation_epoch_end(self, outputs):  # TODO check why val_epoch_end calls one more time than train_epoch_end
        avg_loss = torch.stack([x for x in outputs]).mean()
        self.valid_losses.append(float(avg_loss))

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.train_losses.append(float(avg_loss))

    def get_model_structure(self):
        return {}
