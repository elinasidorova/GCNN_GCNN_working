import torch.nn as nn
import torch.optim.optimizer
from torch_geometric.nn.conv import MFConv, GCNConv, GraphConv
from torch_geometric.nn import global_max_pool as gmp, global_mean_pool as gap, BatchNorm
from torch_geometric.utils import add_self_loops
from torch_geometric.nn.pool import TopKPooling
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import sqrt


class MolNet(LightningModule):
    def __init__(self, dataset, hidden_linear=128,  n_linear=4):
        sample = dataset[0]
        super().__init__()
        self.linear = nn.Linear(in_features=len(sample), out_features=hidden_linear, dtype=torch.float32)
        self.linear1 = nn.Linear(in_features=hidden_linear, out_features=hidden_linear, dtype=torch.float32)
        self.linear2 = nn.Linear(in_features=hidden_linear, out_features=hidden_linear, dtype=torch.float32)
        self.fc_out = nn.Linear(in_features=hidden_linear, out_features=len(sample.y), dtype=torch.float32)
        self.dropout = nn.Dropout(p=0.2)
        self.batch_norm = BatchNorm(hidden_linear)

    def forward(self, data):
        x, batch = data.x, data.batch
        x = F.leaky_relu(self.linear(x))
        # x = self.batch_norm(x)
        # x = self.dropout(x)
        x = F.leaky_relu(self.linear1(x))
        # x = self.batch_norm(x)
        # x = self.dropout(x)
        x = F.leaky_relu(self.linear2(x))

        x = self.fc_out(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def mse_loss(self, true, pred):
        true.resize_(pred.shape)

        return sqrt(F.mse_loss(pred, true.type(torch.float32)))

    def cross_entropy_loss(self, true, pred):
        return F.nll_loss(pred, true)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch, train_batch.y
        logits = self.forward(x)
        loss = self.mse_loss(y, logits)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch, val_batch.y
        logits = self.forward(x)
        loss = self.mse_loss(y, logits)
        self.log('val_loss', loss)
        return loss

