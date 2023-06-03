import json
from inspect import signature

import torch.nn as nn
import torch.nn.functional as F
import torch.optim.optimizer
from pytorch_lightning import LightningModule
from torch import sqrt
from torch.optim.lr_scheduler import ReduceLROnPlateau


class FCNN(LightningModule):
    def __init__(self, input_features, num_targets,
                 hidden=(64,), dropout=0, use_bn=False, actf=nn.LeakyReLU(),
                 use_out_sequential=True,
                 optimizer=torch.optim.Adam, optimizer_parameters=None, mode="regression"):
        super(FCNN, self).__init__()
        param_values = locals()
        self.config = {name: param_values[name] for name in signature(self.__init__).parameters.keys()}

        self.input_features = input_features
        self.num_targets = num_targets

        self.hidden = hidden
        self.use_bn = use_bn
        self.dropout = dropout
        self.actf = actf

        self.use_out_sequential = use_out_sequential

        self.optimizer = optimizer
        self.optimizer_parameters = optimizer_parameters or {}
        self.mode = mode

        self.fc_sequential = self.make_fc_blocks(hidden_dims=(input_features, *hidden),
                                                 actf=actf,
                                                 batch_norm=False,
                                                 dropout=dropout)

        if self.use_out_sequential:
            self.output_dim = num_targets
            if self.mode == "regression":
                self.out_sequential = nn.Sequential(nn.Linear((input_features, *hidden)[-1], num_targets))
                self.loss = lambda *args, **kwargs: sqrt(F.mse_loss(*args, **kwargs))
            elif self.mode == "binary_classification":
                self.out_sequential = nn.Sequential(nn.Linear((input_features, *hidden)[-1], num_targets), nn.Sigmoid())
                self.loss = F.binary_cross_entropy
            elif self.mode == "multiclass_classification":
                self.out_sequential = nn.Sequential(nn.Linear((input_features, *hidden)[-1], num_targets), nn.Softmax())
                self.loss = F.cross_entropy
            else:
                raise ValueError(
                    "Invalid mode value, only 'regression', 'binary_classification' or 'multiclass_classification' are allowed")
        else:
            self.out_sequential = nn.Sequential()
            self.output_dim = (input_features, *hidden)[-1]

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

    def forward(self, x):
        x = self.fc_sequential(x)
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
        x, y = train_batch
        loss = self.loss(self.forward(x), y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, val_batch, *args, **kwargs):
        x, y = val_batch
        loss = self.loss(self.forward(x), y)
        self.log('val_loss', loss)
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
