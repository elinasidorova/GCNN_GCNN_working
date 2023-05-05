import numpy as np
import torch
from torch_geometric.data import Batch

from Source.trainer import ModelTrainer


class GCNNTrainer(ModelTrainer):
    def __init__(self, *args, **kwargs):
        super(GCNNTrainer, self).__init__(*args, **kwargs)

    def get_true_pred(self):
        train_true = []
        train_pred = []
        valid_pred = []
        valid_true = []

        test_batch = Batch.from_data_list(self.test_data.dataset)
        test_true = test_batch.y.tolist()
        test_pred = torch.zeros(test_batch.y.shape)

        for fold_ind, model in enumerate(self.models):
            train_batch = Batch.from_data_list(self.train_valid_data[fold_ind][0].dataset)
            valid_batch = Batch.from_data_list(self.train_valid_data[fold_ind][1].dataset)

            # model.forward() is of shape (num_samples, num_targets)
            valid_pred += model.forward(valid_batch.x, valid_batch.edge_index,
                                        batch=valid_batch.batch).detach().tolist()
            valid_true += valid_batch.y.tolist()
            train_pred += model.forward(train_batch.x, train_batch.edge_index,
                                        batch=train_batch.batch).detach().tolist()
            train_true += train_batch.y.tolist()

            test_pred += model.forward(test_batch.x,
                                       test_batch.edge_index,
                                       batch=test_batch.batch).detach() / len(self.models)

        return (np.array(train_true), np.array(valid_true), np.array(test_true),
                np.array(train_pred), np.array(valid_pred), test_pred.numpy())
