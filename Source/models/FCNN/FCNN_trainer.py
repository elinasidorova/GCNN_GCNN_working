import numpy as np
import torch

from Source.trainer import ModelTrainer


class FCNNTrainer(ModelTrainer):
    def __init__(self, *args, **kwargs):
        super(FCNNTrainer, self).__init__(*args, **kwargs)

    def get_true_pred(self):
        train_true = []
        train_pred = []
        valid_pred = []
        valid_true = []

        test_true = torch.cat([target.unsqueeze(0) for _, target in self.test_data.dataset])
        test_pred = torch.zeros(test_true.shape)
        test_data = self.test_data.dataset

        for fold_ind, model in enumerate(self.models):
            train_data = self.train_valid_data[fold_ind][0].dataset
            valid_data = self.train_valid_data[fold_ind][1].dataset

            train_true += torch.cat([target.unsqueeze(0) for _, target in train_data]).tolist()
            valid_true += torch.cat([target.unsqueeze(0) for _, target in valid_data]).tolist()
            train_pred += torch.cat([model(x.view(1, -1)).view(1, -1).detach() for x, _ in train_data]).tolist()
            valid_pred += torch.cat([model(x.view(1, -1)).view(1, -1).detach() for x, _ in valid_data]).tolist()

            test_pred += torch.cat([model(x.view(1, -1)).view(1, -1).detach() for x, _ in test_data]) / len(self.models)

        return (np.array(train_true), np.array(valid_true), np.array(test_true),
                np.array(train_pred), np.array(valid_pred), test_pred.numpy())
