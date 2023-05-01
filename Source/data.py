import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from sklearn.model_selection import KFold, train_test_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np


def train_test_valid_split(dataset: [], n_splits=5, test_ratio=0.2, batch_size=64, subsample_size=False,
                           return_test=True):
    """
    Makes KFold cross-validation

    Parameters
    ----------
    dataset : Dataset
    n_splits : int, optional
        Number of folds in cross-validatoin
    test_ratio : float from 0.0 to 1.0, optional
        Percentage of test data in dataset
    batch_size : int, optional
    subsample_size : int, optional
        If >0, first subsample_size entries are taken from dataset. Default is 0.
    return_test : bool, optional
        If True, returns test DataLoader (according to test_ratio) in addition to folds. Default is True.


    Returns
    -------
    folds : list
        List of cross-validation folds in format (train_loader, valid_loader)
    test_loader (if return_test) : DataLoader
        Test DataLoader, which does not participate in cross-validation
    """
    if subsample_size:
        dataset = dataset[:subsample_size]
    dataset_size = len(dataset)
    ids = range(dataset_size)
    if return_test:
        train_ids, test_ids = train_test_split(ids, test_size=test_ratio, random_state=14)
        test_loader = DataLoader([val for i, val in enumerate(dataset) if i in test_ids], batch_size=batch_size)
    else:
        train_ids = ids

    folds = []
    kf_split = KFold(n_splits=n_splits)
    for train_index, valid_index in kf_split.split(train_ids):
        train_loader = DataLoader([val for i, val in enumerate(dataset) if i in train_index], batch_size=batch_size)
        valid_loader = DataLoader([val for i, val in enumerate(dataset) if i in valid_index], batch_size=batch_size)
        folds.append((train_loader, valid_loader))
    if return_test:
        return folds, test_loader
    else:
        return folds


def data_to_batch(dataset: [], batch_size=1):
    """
    Packs data to a batched DataLoader

    Parameters
    ----------
    dataset : Dataset
    batch_size : int, optional
        Default is 1.

    Returns
    -------
    dataloader : DataLoader
        Batched DataLoader made according to torch.utils.data.sampler.SequentialSampler
    """
    dataset_size = len(dataset)
    ids = range(dataset_size)
    sampler = SequentialSampler(ids)

    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def get_num_node_features(dataset):
    if isinstance(dataset, list) or isinstance(dataset, tuple):
        dataset = dataset[0]
    if isinstance(dataset, DataLoader):
        dataset = next(iter(dataset))
    if isinstance(dataset, Data):
        return dataset.x.shape[1]
    else:
        raise TypeError(f"Invalid input dataset type: {type(dataset)}")


def get_num_metal_features(dataset):
    if isinstance(dataset, list) or isinstance(dataset, tuple):
        dataset = dataset[0]
    if isinstance(dataset, DataLoader):
        dataset = next(iter(dataset))
    if isinstance(dataset, Data):
        return dataset.metal_x.shape[1]
    else:
        raise TypeError(f"Invalid input dataset type: {type(dataset)}")


def get_num_targets(dataset):
    if isinstance(dataset, list) or isinstance(dataset, tuple):
        dataset = dataset[0]
    if isinstance(dataset, DataLoader):
        dataset = next(iter(dataset))
    if isinstance(dataset, Data):
        return 1 if len(dataset.y.shape) == 1 else dataset.y.shape[1]
    else:
        raise TypeError(f"Invalid input dataset type: {type(dataset)}")


def get_batch_size(dataset):
    if isinstance(dataset, list) or isinstance(dataset, tuple):
        dataset = dataset[0]
    if isinstance(dataset, DataLoader):
        dataset = next(iter(dataset))
    if isinstance(dataset, Data):
        return len(dataset.batch.unique())
    else:
        raise TypeError(f"Invalid input dataset type: {type(dataset)}")
