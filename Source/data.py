from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def train_test_valid_split(dataset, n_splits=5, test_ratio=0.2, batch_size=64, seed=14):
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

    Returns
    -------
    folds : list
        List of cross-validation folds in format (train_loader, valid_loader)
    test_loader : DataLoader
        Test DataLoader, which does not participate in cross-validation
    """
    dataset_size = len(dataset)
    ids = range(dataset_size)
    train_val_ids, test_ids = train_test_split(ids, test_size=test_ratio, random_state=seed) if test_ratio > 0 else (
        ids, [])
    test_loader = DataLoader([val for i, val in enumerate(dataset) if i in test_ids], batch_size=batch_size)

    if n_splits == 1:
        train_ids, val_ids = train_test_split(train_val_ids, test_size=test_ratio, random_state=seed)
        train_loader = DataLoader([val for i, val in enumerate(dataset) if i in train_ids], batch_size=batch_size)
        val_loader = DataLoader([val for i, val in enumerate(dataset) if i in val_ids], batch_size=batch_size)
        return ((train_loader, val_loader),), test_loader

    folds = []
    kf_split = KFold(n_splits=n_splits)
    for train_index, valid_index in kf_split.split(train_val_ids):
        train_loader = DataLoader([val for i, val in enumerate(dataset) if i in train_index], batch_size=batch_size)
        valid_loader = DataLoader([val for i, val in enumerate(dataset) if i in valid_index], batch_size=batch_size)
        folds += [(train_loader, valid_loader)]
    return folds, test_loader


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
