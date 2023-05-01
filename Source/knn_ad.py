import numpy as np
from rdkit import Chem
from sklearn.neighbors import KDTree
from rdkit.Chem import AllChem, DataStructs


def ecfp_molstring(molecule, radius, size):
    """
    Method for make molstring for ecfp fingerprint

    :param molecule: molecule object
    :param fptype: type, radius and size of fingerprint
    :type fptype: dict
    :return: molstring for ecfp fingerprint
    """
    arr = np.zeros((1,), dtype=int)
    DataStructs.ConvertToNumpyArray(
        AllChem.GetMorganFingerprintAsBitVect(
            molecule, radius, size, useFeatures=False
        ), arr
    )

    return arr


def comp_mean_dists(dataset: np.array, n_neighb=5, leaf_size=10):
    tree = KDTree(dataset, leaf_size)
    dist, ind = tree.query(dataset, k=n_neighb + 1)
    mean = [np.mean(i) for i in dist[1:]]
    return tree, mean


def single_vec_dist(input_vector: np.array, dataset: np.array, t_values: np.array):
    distances = [np.linalg.norm(input_vector - temp_vector) for temp_vector in dataset]

    for dist, t_val in zip(distances, t_values):
        if dist <= t_val:
            return True

    return False


def get_dataset_ad(test_dataset: np.array, train_dataset: np.array, n_neighb=5, leaf_size=10):
    tree, mean = comp_mean_dists(train_dataset, n_neighb, leaf_size)
    r_ref = np.percentile(mean, 75) + 1.5 * (np.percentile(mean, 75) - np.percentile(mean, 25))
    dist_matr, ind = tree.query(train_dataset, train_dataset.shape[0])
    counts = []
    for line in dist_matr:
        condition = line[1:] < r_ref
        counts.append(len(np.extract(condition, line[1:])))

    nearest_dist, ind = tree.query(train_dataset, n_neighb)
    t_values = []
    for n, d in zip(counts, nearest_dist):
        if n > 0:
            t_values.append(sum(d) / n)
        else:
            t_values.append(np.nan)

    for i, value in enumerate(t_values):
        if value is None:
            t_values[i] = np.nanmin(t_values)

    results = [single_vec_dist(i, train_dataset, t_values) for i in test_dataset]

    return results


def get_sdfs_ad(test_dataset_path: str, train_dataset_path: str, n_neighb=3, leaf_size=10, finp_r=4, finp_size=512):
    train_dataset = np.array([ecfp_molstring(mol, finp_r, finp_size) for mol in Chem.SDMolSupplier(train_dataset_path)])
    test_dataset = np.array([ecfp_molstring(mol, finp_r, finp_size) for mol in Chem.SDMolSupplier(test_dataset_path)])

    return get_dataset_ad(test_dataset, train_dataset, n_neighb, leaf_size)
