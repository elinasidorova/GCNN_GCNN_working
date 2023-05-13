import os

import joblib
import numpy as np
from rdkit import Chem
from scipy import sparse
from sklearn import model_selection
from sklearn.neighbors import KernelDensity

from Source.applicability_domain.knn_ad import ecfp_molstring


def get_distance(x_predict, centroid, train_mean, train_shape):
    """
    :param x_predict: feature vector for prediction
    :param centroid: centroid of train set
    :param train_mean: mean value of train set
    :param train_shape: shape of train set
    :return: Mahalanobis distance between feature vector and x_train
    """
    variance_covariance_matrix = (centroid.T @ centroid) / train_shape
    mahalanobis_distance = np.sqrt(
        (x_predict - train_mean) @ variance_covariance_matrix.T @ (x_predict - train_mean).T
    )
    return mahalanobis_distance


def estimate_density(x_train, sub_folder, density_model_filename, scale=None):
    """
    Build model for mutivariate kernel density estimation.
    Takes x_train dataframe as input, builds a kernel density model
    with optimised parameters for input vectors,
    and computes density for all input vectors.
    """

    if scale is None:
        bandwidth = np.logspace(-1, 2, 20)
    else:
        bandwidth = np.linspace(0.1, 0.5, 5)

    # find best parameters for KD
    grid = model_selection.GridSearchCV(KernelDensity(), {'bandwidth': bandwidth}, cv=3)
    grid.fit(x_train)

    # compute KD for x_train
    dens_model = KernelDensity(**grid.best_params_).fit(x_train)

    samples = dens_model.score_samples(x_train)

    dens_mean = np.mean(samples)
    dens_std = np.std(samples)

    if density_model_filename is not None:
        path_to_model = os.path.join(sub_folder, density_model_filename)
        joblib.dump(dens_model, path_to_model)

    return dens_model, dens_mean, dens_std


def estimate_distance(x_train, sub_folder, distance_matrix_filename, train_mean_filename):
    """
    Takes x_train dataframe as input,
    calculates Mahalonobis distance between whole input
    dataset and each input vector from the dataset
    """

    centroid = x_train - np.tile(
        np.mean(x_train, axis=0), (x_train.shape[0], 1)
    )
    train_mean = np.mean(x_train, axis=0)
    train_shape = x_train.shape[0] - 1
    x_train = np.asarray(x_train)
    dist_list = np.apply_along_axis(
        get_distance, 1, x_train, centroid=centroid, train_mean=train_mean,
        train_shape=train_shape
    )

    dist_mean = np.mean(dist_list)
    dist_std = np.std(dist_list)
    centroid = sparse.csr_matrix(np.asarray(centroid).astype(dtype='float32'))

    if train_mean_filename is not None:
        train_mean_file_path = os.path.join(sub_folder, train_mean_filename)
        np.savetxt(train_mean_file_path, train_mean)

    if distance_matrix_filename is not None:
        matrix_file_path = os.path.join(sub_folder, distance_matrix_filename)
        sparse.save_npz(matrix_file_path, centroid)

    return train_mean, centroid, dist_mean, dist_std


def check_mol(dens, dens_mean, dens_std, dist, dist_mean, dist_std):
    distance = 0 if dist > dist_mean + 3 * dist_std else 1
    density = 0 if dens < dens_mean - 3 * dens_std else 1
    return distance, density


def get_dataset_ad(x_train, x_test, output_folder=None, density_model_filename=None, distance_matrix_filename=None,
                   train_mean_filename=None):
    dens_model, dens_mean, dens_std = estimate_density(
        x_train, output_folder,
        density_model_filename=density_model_filename,
        scale=None)
    train_mean, distance_matrix, dist_mean, dist_std = estimate_distance(
        x_train, output_folder,
        distance_matrix_filename=distance_matrix_filename,
        train_mean_filename=train_mean_filename)

    ad_by_distances = []
    ad_by_densities = []
    for x_predict in x_test:
        train_shape = x_train.shape[0] - 1
        dist = get_distance(np.asarray(x_predict).reshape(train_mean.shape), distance_matrix, train_mean, train_shape)
        dens = abs(dens_model.score_samples(x_predict.reshape(1, -1)))
        distance_bool, density_bool = check_mol(dens, dens_mean, dens_std, dist, dist_mean, dist_std)
        ad_by_distances += [distance_bool]
        ad_by_densities += [density_bool]
    return np.array(ad_by_distances), np.array(ad_by_densities)


def get_sdfs_ad(test_dataset_path, train_dataset_path, sub_folder, density_model_filename, distance_matrix_filename,
                train_mean_filename, finp_r=4, finp_size=512):
    x_train = np.array([ecfp_molstring(mol, finp_r, finp_size) for mol in Chem.SDMolSupplier(train_dataset_path)])
    x_test = np.array([ecfp_molstring(mol, finp_r, finp_size) for mol in Chem.SDMolSupplier(test_dataset_path)])

    return get_dataset_ad(x_train, x_test, sub_folder, density_model_filename, distance_matrix_filename,
                          train_mean_filename)
