"""Module containing functions for loading data"""

import os

import numpy as np
from scipy.stats import zscore
from voxelwise_tutorials.io import download_datalad, load_hdf5_array
from voxelwise_tutorials.utils import explainable_variance

from .config import DATA_DIR

shortclips_dir = os.path.join(DATA_DIR, "shortclips")


def _check_data_downloaded(subject):
    """Check that the data is downloaded, if not download it."""
    files = [
        "features/motion_energy.hdf",
        "features/wordnet.hdf",
        f"mappers/{subject}_mappers.hdf",
        f"responses/{subject}_responses.hdf",
    ]
    source = "https://gin.g-node.org/gallantlab/shortclips"
    for f in files:
        if not os.path.exists(os.path.join(shortclips_dir, f)):
            download_datalad(f, destination=shortclips_dir, source=source)


def load_data_for_fitting(subject):
    """Load the data needed to fit the model.

    Parameters
    ----------
    subject : str
        Subject identifier. Can be 'S01', 'S02', 'S03', 'S04', or 'S05'.
        If the data is not already downloaded, it will be downloaded.

    Returns
    -------
    X_train : ndarray
        Training features (wordnet and motion_energy)
    X_test : ndarray
        Test features (wordnet and motion_energy)
    Y_train : ndarray
        Training responses
    Y_test : ndarray
        Test responses averaged across repetitions
    run_onsets : ndarray
        Onsets of the runs (used for cross-validation)
    n_features_list : list of int
        Number of features in each feature space
    feature_names : list of str
        Name of each feature space
    """
    _check_data_downloaded(subject)

    file_name = os.path.join(shortclips_dir, "responses", f"{subject}_responses.hdf")

    ##############################################################################
    # We load the run onsets which will be needed for cross-validation
    run_onsets = load_hdf5_array(file_name, key="run_onsets")

    ##############################################################################
    # Then we load the responses
    Y_train = load_hdf5_array(file_name, key="Y_train").astype(float)
    Y_test = load_hdf5_array(file_name, key="Y_test").astype(float)
    # We zscore the data within each run
    Y_train = np.array_split(Y_train, run_onsets[1:], axis=0)
    assert len(Y_train) == 12
    Y_train = np.concatenate([zscore(yt, axis=0) for yt in Y_train], axis=0)
    Y_test = zscore(Y_test, axis=1)

    ###############################################################################
    # We average the test responses across the repeats, to remove the non-repeatable
    # part of fMRI responses.
    Y_test = Y_test.mean(0)
    # Then we zscore again to ensure the same mean and std between train and test
    Y_test = zscore(Y_test, axis=0)
    # These are the final responses that we will model.
    print("(n_samples_train, n_voxels) =", Y_train.shape)
    print("(n_samples_test, n_voxels) =", Y_test.shape)

    ##############################################################################
    # And finally we load the features
    Xs_train, Xs_test, feature_names, n_features_list = load_features()

    # concatenate the feature spaces
    X_train = np.concatenate(Xs_train, 1).astype(float)
    X_test = np.concatenate(Xs_test, 1).astype(float)

    # demean the features within each run
    X_train = np.array_split(X_train, run_onsets[1:], axis=0)
    X_train = np.concatenate([xt - xt.mean(0) for xt in X_train], axis=0)
    X_test -= X_test.mean(0)

    print("(n_samples_train, n_features_total) =", X_train.shape)
    print("(n_samples_test, n_features_total) =", X_test.shape)
    print("[n_features_wordnet, n_features_motion_energy] =", n_features_list)

    # Finally cast to float32 for use with GPU
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    Y_train = Y_train.astype(np.float32)
    Y_test = Y_test.astype(np.float32)

    return X_train, X_test, Y_train, Y_test, run_onsets, n_features_list, feature_names


def load_features(feature_names=("wordnet", "motion_energy")):
    if isinstance(feature_names, str):
        feature_names = [feature_names]
    Xs_train = []
    Xs_test = []
    n_features_list = []
    for feature_space in feature_names:
        file_name = os.path.join(shortclips_dir, "features", f"{feature_space}.hdf")
        Xi_train = load_hdf5_array(file_name, key="X_train")
        Xi_test = load_hdf5_array(file_name, key="X_test")

        Xs_train.append(Xi_train)
        Xs_test.append(Xi_test)
        n_features_list.append(Xi_train.shape[1])
    return Xs_train, Xs_test, feature_names, n_features_list


def load_mapper(subject):
    _check_data_downloaded(subject)
    mapper_file = os.path.join(
        DATA_DIR, "shortclips", "mappers", f"{subject}_mappers.hdf"
    )
    mapper = load_hdf5_array(mapper_file)
    return mapper, mapper_file


def load_ev(subject):
    _check_data_downloaded(subject)
    file_name = os.path.join(shortclips_dir, "responses", f"{subject}_responses.hdf")
    Y_test = load_hdf5_array(file_name, key="Y_test")
    Y_test = np.nan_to_num(Y_test)
    ev = explainable_variance(Y_test)
    return ev
