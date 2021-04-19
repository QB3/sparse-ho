import numpy as np
from numpy.linalg import norm

import scipy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import pandas as pd


def get_splits(X, y, train_size=0.333):
    """
    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        Design matrix.
    y: ndarray, shape (n_samples,)
        Observation vector.
    train_size: float
        Proportion of the dataset to be used for the training
    """

    idx_train, idx = train_test_split(
        np.arange(len(y)), stratify=y, train_size=train_size)

    idx_val, idx_test = train_test_split(idx, stratify=y[idx], test_size=0.5)

    return idx_train, idx_val, idx_test


def clean_dataset(X, y, n_samples, n_features, seed=0):
    """Reduce the number of features and / or samples.
    And remove lines or columns with only 0.

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        Design matrix.
    y: ndarray, shape (n_samples,)
        Observation vector.
    n_samples: int
        Number of samples to keep
    n_features: int
        Number of features to keep
    seed: int
        Seed for the random selection of the samples or features
    """
    np.random.seed(seed)
    idx = np.random.choice(
        X.shape[0], min(n_samples, X.shape[0]), replace=False)
    feats = np.random.choice(
        X.shape[1], min(n_features, X.shape[1]), replace=False)
    X = X[idx, :]
    X = X[:, feats]
    y = y[idx]

    bool_to_keep = scipy.sparse.linalg.norm(X, axis=0) != 0
    X = X[:, bool_to_keep]
    bool_to_keep = scipy.sparse.linalg.norm(X, axis=1) != 0
    X = X[bool_to_keep, :]
    y = y[bool_to_keep]

    ypd = pd.DataFrame(y)
    bool_to_keep = ypd.groupby(0)[0].transform(len) > 2
    ypd = ypd[bool_to_keep]
    X = X[bool_to_keep.to_numpy(), :]
    y = y[bool_to_keep.to_numpy()]

    bool_to_keep = scipy.sparse.linalg.norm(X, axis=0) != 0
    X = X[:, bool_to_keep]
    bool_to_keep = scipy.sparse.linalg.norm(X, axis=1) != 0
    X = X[bool_to_keep, :]
    y = y[bool_to_keep]

    return X, y


def alpha_max_multiclass(X, y):
    """
    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        Design matrix.
    y: ndarray, shape (n_samples,)
        Observation vector.
    """
    ypd = pd.DataFrame(y)
    one_hot_code = OneHotEncoder(sparse=False).fit_transform(ypd)
    n_classes = one_hot_code.shape[1]

    alpha_max = np.infty
    for k in range(n_classes):
        alpha_max = min(alpha_max, norm(
            X.T @ (2 * one_hot_code[:, k] - 1), ord=np.inf) / (2 * X.shape[0]))
        if alpha_max == 0:
            raise ValueError("X and y are uncorrelated")
    return alpha_max, n_classes
