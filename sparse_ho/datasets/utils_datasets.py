import numpy as np
from numpy.linalg import norm

import scipy

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder

import pandas as pd


def get_splits(
        X, y,
        sss1=StratifiedShuffleSplit(
            n_splits=2, test_size=0.3333, random_state=0),
        sss2=StratifiedShuffleSplit(
            n_splits=2, test_size=0.5, random_state=0)):
    for i, (idx_train, idx_test) in enumerate(sss1.split(X, y)):
        # _ = X[idx_test, :]
        # y_test = y[idx_test]
        X_train = X[idx_train, :]
        y_train = y[idx_train]
        if i == 0:
            break

    for i, (idx_train, idx_val) in enumerate(sss2.split(X_train, y_train)):
        # X_val = X_train[idx_val, :]
        # y_val = y[idx_val]
        # X_train = X_train[idx_train, :]
        # y_train = y[idx_train]
        if i == 0:
            break

    # X_train = X_train.tocsc()
    # X_val = X_val.tocsc()
    # X_test = X_test.tocsc()

    return idx_train, idx_val, idx_test


def clean_dataset(X, y, n_samples, n_features, seed=0):
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

    # ypd = pd.DataFrame(y)
    # bool_to_keep = ypd.groupby(0)[0].transform(len) > 2
    # ypd = ypd[bool_to_keep]
    # X = X[bool_to_keep.to_numpy(), :]
    # y = y[bool_to_keep.to_numpy()]

    return X, y


def get_alpha_max(X, y):
    ypd = pd.DataFrame(y)
    enc = OneHotEncoder(sparse=False)
    one_hot_code = enc.fit_transform(ypd)

    one_hot_code = enc.fit_transform(ypd)
    n_classes = one_hot_code.shape[1]

    alpha_max = np.infty
    for k in range(n_classes):
        alpha_max = min(alpha_max, norm(
            X.T @ (2 * one_hot_code[:, k] - 1), ord=np.inf) / (2 * X.shape[0]))
        if alpha_max == 0:
            1 / 0
    return alpha_max, n_classes
