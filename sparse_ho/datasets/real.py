"""File to download and load real data from libsvm.
This code is mostly taken from
https://github.com/mathurinm/celer/blob/master/celer/datasets/libsvm.py.
"""

import os
from bz2 import BZ2Decompressor
from sklearn import preprocessing
from sklearn.datasets import load_svmlight_file
from download import download
from scipy import sparse
import numpy as np

from os.path import join as pjoin
from pathlib import Path

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

PATH = pjoin(str(Path.home()), 'imp_forward_data')


NAMES = {'rcv1_train': 'binary/rcv1_train.binary',
         'news20': 'binary/news20.binary',
         'finance': 'regression/log1p.E2006.train',
         'leu': 'binary/leu',
         'real-sim': 'binary/real-sim'}

N_FEATURES = {'rcv1_train': 47236,
              'finance': 4272227,
              'news20': 1355191,
              'leu': 7129,
              'real-sim': 20958}


def get_20newsgroup(sparse=True):
    print("Loading data...")
    newsgroups = fetch_20newsgroups()
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(newsgroups.data)
    y = newsgroups.target
    # X, y = load_libsvm("news20")

    X = X.tocsc()
    # NNZ = np.diff(X.indptr)  # number of non zero elements per feature
    # # keep only features with >=3 non zero values
    # X = X[:, NNZ >= 3]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.5, random_state=42)

    if not sparse:
        X_train = np.array(X_train.tocsc().todense())
        X_val = np.array(X_val.tocsc().todense())
        X_test = np.array(X_test.tocsc().todense())
    else:
        X_train = X_train.tocsc()
        X_val = X_val.tocsc()
        X_test = X_test.tocsc()
    print("Finished loading data...")

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_finance():
    X, y = load_libsvm("finance")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.5, random_state=42)
    X_train = X_train.tocsc()
    X_val = X_val.tocsc()
    X_test = X_test.tocsc()

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_rcv1(csr=False):
    X, y = load_libsvm("rcv1_train")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.5, random_state=42)
    if csr:
        X_train = X_train.tocsr()
        X_val = X_val.tocsr()
        X_test = X_test.tocsr()
    else:
        X_train = X_train.tocsc()
        X_val = X_val.tocsc()
        X_test = X_test.tocsc()

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_leukemia():
    X, y = load_libsvm("leu")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.5, random_state=42)
    X_train = X_train.tocsc()
    X_val = X_val.tocsc()
    X_test = X_test.tocsc()
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_real_sim(csr=False):
    X, y = load_libsvm("real-sim")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.5, random_state=42)
    if csr:
        X_train = X_train.tocsr()
        X_val = X_val.tocsr()
        X_test = X_test.tocsr()
    else:
        X_train = X_train.tocsc()
        X_val = X_val.tocsc()
        X_test = X_test.tocsc()
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_data(dataset_name):
    if dataset_name == "finance":
        return get_finance()
    elif dataset_name == "20newsgroups":
        return get_20newsgroup()
    elif dataset_name == "rcv1":
        return get_rcv1()
    elif dataset_name == "leukemia":
        return get_leukemia()
    elif dataset_name == "real-sim":
        return get_real_sim()
    else:
        raise ValueError("dataset_name %s do not exist" % dataset_name)


def download_libsvm(dataset, destination, replace=False):
    """Download a dataset from LIBSVM website."""
    if dataset not in NAMES:
        raise ValueError("Unsupported dataset %s" % dataset)
    url = ("https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/" +
           NAMES[dataset])
    path = download(url + '.bz2', destination, replace=replace)
    return path


def decompress_data(compressed_path, decompressed_path):
    """Decompress a Libsvm dataset."""
    decompressor = BZ2Decompressor()
    with open(decompressed_path, "wb") as f, open(compressed_path, "rb") as g:
        for data in iter(lambda: g.read(100 * 1024), b''):
            f.write(decompressor.decompress(data))


def preprocess_libsvm(dataset, decompressed_path, X_path, y_path,
                      is_regression=False):
    """Preprocess a LIBSVM dataset."""
    # Normalization performed:
    # - X with only columns with >= 3 non zero elements, norm-1 columns
    # - y centered and set to std equal to 1
    # """
    n_features_total = N_FEATURES[dataset]
    with open(decompressed_path, 'rb') as f:
        X, y = load_svmlight_file(f, n_features_total)
        X = sparse.csc_matrix(X)

        NNZ = np.diff(X.indptr)  # number of non zero elements per feature
        # keep only features with >=3 non zero values
        X_new = X[:, NNZ >= 3]

        # set all feature norms to 1
        X_new = preprocessing.normalize(X_new, axis=0)
        if is_regression:
            # center y
            y -= np.mean(y)
            # normalize y to get a first duality gap of 0.5
            y /= np.std(y)

        # very important for sparse/sparse dot products: have sorted X.indices
        X_new.sort_indices()
        sparse.save_npz(X_path, X_new)
        np.save(y_path, y)


def download_preprocess_libsvm(dataset, replace=False, repreprocess=False):
    """Download and preprocess a given libsvm dataset."""

    paths = [PATH, pjoin(PATH, 'regression'),
             pjoin(PATH, 'binary'),
             pjoin(PATH, 'preprocessed')]
    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)

    print("Dataset: %s" % dataset)
    compressed_path = pjoin(PATH, "%s.bz2" % NAMES[dataset])
    download_libsvm(dataset, compressed_path, replace=replace)

    decompressed_path = pjoin(PATH, "%s" % NAMES[dataset])
    if not os.path.isfile(decompressed_path):
        decompress_data(compressed_path, decompressed_path)

    y_path = pjoin(PATH, "preprocessed", "%s_target.npy" % dataset)
    X_path = pjoin(PATH, "preprocessed", "%s_data.npz" % dataset)

    if (repreprocess or not os.path.isfile(y_path) or
            not os.path.isfile(X_path)):
        print("Preprocessing...")
        preprocess_libsvm(dataset, decompressed_path, X_path, y_path)


def load_libsvm(dataset):
    try:
        X = sparse.load_npz(pjoin(PATH, 'preprocessed',
                                  '%s_data.npz' % dataset))
        y = np.load(pjoin(PATH, 'preprocessed',
                          '%s_target.npy' % dataset))
    except FileNotFoundError:
        download_preprocess_libsvm(dataset, replace=False, repreprocess=True)
        X = sparse.load_npz(pjoin(PATH, 'preprocessed',
                                  '%s_data.npz' % dataset))
        y = np.load(pjoin(PATH, 'preprocessed',
                          '%s_target.npy' % dataset))
    return X, y
