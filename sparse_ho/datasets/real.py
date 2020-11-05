"""File to download and load real data from libsvm, using libsvmdata.
"""

from libsvmdata import fetch_libsvm
from sklearn.model_selection import train_test_split


def get_data(dataset_name, csr=False):
    X, y = fetch_libsvm(dataset_name)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.5, random_state=42)
    # be careful train_test_split returns crs matrices

    if csr:
        X_train = X_train.tocsr()
        X_val = X_val.tocsr()
        X_test = X_test.tocsr()
    else:
        X_train = X_train.tocsc()
        X_val = X_val.tocsc()
        X_test = X_test.tocsc()

    print("Finished loading data: %s ..." % dataset_name)

    return X_train, X_val, X_test, y_train, y_val, y_test
