import numpy as np

from sparse_ho.datasets.real import get_data


def test_20news():
    X_train, X_val, X_test, y_train, y_val, y_test = get_data("rcv1_train")
    # X_train, X_val, X_test, y_train, y_val, y_test = get_data("news20")

    np.testing.assert_equal(X_train.shape[0], y_train.shape[0])
    np.testing.assert_equal(X_test.shape[0], y_test.shape[0])
    np.testing.assert_equal(X_val.shape[0], y_val.shape[0])

    np.testing.assert_equal(X_train.shape[1], X_test.shape[1])
    np.testing.assert_equal(X_train.shape[1], X_val.shape[1])
