import numpy as np
from numpy.linalg import norm
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from celer.datasets import fetch_libsvm
from celer import LogisticRegression


n_samples = 100
X, y = fetch_libsvm('mnist')
X = X[:n_samples, :]
y = y[:n_samples]
n_samples, n_features = X.shape

enc = OneHotEncoder(sparse=False)
# return a n_samples * n_classes matrix
one_hot_code = enc.fit_transform(y.reshape(-1, 1))
n_classes = one_hot_code.shape[1]

# one hot encoding sur
all_coefs = np.zeros((n_features, n_classes))

for k in range(n_classes):
    X_train, X_test, y_train, y_test = train_test_split(
        X, one_hot_code[:, k], random_state=42)
    C_min = 2 / norm(X_train.T @ y_train, ord=np.inf)
    clf = LogisticRegression(C=C_min * 10, fit_intercept=False)
    clf.fit(X_train, y_train)
    coefs = clf.coef_.ravel()
    all_coefs[:, k] = coefs.copy()
