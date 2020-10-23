import numpy as np
from numpy.linalg import norm
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from celer.datasets import fetch_libsvm
from celer import LogisticRegression

from sparse_ho.criterion import LogisticMulticlass

n_samples = 100
X, y = fetch_libsvm('mnist')
X = X[:n_samples, :]
y = y[:n_samples]
n_samples, n_features = X.shape

enc = OneHotEncoder(sparse=False)
# return a n_samples * n_classes matrix
one_hot_code = enc.fit_transform(y.reshape(-1, 1))
n_classes = one_hot_code.shape[1]

nC = 500
# one hot encoding sur
all_betas = np.zeros((n_features, n_classes))
dict_clf = {}
list_mult_C = np.geomspace(1, 100, nC)

np.random.seed(0)
all_Cs = np.exp(np.random.uniform(size=(n_classes, nC)) * 5)

mclass_ces = np.zeros(nC)

# for idx_C, mult_C in enumerate(all_Cs):
for idx_C in range(nC):
    for k in range(n_classes):
        X_train, _, y_train, _ = train_test_split(
            X, one_hot_code[:, k], random_state=42)
        C_min = 2 / norm(X_train.T @ y_train, ord=np.inf)
        clf = dict_clf.get(k, LogisticRegression(
            C=C_min, fit_intercept=False, warm_start=True))
        clf.C = C_min * all_Cs[k, idx_C]
        clf.fit(X_train, y_train)
        coefs = clf.coef_.ravel()
        all_betas[:, k] = coefs.copy()

    _, X_test, _, y_test = train_test_split(X, y, random_state=42)
    criterion = LogisticMulticlass()
    mclass_ce = criterion.cross_entropy(all_betas, X_test, y_test)
    print("mce: %0.2f" % mclass_ce)
    mclass_ces[idx_C] = mclass_ce
