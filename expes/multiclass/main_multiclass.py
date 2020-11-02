# import numpy as np
# from numpy.linalg import norm
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.model_selection import train_test_split
# import pandas as pd

# from celer.datasets import fetch_libsvm
# from celer import LogisticRegression

# from sparse_ho.criterion import LogisticMulticlass

# n_samples = 1000
# n_features = 1000
# # X, y = fetch_libsvm('smallNORB')
# # X, y = fetch_libsvm('mnist')
# X, y = fetch_libsvm('rcv1_multiclass')
# np.random.seed(0)
# idx = np.random.choice(X.shape[0], n_samples, replace=False)
# feats = np.random.choice(X.shape[1], n_features, replace=False)
# X = X[idx, :]
# X = X[:, feats]
# y = y[idx]
# ypd = pd.DataFrame(y)
# bool_rm = ypd.groupby(0)[0].transform(len) > 1
# ypd = ypd[bool_rm]
# X = X[bool_rm.to_numpy(), :]
# y = y[bool_rm.to_numpy()]

# n_samples, n_features = X.shape

# # return a n_samples * n_classes matrix
# enc = OneHotEncoder(sparse=False)
# one_hot_code = enc.fit_transform(ypd)
# n_classes = one_hot_code.shape[1]


# # one hot encoding sur
# all_betas = np.zeros((n_features, n_classes))
# dict_clf = {}

# from sklearn.model_selection import StratifiedShuffleSplit
# sss = StratifiedShuffleSplit(1, test_size=0.5, random_state=0)
# sss.get_n_splits(X, y)

# for idx_ in sss.split(X, y):
#     idx_train, idx_test = idx_


# # random search
# nC = 20
# # all_Cs = np.exp(np.random.uniform(size=(n_classes, nC)) * np.log(100))

# # grid search
# # nC = 10
# all_Cs = np.geomspace(1, 1000, nC)

# mclass_ces = np.zeros(nC)


# # for idx_C, mult_C in enumerate(all_Cs):
# for idx_C in range(nC):
#     for k in range(n_classes):
#         # X_train, _, y_train, _ = train_test_split(
#         #     X, one_hot_code[:, k], random_state=42)
#         X_train = X[idx_train, :]
#         y_train = one_hot_code[idx_train, k]
#         C_min = 2 / norm(X_train.T @ y_train, ord=np.inf)
#         clf = dict_clf.get(k, LogisticRegression(
#             C=C_min, fit_intercept=False, warm_start=True, verbose=True))
#         try:
#             clf.C = C_min * all_Cs[k, idx_C]
#         except Exception:
#             clf.C = C_min * all_Cs[idx_C]
#         clf.fit(X_train, y_train)
#         coefs = clf.coef_.ravel()
#         all_betas[:, k] = coefs.copy()

#     X_test = X[idx_test, :]
#     y_test = y[idx_test]
#     # _, X_test, _, y_test = train_test_split(X, y, random_state=42)
#     # criterion = LogisticMulticlass(X, y)
#     # mclass_ce = criterion.cross_entropy(all_betas, X_test, y_test)
#     # print("mce: %0.2f" % mclass_ce)
#     # mclass_ces[idx_C] = mclass_ce

# # mclass_ces[np.isnan(mclass_ces)] = np.infty
# # print("mce mni: %0.2f" % mclass_ces.min())
