# import time
import numpy as np
from numpy.linalg import norm

from sklearn.datasets import make_regression
# from sklearn.linear_model import LassoCV
# from celer.datasets import load_libsvm
# from sparse_ho.datasets.real import load_libsvm
from sparse_ho.models import Lasso

# X, y = load_libsvm('rcv1_train')
X, y = make_regression(
    n_samples=1000, n_features=1000)

print("Starting path computation...")
n_samples = len(y)
alpha_max = np.max(np.abs(X.T.dot(y))) / n_samples

n_alphas = 100
alphas = alpha_max * np.geomspace(1, 0.001, n_alphas)

tol = 1e-3
tol_celer = tol * norm(y) ** 2

# print('scikit started')

# t0 = time.time()
# reg = LassoCV(cv=5, random_state=0, verbose=True).fit(X, y)
# reg.score(X, y)
# t_sk = time.time() - t0

# print('scikit finished')

# print("Time to compute CV for scikit: %.2f" % t_sk)


model = Lasso(X, y, np.log(alpha_max/10), use_sk=True)


# print('celer started')

# t0 = time.time()
# _, coefs_celer, dual_gaps_celer = celer_path(
#     X, y, tol=tol_celer, alphas=alphas, verbose=True, pb='lasso')
# t_celer = time.time() - t0

# print('Celer finished')

# print("Time to compute path for celer: %.2f" % t_celer)

# print("np.abs(coefs_celer).sum()", np.abs(coefs_celer).sum())
