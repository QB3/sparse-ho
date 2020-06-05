import time
import numpy as np
# import scipy

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sparse_ho.models import Lasso
from sparse_ho.criterion import CV
from sparse_ho.implicit_forward import ImplicitForward
from sparse_ho.forward import Forward
from sparse_ho.ho import grad_search_wolfe
# from sklearn.datasets import make_regression
# from sparse_ho.ho import grad_search, grad_search_wolfe
from sparse_ho.utils import Monitor
from sparse_ho.datasets.real import load_libsvm
from sparse_ho.grid_search import grid_search

X, y = load_libsvm('leu')
# X = X.todense()
# X, y = make_regression(
#     n_samples=3000, n_features=1000)
# X = scipy.sparse.csc_matrix(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.5, random_state=42)


print("Starting path computation...")
n_samples = len(y)
alpha_max = np.max(np.abs(X_train.T.dot(y_train))) / X_train.shape[0]

n_alphas = 10
alphas = alpha_max * np.geomspace(1, 0.001, n_alphas)

tol = 1e-7


print('grid search started')

t0 = time.time()
model = Lasso(X_train, y_train, np.log(alpha_max/10), use_sk=True)
criterion = CV(X_val, y_val, model, X_test=X_test, y_test=y_test)
algo = Forward(criterion, use_sk=True)
monitor_grid = Monitor()
grid_search(
    algo, None, None, monitor_grid, log_alphas=np.log(alphas))
t_gradsearch = time.time() - t0

print('grid search finished')

print('grid search started')

t0 = time.time()
model = Lasso(X_train, y_train, np.log(alpha_max/10))
criterion = CV(X_val, y_val, model, X_test=X_test, y_test=y_test)
algo = Forward(criterion, use_sk=False)
monitor_grid_us = Monitor()
grid_search(
    algo, None, None, monitor_grid_us, log_alphas=np.log(alphas),
    tol=tol)
t_gradsearch = time.time() - t0

print('grid search finished')

# 1 / 0

print('grad search started')

t0 = time.time()
model = Lasso(X_train, y_train, np.log(alpha_max/10))
criterion = CV(X_val, y_val, model, X_test=X_test, y_test=y_test)
algo = ImplicitForward(criterion, n_iter_jac=100)
monitor_wolfe = Monitor()
grad_search_wolfe(
    algo, np.log(alpha_max/10), monitor_wolfe, n_outer=10, tol=tol, maxit_ln=5)
t_gradsearch = time.time() - t0

print('grad search finished')

plt.figure()
plt.semilogy(monitor_grid.times, monitor_grid.objs, label='grid-search')
# plt.semilogy(
#     monitor_wolfe.times, monitor_wolfe.objs, label='grad-search')
plt.legend()
plt.show(block=False)

# plt.figure()
# plt.semilogy(monitor_grid.times, monitor_grid.objs_test, label='grid-search')
# plt.semilogy(
#     monitor_wolfe.times, monitor_wolfe.objs_test, label='grad-search')
# plt.legend()
# plt.show(block=False)


# plt.figure()
# plt.plot(monitor_grid.log_alphas, monitor_grid.objs)
# plt.plot(monitor_wolfe.log_alphas, monitor_wolfe.objs)
# plt.ylabel("value on validation set")
# plt.xlabel("log alpha")
# plt.show(block=False)
