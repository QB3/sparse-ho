"""
======================
Grad Search
======================

...

"""

# Authors: Quentin Bertrand <quentin.bertrand@inria.fr>
#          Quentin Klopfenstein <quentin.klopfenstein@u-bourgogne.fr>
#
# License: BSD (3-clause)

# import numpy as np
# import matplotlib.pyplot as plt

# from sparse_ho.models import Lasso
# from sparse_ho.criterion import CV
# from sparse_ho.implicit_forward import ImplicitForward
# from sparse_ho.forward import Forward
# from sparse_ho.ho import grad_search
# from sparse_ho.utils import Monitor
# from sparse_ho.datasets.real import get_real_sim
# # from sparse_ho.datasets.real import get_rcv1
# # from sparse_ho.datasets.real import get_leukemia
# from sparse_ho.grid_search import grid_search

# X_train, X_val, X_test, y_train, y_val, y_test = get_real_sim()
# # X_train, X_val, X_test, y_train, y_val, y_test = get_rcv1()
# # X_train, X_val, X_test, y_train, y_val, y_test = get_leukemia()
# n_samples, n_features = X_train.shape

# print("Starting path computation...")
# n_samples = len(y_train)
# alpha_max = np.max(np.abs(X_train.T.dot(y_train))) / X_train.shape[0]
# log_alpha0 = np.log(alpha_max / 10)

# n_alphas = 100
# alphas = alpha_max * np.geomspace(1, 0.001, n_alphas)
# log_alphas = np.log(alphas)

# tol = 1e-7

# model = Lasso(X_train, y_train, np.log(alpha_max/10))
# criterion = CV(X_val, y_val, model, X_test=X_test, y_test=y_test)
# algo = Forward(criterion, use_sk=False)
# monitor_grid = Monitor()
# grid_search(
#     algo, None, None, monitor_grid, log_alphas=log_alphas,
#     tol=tol)

# # grid search
# model = Lasso(X_train, y_train, np.log(alpha_max/10))
# criterion = CV(X_val, y_val, model, X_test=X_test, y_test=y_test)
# algo = Forward(criterion, use_sk=True)
# monitor_grid_sk = Monitor()
# grid_search(
#     algo, None, None, monitor_grid_sk, log_alphas=log_alphas,
#     tol=tol)

# # grad search
# model = Lasso(X_train, y_train, np.log(alpha_max/10))
# criterion = CV(X_val, y_val, model, X_test=X_test, y_test=y_test)
# algo = Forward(criterion, use_sk=False)
# monitor_grad = Monitor()
# grad_search(
#     algo, log_alpha0, monitor_grad, tol=tol, n_outer=50)
# print('grad search finished')

# # grad search
# model = Lasso(X_train, y_train, np.log(alpha_max/10))
# criterion = CV(X_val, y_val, model, X_test=X_test, y_test=y_test)
# algo = ImplicitForward(criterion, use_sk=True)
# monitor_grad_sk = Monitor()
# grad_search(
#     algo, log_alpha0, monitor_grad_sk, tol=tol, n_outer=50)
# print('grad search finished')

# plt.figure()
# plt.semilogy(monitor_grid.times, monitor_grid.objs, label='grid-search')
# plt.semilogy(monitor_grid_sk.times, monitor_grid_sk.objs, label='grid-search sk')
# plt.semilogy(monitor_grad.times, monitor_grad.objs, label='grad-search')
# plt.semilogy(
#     monitor_grad_sk.times, monitor_grad_sk.objs, label='grad-search-sk')
# #     monitor_wolfe.times, monitor_wolfe.objs, label='grad-search')
# plt.legend()
# plt.show(block=False)
