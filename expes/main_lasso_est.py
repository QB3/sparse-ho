# import numpy as np
# from numpy.linalg import norm
# from joblib import Parallel, delayed
# import pandas
# from sklearn.linear_model import Lasso
# from itertools import product

# from sparse_ho.datasets.synthetic import get_synt_data
# from sparse_ho.lasso import (
#         Monitor, grid_searchCV, line_search, WarmStart)
# from sparse_ho.bayesian import hyperopt_lasso
# from sparse_ho.utils import iou_beta
# from sparse_ho.utils import my_lasso


# def err_cv(X, y, beta_hat):
#     n_samples, _ = X.shape
#     return norm(X @ beta_hat - y) ** 2 / norm(y) ** 2


# def err_pred(X, beta_hat, beta_star, y):
#     return norm(X @ (beta_hat - beta_star)) ** 2 / norm(y) ** 2


# def err_est(beta_hat, beta_star):
#     return norm(beta_hat - beta_star) / norm(beta_star)


# def get_metrics(X, y, beta_hat, beta_star):
#     return (
#         err_cv(X, y, beta_hat), err_pred(X, beta_hat, beta_star, y),
#         err_est(beta_hat, beta_star), iou_beta(beta_hat, beta_star))


# def get_beta(X, y, alpha):
#     clf = Lasso(
#             alpha=alpha, fit_intercept=False, warm_start=True, tol=tol,
#             max_iter=10000)
#     clf.fit(X, y)
#     return clf.coef_


# def parallel_function(
#         model, method, n_samples, n_features, SNR, n_active, rho, p_alpha_min,
#         n_pts_roc, seed, tol, n_outer):

#     X, y, beta_star, noise, sigma_star = get_synt_data(
#         "Gaussian", noise_type="Gaussian_iid", n_samples=n_samples,
#         n_features=n_features, n_times=1, n_active=n_active, SNR=SNR,
#         seed=seed)

#     alpha_max = norm(X.T @ y, ord=np.inf) / n_samples
#     p_alphas = np.geomspace(1, p_alpha_min, n_pts_roc)
#     log_alphas = np.log(alpha_max * p_alphas)

#     monitor = Monitor()
#     warm_start = WarmStart()
#     if model == "lasso":
#         if method == "GridSearch":
#             func = grid_searchCV
#             kwargs = {
#                 'X': X, 'y': y, 'X_val': None, 'y_val': None, 'X_test': None,
#                 'y_test': None, 'log_alphas': log_alphas,
#                 'tol': tol, 'monitor': monitor, 'sk': False,
#                 'sigma': sigma_star, 'criterion': "sure",
#                 'beta_star': beta_star}
#         elif method == "implicit_forward":
#             func = line_search
#             kwargs = {
#                 'X_train': X, 'y_train': y, 'X_val': None, 'y_val': None,
#                 'X_test': None, 'y_test': None,
#                 'log_alpha0': np.log(rho * alpha_max),
#                 'tol': tol, 'n_outer': n_outer,
#                 'monitor': monitor, 'method': "implicit_forward",
#                 'niter_jac': 500, 'warm_start': warm_start, 'model': "lasso",
#                 'sigma': sigma_star,
#                 'criterion': "sure", 'beta_star': beta_star}
#         elif method == "implicit":
#             func = line_search
#             kwargs = {
#                 'X_train': X, 'y_train': y, 'X_val': None, 'y_val': None,
#                 'X_test': None, 'y_test': None,
#                 'log_alpha0': np.log(rho * alpha_max),
#                 'tol': tol, 'n_outer': n_outer,
#                 'monitor': monitor, 'method': "implicit", 'niter_jac': 500,
#                 'warm_start': warm_start, 'model': "lasso",
#                 'sigma': sigma_star,
#                 'criterion': "sure", 'beta_star': beta_star}
#         elif method == "forward":
#             func = line_search
#             kwargs = {
#                 'X_train': X, 'y_train': y, 'X_val': None, 'y_val': None,
#                 'X_test': None, 'y_test': None,
#                 'log_alpha0': np.log(rho * alpha_max),
#                 'tol': tol, 'n_outer': n_outer,
#                 'monitor': monitor, 'method': "forward", 'niter_jac': 500,
#                 'warm_start': warm_start, 'model': "lasso",
#                 'sigma': sigma_star,
#                 'criterion': "sure", 'beta_star': beta_star}
#         elif method == "bayesian":
#             func = hyperopt_lasso
#             kwargs = {
#                 'X_train': X, 'y_train': y, 'X_val': None, 'y_val': None,
#                 'X_test': None, 'y_test': None,
#                 'log_alpha': np.log(rho * alpha_max), 'tol': tol,
#                 'maxit': 1000, 'max_evals': n_outer, 'method': "bayesian",
#                 'sigma': sigma_star, 'criterion': "sure",
#                 'beta_star': beta_star}
#         elif method == "random":
#             func = grid_searchCV
#             kwargs = {
#                 'X': X, 'y': y, 'X_val': None, 'y_val': None, 'X_test': None,
#                 'y_test': None, 'log_alphas': log_alphas,
#                 'tol': tol, 'monitor': monitor, 'sk': False,
#                 'sigma': sigma_star, 'criterion': "sure",
#                 'beta_star': beta_star,
#                 'random': True, 'max_evals': 100}

#     if method == "GridSearch" or method == "random":
#         func(**kwargs)
#     elif method == "bayesian":
#         monitor = func(**kwargs)
#     else:
#         func(**kwargs)

#     alpha_opt = np.exp(monitor.log_alphas[np.argmin(monitor.objs)])
#     if model != "alasso":
#         beta_hat = get_beta(X, y, alpha_opt)
#     else:
#         beta_hat = my_lasso(X, y, alpha_opt + 1e-5)
#     p_alpha_opt = alpha_opt / alpha_max
#     metrics = get_metrics(X, y, beta_hat, beta_star)

#     return (model, method, n_samples, n_features, SNR, n_active, rho,
#             p_alpha_min, p_alpha_opt, alpha_max, monitor.times[-1], seed,
#             *metrics, monitor.rmse)


# repeat = 1
# # uncomment for more repetitions
# # repeat = 50
# n_num = 10
# model = "lasso"
# list_method = [
#     "GridSearch", "random", "bayesian", "forward", "implicit_forward",
#     "implicit"]
# n_samples = 100

# list_n_features = np.linspace(200, 10000, num=n_num, dtype=int)
# list_seed = np.arange(0, repeat, 1)
# SNR = 3.0
# n_active = 5
# p_alpha_min = 0.0001
# n_pts_roc = 50
# tol = 1e-5
# rho = 0.7
# n_jobs = -1
# n_outer = 50

# print('Begin parallel')
# results = Parallel(n_jobs=n_jobs, verbose=100, backend='multiprocessing'(
#     delayed(parallel_function)(
#         model, method, n_samples, nfeatures, SNR, n_active, rho, p_alpha_min,
#         n_pts_roc, seed, tol, n_outer)
#     for method, nfeatures, seed in product(
#         list_method, list_n_features, list_seed))
# print('OK finished parallel')

# df = pandas.DataFrame(results)
# df.columns = [
#     'Model','method', 'n', 'p', 'SNR', 'Sparsity', 'rho', 'per alpha min',
#     'per alpha opt', 'alpha max', 'time', 'seed',
#     'Error cv', 'Error pred', 'Error est','IOU', 'mrse']


# df.to_pickle("%s.pkl" % "results_lasso_cor")
