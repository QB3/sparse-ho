import itertools
import numpy as np
from scipy.sparse import csc_matrix
from sklearn import linear_model

import celer
from celer.datasets import make_correlated_data

from sparse_ho.models import (
    Lasso, ElasticNet, WeightedLasso, SparseLogreg, SVM, SVR, SimplexSVR)
from sparse_ho.tests.cvxpylayer import (
    enet_cvxpy, weighted_lasso_cvxpy, logreg_cvxpy, lasso_cvxpy,
    lasso_sure_cvxpy, svm_cvxpy, svr_cvxpy, ssvr_cvxpy)

# Generate data
n_samples, n_features = 10, 10
X, y, _ = make_correlated_data(
    n_samples, n_features, corr=0.1, snr=3, random_state=42)
sigma_star = 0.1

y = np.sign(y)
X_s = csc_matrix(X)
idx_train = np.arange(0, n_samples//2)
idx_val = np.arange(n_samples//2, n_features)

# Set alpha for the Lasso
alpha_max = (np.abs(X[idx_train, :].T @ y[idx_train])).max() / n_samples
p_alpha = 0.8
alpha = p_alpha * alpha_max
log_alpha = np.log(alpha)
log_alpha_max = np.log(alpha_max)

# Set alpha1 alpha2 for the enet
alpha_1 = p_alpha * alpha_max
alpha_2 = 0.1
log_alpha1 = np.log(alpha_1)
log_alpha2 = np.log(alpha_2)

dict_log_alpha = {}
dict_log_alpha["lasso"] = log_alpha
dict_log_alpha["enet"] = np.array([log_alpha1, log_alpha2])
tab = np.linspace(1, 1000, n_features)
dict_log_alpha["wLasso"] = log_alpha + np.log(tab / tab.max())
dict_log_alpha["logreg"] = (log_alpha - np.log(2))
dict_log_alpha["svm"] = 1e-4
dict_log_alpha["svr"] = np.log(np.array([1e-2, 1e-1]))
dict_log_alpha["ssvr"] = np.log(np.array([0.01, 0.1]))

# Set models to be tested
models = {}
models["lasso"] = Lasso(estimator=None)
models["enet"] = ElasticNet(estimator=None)
models["wLasso"] = WeightedLasso(estimator=None)
models["logreg"] = SparseLogreg(estimator=None)
models["svm"] = SVM(estimator=None)
models["svr"] = SVR(estimator=None)
models["ssvr"] = SimplexSVR(estimator=None)


custom_models = {}
custom_models["lasso"] = Lasso(estimator=celer.Lasso(
    warm_start=True, fit_intercept=False))
custom_models["enet"] = ElasticNet(
    estimator=linear_model.ElasticNet(warm_start=True, fit_intercept=False))
custom_models["logreg"] = SparseLogreg(
    estimator=celer.LogisticRegression(warm_start=True, fit_intercept=False))

# Compute "ground truth" with cvxpylayer
dict_cvxpy_func = {
    'lasso': lasso_cvxpy,
    'enet': enet_cvxpy,
    'wLasso': weighted_lasso_cvxpy,
    'logreg': logreg_cvxpy,
    'svm': svm_cvxpy,
    'svr': svr_cvxpy,
    'ssvr': ssvr_cvxpy
}

dict_vals_cvxpy = {}
dict_grads_cvxpy = {}
for model in models.keys():
    val_cvxpy, grad_cvxpy = dict_cvxpy_func[model](
        X, y, np.exp(dict_log_alpha[model]), idx_train, idx_val)
    dict_vals_cvxpy[model, 'MSE'] = val_cvxpy
    grad_cvxpy *= np.exp(dict_log_alpha[model])
    dict_grads_cvxpy[model, 'MSE'] = grad_cvxpy

val_cvxpy, grad_cvxpy = lasso_sure_cvxpy(
    X, y, np.exp(dict_log_alpha["lasso"]), sigma_star)
grad_cvxpy *= np.exp(dict_log_alpha["lasso"])
dict_vals_cvxpy["lasso", "SURE"] = val_cvxpy
dict_grads_cvxpy["lasso", "SURE"] = grad_cvxpy


# log alpha to be tested by checkgrad
dict_list_log_alphas = {}
dict_list_log_alphas["lasso"] = np.log(
    np.geomspace(alpha_max/2, alpha_max/5, num=5))
dict_list_log_alphas["wLasso"] = [
    log_alpha * np.ones(n_features) for log_alpha in
    dict_list_log_alphas["lasso"]]
dict_list_log_alphas["logreg"] = np.log(
    np.geomspace(alpha_max/5, alpha_max/40, num=5))
dict_list_log_alphas["enet"] = [np.array(i) for i in itertools.product(
    dict_list_log_alphas["lasso"], dict_list_log_alphas["lasso"])]
dict_list_log_alphas["svm"] = np.log(np.geomspace(1e-8, 1e-5, num=5))
dict_list_log_alphas["svr"] = [
    np.array(i) for i in itertools.product(
        np.log(np.geomspace(1e-2, 1e-1, num=5)),
        np.log(np.geomspace(1e-2, 1e-1, num=5)))]
dict_list_log_alphas["ssvr"] = [
    np.array(i) for i in itertools.product(
        np.log(np.geomspace(0.01, 0.1, num=5)),
        np.log(np.geomspace(0.01, 0.1, num=5)))]


def get_grad_outer(mask, dense):
    return 2 * (X[np.ix_(idx_val, mask)].T @ (
        X[np.ix_(idx_val, mask)] @ dense - y[idx_val])) / len(idx_val)


list_model_crit = [
    ('lasso', 'MSE'),
    ('enet', 'MSE'),
    ('wLasso', 'MSE'),
    ('lasso', 'SURE'),
    ('logreg', 'logistic'),
    ('svm', 'MSE'),
    ('svr', 'MSE'),
    ('ssvr', 'MSE')
]

list_model_names = ["lasso", "enet", "wLasso", "logreg", "svm", "svr", "ssvr"]
