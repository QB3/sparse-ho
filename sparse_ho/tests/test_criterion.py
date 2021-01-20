import pytest
import itertools
import numpy as np
from sklearn import linear_model
import celer
from celer.datasets import make_correlated_data

from sparse_ho.models import Lasso, ElasticNet, WeightedLasso, SparseLogreg
from sparse_ho.criterion import (
    HeldOutMSE, HeldOutLogistic, FiniteDiffMonteCarloSure)
from sparse_ho.utils import Monitor
from sparse_ho import Forward

# Generate data
n_samples, n_features = 10, 10
X, y, _ = make_correlated_data(
    n_samples, n_features, corr=0.1, snr=3, random_state=42)
sigma_star = 0.1

y = np.sign(y)
idx_train = np.arange(0, n_samples//2)
idx_val = np.arange(n_samples//2, n_features)

# Set alpha for the Lasso
alpha_max = (np.abs(X[idx_train, :].T @ y[idx_train])).max() / n_samples
p_alpha = 0.8
alpha = p_alpha * alpha_max
log_alpha = np.log(alpha)
log_alpha_max = np.log(alpha_max)
tol = 1e-15

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

# Set models to be tested
models = {}
models["lasso"] = Lasso(estimator=None)
models["enet"] = ElasticNet(estimator=None)
models["wLasso"] = WeightedLasso(estimator=None)
models["logreg"] = SparseLogreg(estimator=None)

custom_models = {}
custom_models["lasso"] = Lasso(estimator=celer.Lasso(
    warm_start=True, fit_intercept=False))
custom_models["enet"] = ElasticNet(
    estimator=linear_model.ElasticNet(warm_start=True, fit_intercept=False))
custom_models["logreg"] = SparseLogreg(
    estimator=celer.LogisticRegression(warm_start=True, fit_intercept=False))


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

list_model_crit = [
    ('lasso',  HeldOutMSE(idx_train, idx_val)),
    ('enet', HeldOutMSE(idx_train, idx_val)),
    ('wLasso', HeldOutMSE(idx_train, idx_val)),
    ('lasso', FiniteDiffMonteCarloSure(sigma_star)),
    ('logreg', HeldOutLogistic(idx_train, idx_val))]


@pytest.mark.parametrize('model_name,criterion', list_model_crit)
def test_cross_val_criterion(model_name, criterion):
    # verify dtype from criterion, bonne shape
    algo = Forward()
    monitor_get_val = Monitor()
    monitor_get_val_grad = Monitor()

    model = models[model_name]
    for log_alpha in dict_list_log_alphas[model_name]:
        criterion.get_val(
            model, X, y, log_alpha, tol=tol, monitor=monitor_get_val)
        criterion.get_val_grad(
            model, X, y, log_alpha, algo.get_beta_jac_v,
            tol=tol, monitor=monitor_get_val_grad)

    obj_val = np.array(monitor_get_val.objs)
    obj_val_grad = np.array(monitor_get_val_grad.objs)

    np.testing.assert_allclose(obj_val, obj_val_grad)


if __name__ == '__main__':
    # for model_name, criterion in list_model_crit:
    test_cross_val_criterion('logreg', HeldOutLogistic(idx_train, idx_val))
