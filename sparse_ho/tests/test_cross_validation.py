import numpy as np
import sklearn
from sklearn.model_selection import KFold
from sklearn.linear_model import LassoCV

from sparse_ho.models import Lasso
from sparse_ho.criterion import CrossVal, HeldOutMSE
from sparse_ho.utils import Monitor
from sparse_ho.datasets.synthetic import get_synt_data
from sparse_ho import Forward
from sparse_ho.grid_search import grid_search

n_samples = 100
n_features = 10
n_active = 2
snr = 3
rho = 0.5

X, y = get_synt_data(
    n_samples=n_samples, n_features=n_features, n_times=1, n_active=n_active,
    rho=rho, snr=snr, seed=0)[:2]

alpha_max = (np.abs(X.T @ y)).max() / n_samples
p_alpha = 0.7
alpha = p_alpha * alpha_max
log_alpha = np.log(alpha)

tol = 1e-16

dict_log_alpha = {}
dict_log_alpha["lasso"] = log_alpha
tab = np.linspace(1, 1000, n_features)
dict_log_alpha["wlasso"] = log_alpha + np.log(tab / tab.max())

estimator = sklearn.linear_model.Lasso(
    fit_intercept=False, max_iter=1000, warm_start=True)
model = Lasso(estimator=estimator)


def test_cross_val_criterion():
    # TODO we also need to add a test for sparse matrices
    alpha_min = alpha_max / 10
    log_alpha_max = np.log(alpha_max)
    log_alpha_min = np.log(alpha_min)
    max_iter = 10000
    n_alphas = 10
    kf = KFold(n_splits=5, shuffle=True, random_state=56)

    monitor_grid = Monitor()
    mse = HeldOutMSE(None, None)
    criterion = CrossVal(mse, cv=kf)
    algo = Forward()
    grid_search(
        algo, criterion, model, X, y, log_alpha_min, log_alpha_max,
        monitor_grid, max_evals=n_alphas)

    reg = LassoCV(
        cv=kf, verbose=True, tol=tol, fit_intercept=False,
        alphas=np.geomspace(alpha_max, alpha_min, num=n_alphas),
        max_iter=max_iter).fit(X, y)
    reg.score(X, y)
    objs_grid_sk = reg.mse_path_.mean(axis=1)
    np.testing.assert_allclose(objs_grid_sk, monitor_grid.objs, rtol=1e-6)


if __name__ == '__main__':
    test_cross_val_criterion()
