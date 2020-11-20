import numpy as np
from scipy.sparse import csc_matrix
import sklearn
from sklearn.model_selection import KFold
from sklearn.linear_model import LassoCV

from sparse_ho.models import LassoGradSearch
from sparse_ho.criterion import CrossVal
from sparse_ho.utils import Monitor
from sparse_ho.datasets.synthetic import get_synt_data

from sparse_ho.forward import Forward
from sparse_ho.grid_search import grid_search

n_samples = 10
n_features = 10
n_active = 2
SNR = 3
rho = 0.5

X, y, beta_star, noise, sigma_star = get_synt_data(
    dictionary_type="Toeplitz", n_samples=n_samples,
    n_features=n_features, n_times=1, n_active=n_active, rho=rho,
    SNR=SNR, seed=0)

X_train, y_train, beta_star, noise, sigma_star = get_synt_data(
    dictionary_type="Toeplitz", n_samples=n_samples,
    n_features=n_features, n_times=1, n_active=n_active, rho=rho,
    SNR=SNR, seed=0)
X_train_s = csc_matrix(X_train)

X_test, y_test, beta_star, noise, sigma = get_synt_data(
    dictionary_type="Toeplitz", n_samples=n_samples,
    n_features=n_features, n_times=1, n_active=n_active, rho=rho,
    SNR=SNR, seed=1)
X_test_s = csc_matrix(X_test)

X_val, y_val, beta_star, noise, sigma = get_synt_data(
    dictionary_type="Toeplitz", n_samples=n_samples,
    n_features=n_features, n_times=1, n_active=n_active, rho=rho,
    SNR=SNR, seed=2)
X_test_s = csc_matrix(X_test)


alpha_max = (X_train.T @ y_train).max() / n_samples
p_alpha = 0.7
alpha = p_alpha * alpha_max
log_alpha = np.log(alpha)

log_alphas = np.log(alpha_max * np.geomspace(1, 0.1))
tol = 1e-16

dict_log_alpha = {}
dict_log_alpha["lasso"] = log_alpha
tab = np.linspace(1, 1000, n_features)
dict_log_alpha["wlasso"] = log_alpha + np.log(tab / tab.max())

models = [
    LassoGradSearch(X_train, y_train, dict_log_alpha["lasso"])
]


def test_cross_val_criterion():
    alpha_min = alpha_max / 10
    log_alpha_max = np.log(alpha_max)
    log_alpha_min = np.log(alpha_min)
    max_iter = 10000
    n_alphas = 10
    kf = KFold(n_splits=5, shuffle=True, random_state=56)

    estimator = sklearn.linear_model.Lasso(
        fit_intercept=False, max_iter=1000, warm_start=True)
    monitor_grid = Monitor()
    criterion = CrossVal(X, y, Lasso, cv=kf, estimator=estimator)
    algo = Forward(criterion)
    grid_search(
        algo, log_alpha_min, log_alpha_max, monitor_grid, max_evals=n_alphas,
        tol=tol)

    reg = LassoCV(
        cv=kf, verbose=True, tol=tol, fit_intercept=False,
        alphas=np.geomspace(alpha_max, alpha_min, num=n_alphas),
        max_iter=max_iter).fit(X, y)
    reg.score(X, y)
    objs_grid_sk = reg.mse_path_.mean(axis=1)
    # these 2 value should be the same
    (objs_grid_sk - np.array(monitor_grid.objs))
    assert np.allclose(objs_grid_sk, monitor_grid.objs)


if __name__ == '__main__':
    test_cross_val_criterion()
