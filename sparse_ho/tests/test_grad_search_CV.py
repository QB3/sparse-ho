import numpy as np
from scipy.sparse import csc_matrix
from sklearn.model_selection import KFold
from sklearn.linear_model import LassoCV

from sparse_ho.models import Lasso, wLasso
from sparse_ho.criterion import CV, CrossVal
from sparse_ho.utils import Monitor
from sparse_ho.datasets.synthetic import get_synt_data

from sparse_ho.forward import Forward
from sparse_ho.implicit_forward import ImplicitForward
from sparse_ho.grid_search import grid_search
# from sparse_ho.ho import grad_search
from sparse_ho.grad_search_CV import grad_search_CV

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
    Lasso(X_train, y_train, dict_log_alpha["lasso"]),
    wLasso(X_train, y_train, dict_log_alpha["wlasso"])
]

# @pytest.mark.parametrize('model', models)
# @pytest.mark.parametrize('crit', ['cv', 'sure'])


def test_grad_search():
    monitor = Monitor()
    grad_search_CV(
        X, y, Lasso, CV, ImplicitForward, log_alpha, monitor, n_outer=15)


def test_cross_val_criterion():
    alpha_min = alpha_max / 100
    log_alpha_max = np.log(alpha_max)
    log_alpha_min = np.log(alpha_min)
    # log_alpha0 = np.log(alpha_max / 10)
    max_iter = 10000
    n_alphas = 10
    kf = KFold(n_splits=2, shuffle=True, random_state=42)
    # kf = KFold(n_splits=5, shuffle=True)

    # monitor_grad = Monitor()
    # criterion = CrossVal(X, y, Lasso, cv=kf)
    # algo = ImplicitForward(criterion, use_sk=True, max_iter=max_iter)
    # grad_search(
    #     algo, log_alpha0, monitor_grad, n_outer=3, tol=tol)

    monitor_grid = Monitor()
    criterion = CrossVal(X, y, Lasso, cv=kf)
    algo = Forward(criterion, use_sk=True)
    grid_search(
        algo, log_alpha_min, log_alpha_max, monitor_grid, max_evals=n_alphas)

    reg = LassoCV(
        cv=kf, verbose=True, tol=tol, fit_intercept=False,
        alphas=np.geomspace(alpha_max, alpha_min, num=n_alphas),
        max_iter=max_iter).fit(X, y)
    reg.score(X, y)
    objs_grid_sk = reg.mse_path_.mean(axis=1)
    # these 2 value should be the same or I did not understand smth
    assert np.allclose(objs_grid_sk, monitor_grid.objs)


if __name__ == '__main__':
    test_grad_search()
    test_cross_val_criterion()
