import numpy as np
from scipy.sparse import csc_matrix

from sparse_ho.utils import Monitor

from sparse_ho.datasets.synthetic import get_synt_data
from sparse_ho.models import Lasso, wLasso

# from sparse_ho.forward import Forward
from sparse_ho.implicit_forward import ImplicitForward
# from sparse_ho.implicit import Implicit
# from sparse_ho.backward import Backward
from sparse_ho.criterion import CV
# from sparse_ho.criterion import SURE
# from sparse_ho.ho import grad_search
from sparse_ho.grad_search_CV import grad_search_CV

n_samples = 100
n_features = 100
n_active = 5
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
    # monitor = Monitor()
    # grad_search_CV(
    #     X, y, Lasso, CV, ImplicitForward, log_alpha, monitor, n_outer=5)
    monitor = Monitor()
    grad_search_CV(
        X, y, Lasso, CV, ImplicitForward, log_alpha, monitor, n_outer=15)


if __name__ == '__main__':
    # models = [
    #     Lasso(X_train, y_train, dict_log_alpha["lasso"]),
    #     wLasso(X_train, y_train, dict_log_alpha["wlasso"])]
    # crits = ['cv', 'sure']
    # # for model in models:
    # #     for crit in crits:
    # #         test_grad_search(model, crit)
    # test_grad_search()
    monitor = Monitor()
    grad_search_CV(
        X, y, Lasso, CV, ImplicitForward, log_alpha, monitor, n_outer=15)
