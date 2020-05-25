import numpy as np
from scipy.sparse import csc_matrix

from sparse_ho.datasets.synthetic import get_synt_data

from sparse_ho.forward import Forward
from sparse_ho.criterion import CV


n_samples = 100
n_features = 100
n_active = 5
SNR = 3
rho = 0.5

X_train, y_train, beta_star, noise, sigma = get_synt_data(
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

# models = ["lasso", "wlasso"]
# models = {}
# models["lasso"] = Lasso(dict_log_alpha["lasso"])
# models["wlasso"] = wLasso(dict_log_alpha["wlasso"])

# definition of the inner problem
model = Lasso(log_alpha)
# definition of the outer problem
criterion = CV(X_val, y_val, model)
# definition of the solver
algo = Forward(criterion)

val, grad = algo.get_val_grad(
    X_train, y_train, log_alpha)
