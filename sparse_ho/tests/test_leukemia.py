import numpy as np

from sparse_ho.forward import get_beta_jac_iterdiff
from sparse_ho.implicit_forward import get_beta_jac_fast_iterdiff
from sparse_ho.models import Lasso

from sparse_ho.datasets.real import get_leukemia

X_train, X_val, X_test, y_train, y_val, y_test = get_leukemia()
n_samples, n_features = X_train.shape


alpha_max = (X_train.T @ y_train).max() / n_samples
p_alpha = 0.7
alpha = p_alpha * alpha_max
log_alpha = np.log(alpha)
X_train, y_train, X_test, y_test, X_val, y_val
log_alphas = np.log(alpha_max * np.geomspace(1, 0.1))
tol = 1e-10

dict_log_alpha = {}
dict_log_alpha["lasso"] = log_alpha
tab = np.linspace(1, 1000, n_features)
dict_log_alpha["wlasso"] = log_alpha + np.log(tab / tab.max())

models = {}
models["lasso"] = Lasso(X_train, y_train, dict_log_alpha["lasso"])
# models["wlasso"] = wLasso(X_train, y_train, dict_log_alpha["wlasso"])


def get_v(mask, dense):
    return 2 * (X_val[:, mask].T @ (
        X_val[:, mask] @ dense - y_val)) / X_val.shape[0]


def test_beta_jac():
    #########################################################################
    # check that the methods computing the full Jacobian compute the same sol
    # maybe we could add a test comparing with sklearn
    for key in models.keys():
        supp1, dense1, jac1 = get_beta_jac_iterdiff(
            X_train, y_train, dict_log_alpha[key], tol=tol,
            model=models[key])
        supp1sk, dense1sk, jac1sk = get_beta_jac_iterdiff(
            X_train, y_train, dict_log_alpha[key], tol=tol,
            model=models[key], use_sk=True)
        supp2, dense2, jac2 = get_beta_jac_fast_iterdiff(
            X_train, y_train, dict_log_alpha[key], X_test, y_test, get_v,
            tol=tol, model=models[key], tol_jac=tol)

        assert np.all(supp1 == supp1sk)
        assert np.all(supp1 == supp2)
        assert np.allclose(dense1, dense1sk, rtol=1e-4)
        assert np.allclose(dense1, dense2)
        assert np.allclose(jac1, jac2, atol=1e-2)

        # get_beta_jac_t_v_implicit(
        #     X_train, y_train, dict_log_alpha[key], X_test, y_test, get_v,
        #     model=models[key])


if __name__ == '__main__':
    test_beta_jac()
    # test_val_grad()
