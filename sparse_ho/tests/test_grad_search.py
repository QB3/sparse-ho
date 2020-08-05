import numpy as np
from scipy.sparse import csc_matrix
import pytest
import sklearn

from sparse_ho.utils import Monitor

from sparse_ho.datasets.synthetic import get_synt_data
from sparse_ho.models import Lasso

from sparse_ho.forward import Forward
from sparse_ho.implicit_forward import ImplicitForward
from sparse_ho.implicit import Implicit
# from sparse_ho.backward import Backward
from sparse_ho.criterion import CV, SURE
from sparse_ho.ho import grad_search


n_samples = 100
n_features = 100
n_active = 5
SNR = 3
rho = 0.5

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
max_iter = 1000

dict_log_alpha0 = {}
dict_log_alpha0["lasso"] = log_alpha
tab = np.linspace(1, 1000, n_features)
dict_log_alpha0["wlasso"] = log_alpha + np.log(tab / tab.max())


estimator = sklearn.linear_model.Lasso(
    fit_intercept=False, max_iter=1000, warm_start=True)

models = [
    Lasso(X_train, y_train, max_iter=max_iter, estimator=estimator),
    # Lasso(X_train_s, y_train, dict_log_alpha["lasso"]),
    # wLasso(X_train, y_train, dict_log_alpha0["wlasso"])
]


@pytest.mark.parametrize('model', models)
@pytest.mark.parametrize('crit', ['cv', 'sure'])
def test_grad_search(model, crit):
    """check that the paths are the same in the line search"""
    if crit == 'cv':
        n_outer = 2
        criterion = CV(X_val, y_val, model, X_test=X_test, y_test=y_test)
    else:
        n_outer = 2
        criterion = SURE(X_train, y_train, model, sigma=sigma_star,
                         X_test=X_test, y_test=y_test)

    # criterion = SURE(
    #     X_train, y_train, model, sigma=sigma_star, X_test=X_test,
    #     y_test=y_test)
    criterion = CV(X_val, y_val, model, X_test=X_test, y_test=y_test)
    monitor1 = Monitor()
    algo = Forward(criterion)
    grad_search(algo, log_alpha, monitor1, n_outer=n_outer,
                tol=1e-16)

    # criterion = SURE(
    #     X_train, y_train, model, sigma=sigma_star, X_test=X_test,
    #     y_test=y_test)
    criterion = CV(X_val, y_val, model, X_test=X_test, y_test=y_test)
    monitor2 = Monitor()
    algo = Implicit(criterion)
    grad_search(algo, log_alpha, monitor2, n_outer=n_outer,
                tol=1e-16)

    # criterion = SURE(
    #     X_train, y_train, model, sigma=sigma_star, X_test=X_test,
    #     y_test=y_test)
    criterion = CV(X_val, y_val, model, X_test=X_test, y_test=y_test)
    monitor3 = Monitor()
    algo = ImplicitForward(criterion, tol_jac=1e-8, n_iter_jac=5000)
    grad_search(algo, log_alpha, monitor3, n_outer=n_outer,
                tol=1e-16)

    # criterion = SURE(
    #     X_train, y_train, model, sigma=sigma_star, X_test=X_test,
    #     y_test=y_test)
    # criterion = CV(X_val, y_val, model, X_test=X_test, y_test=y_test)
    # monitor4 = Monitor()
    # algo = Backward(criterion)
    # grad_search(algo, model.log_alpha, monitor4, n_outer=n_outer,
    #             tol=1e-16)

    assert np.allclose(
        np.array(monitor1.log_alphas), np.array(monitor3.log_alphas))
    assert np.allclose(
        np.array(monitor1.grads), np.array(monitor3.grads))
    assert np.allclose(
        np.array(monitor1.objs), np.array(monitor3.objs))
    assert np.allclose(
        np.array(monitor1.objs_test), np.array(monitor3.objs_test))
    assert not np.allclose(
        np.array(monitor1.times), np.array(monitor3.times))
    # assert np.allclose(
    #     np.array(monitor1.objs), np.array(monitor4.objs))
    # assert np.allclose(
    #     np.array(monitor1.log_alphas), np.array(monitor4.log_alphas),
    #     atol=1e-6)
    # assert np.allclose(
    #     np.array(monitor1.grads), np.array(monitor4.grads), atol=1e-7)

    # assert np.allclose(
    #     np.array(monitor1.objs_test), np.array(monitor4.objs_test))
    # assert not np.allclose(
    #     np.array(monitor1.times), np.array(monitor4.times))


if __name__ == '__main__':
    models = [
        Lasso(X_train, y_train, max_iter=max_iter, estimator=estimator)]
    crits = ['cv']
    # crits = ['cv', 'sure']
    for model in models:
        for crit in crits:
            test_grad_search(model, crit)
