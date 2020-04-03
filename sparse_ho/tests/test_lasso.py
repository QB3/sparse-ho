import numpy as np
from scipy.sparse import csc_matrix

from my_data.synthetic import get_synt_data
from sparse_ho.forward import get_beta_jac_iterdiff
from sparse_ho.implicit_forward import get_beta_jac_fast_iterdiff
from sparse_ho.ho import get_val_grad, grad_search
from sparse_ho.grid_search import grid_searchCV
from sparse_ho.utils import Monitor, WarmStart


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
tol = 1e-7
log_alphas = np.log(alpha_max * np.geomspace(1, 0.1))
tol = 1e-16

dict_log_alpha = {}
dict_log_alpha["lasso"] = log_alpha
tab = np.linspace(1, 1000, n_features)
dict_log_alpha["wlasso"] = log_alpha + np.log(tab / tab.max())

models = ["lasso", "wlasso"]

def test_beta_jac():
    #########################################################################
    # check that the methods computing the full Jacobian compute the same sol
    # maybe we could add a test comparing with sklearn
    for model in models:
        supp1, dense1, jac1 = get_beta_jac_iterdiff(
            X_train, y_train, dict_log_alpha[model], tol=tol, model=model)
        supp2, dense2, jac2 = get_beta_jac_fast_iterdiff(
            X_train, y_train, dict_log_alpha[model], X_test, y_test,
            tol=tol, model=model, tol_jac=tol)
        supp3, dense3, jac3 = get_beta_jac_iterdiff(
            X_train_s, y_train, dict_log_alpha[model], tol=tol, model=model)
        supp4, dense4, jac4 = get_beta_jac_fast_iterdiff(
            X_train_s, y_train, dict_log_alpha[model], X_test, y_test,
            tol=tol, model=model, tol_jac=tol)

        assert np.all(supp1 == supp2)
        assert np.allclose(dense1, dense2)
        assert np.allclose(jac1, jac2)

        assert np.all(supp2 == supp3)
        assert np.allclose(dense2, dense3)
        assert np.allclose(jac2, jac3)

        assert np.all(supp3 == supp4)
        assert np.allclose(dense3, dense4)
        assert np.allclose(jac3, jac4)


def test_val_grad():
        #######################################################################
        # Not all methods computes the full Jacobian, but all
        # compute the gradients
        # check that the gradient returned by all methods are the same
    for model in models:
        monitor = Monitor()
        warm_start = WarmStart()
        value1, gradient1 = get_val_grad(
                X_train, y_train, dict_log_alpha[model], X_val, y_val, X_test,
                y_test, tol, monitor, warm_start, method="implicit",
                model=model)

        monitor = Monitor()
        warm_start = WarmStart()
        value2, gradient2 = get_val_grad(
                X_train, y_train, dict_log_alpha[model], X_val, y_val, X_test,
                y_test, tol, monitor, warm_start, method="implicit_forward",
                tol_jac=tol, model=model)

        monitor = Monitor()
        warm_start = WarmStart()
        value3, gradient3 = get_val_grad(
                X_train, y_train, dict_log_alpha[model], X_val, y_val, X_test,
                y_test, tol, monitor, warm_start, method="forward",
                model=model)

        monitor = Monitor()
        warm_start = WarmStart()
        value4, gradient4 = get_val_grad(
                X_train_s, y_train, dict_log_alpha[model], X_val, y_val, X_test,
                y_test, tol, monitor, warm_start, method="implicit",
                model=model)

        monitor = Monitor()
        warm_start = WarmStart()
        value5, gradient5 = get_val_grad(
                X_train_s, y_train, dict_log_alpha[model], X_val, y_val, X_test,
                y_test, tol, monitor, warm_start, method="forward",
                model=model)

        monitor = Monitor()
        warm_start = WarmStart()
        value6, gradient6 = get_val_grad(
                X_train_s, y_train, dict_log_alpha[model], X_val, y_val, X_test,
                y_test, tol, monitor, warm_start, method="implicit_forward",
                model=model, tol_jac=tol)

        # monitor = Monitor()
        # warm_start = WarmStart()
        # value3, gradient3 = get_val_grad(
        #         X_train, y_train, dict_log_alpha[model], X_val, y_val, X_test,
        #         y_test, tol, monitor, warm_start, method="forward",
        #         model=model)

        assert np.allclose(value1, value2)
        assert np.allclose(gradient1, gradient2)
        assert np.allclose(value2, value3)
        assert np.allclose(gradient2, gradient3)
        assert np.allclose(value3, value4)
        assert np.allclose(gradient3, gradient4)
        assert np.allclose(value4, value5)
        assert np.allclose(gradient4, gradient5)
        assert np.allclose(value5, value6)
        assert np.allclose(gradient5, gradient6)

        if model == "lasso":
            # check that the backward passes the tests
            monitor = Monitor()
            warm_start = WarmStart()
            value4, gradient4 = get_val_grad(
                    X_train, y_train, log_alpha, X_val, y_val, X_test, y_test, tol,
                    monitor, warm_start, method="backward")

            assert np.allclose(value3, value4)
            assert np.allclose(gradient3, gradient4)


def test_grad_search():
    for model in models:
    #########################################################################
    # check that the paths are the same in the line search
        n_outer = 5
        monitor1 = Monitor()
        warm_start = WarmStart()
        grad_search(
            X_train, y_train, dict_log_alpha[model], X_val, y_val,
            X_test, y_test, tol, monitor1, method="forward", maxit=10000,
            n_outer=n_outer, warm_start=warm_start, niter_jac=10000,
            tol_jac=tol, model=model)

        monitor2 = Monitor()
        warm_start = WarmStart()
        grad_search(
            X_train, y_train, dict_log_alpha[model], X_val, y_val,
            X_test, y_test,
            tol, monitor2, method="implicit", maxit=10000,
            n_outer=n_outer,
            warm_start=warm_start, niter_jac=10000, tol_jac=tol, model=model)

        monitor3 = Monitor()
        warm_start = WarmStart()
        grad_search(
            X_train, y_train, dict_log_alpha[model], X_val, y_val, X_test,
            y_test,
            tol, monitor3, method="implicit_forward", maxit=1000,
            n_outer=n_outer,
            warm_start=warm_start, niter_jac=10000, tol_jac=tol, model=model)

        monitor4 = Monitor()
        warm_start = WarmStart()
        grad_search(
            X_train_s, y_train, dict_log_alpha[model], X_val, y_val,
            X_test, y_test, tol, monitor4, method="forward", maxit=10000,
            n_outer=n_outer, warm_start=warm_start, niter_jac=10000,
            tol_jac=tol, model=model)

        # need to regularize the solution in this case:
        # this means tha the solutions returned are different
        # monitor5 = Monitor()
        # warm_start = WarmStart()
        # grad_search(
        #     X_train_s, y_train, dict_log_alpha[model], X_val, y_val,
        #     X_test, y_test,
        #     tol, monitor5, method="implicit", maxit=10000,
        #     n_outer=n_outer,
        #     warm_start=warm_start, niter_jac=10000, tol_jac=tol, model=model)

        monitor5 = Monitor()
        warm_start = WarmStart()
        grad_search(
            X_train_s, y_train, dict_log_alpha[model], X_val, y_val, X_test,
            y_test,
            tol, monitor5, method="implicit_forward", maxit=1000,
            n_outer=n_outer,
            warm_start=warm_start, niter_jac=10000, tol_jac=tol, model=model)


        assert np.allclose(
            np.array(monitor1.log_alphas), np.array(monitor2.log_alphas))
        assert np.allclose(
            np.array(monitor1.grads), np.array(monitor2.grads))
        assert np.allclose(
            np.array(monitor1.objs), np.array(monitor2.objs))
        assert np.allclose(
            np.array(monitor1.objs_test), np.array(monitor2.objs_test))
        assert not np.allclose(
            np.array(monitor1.times), np.array(monitor2.times))

        assert np.allclose(
            np.array(monitor2.log_alphas), np.array(monitor3.log_alphas))
        assert np.allclose(
            np.array(monitor2.grads), np.array(monitor3.grads))
        assert np.allclose(
            np.array(monitor2.objs), np.array(monitor3.objs))
        assert np.allclose(
            np.array(monitor2.objs_test), np.array(monitor3.objs_test))
        assert not np.allclose(
            np.array(monitor2.times), np.array(monitor3.times))

        assert np.allclose(
            np.array(monitor3.log_alphas), np.array(monitor4.log_alphas))
        assert np.allclose(
            np.array(monitor3.grads), np.array(monitor4.grads))
        assert np.allclose(
            np.array(monitor3.objs), np.array(monitor4.objs))
        assert np.allclose(
            np.array(monitor3.objs_test), np.array(monitor4.objs_test))
        assert not np.allclose(
            np.array(monitor3.times), np.array(monitor4.times))

        assert np.allclose(
            np.array(monitor4.log_alphas), np.array(monitor5.log_alphas))
        assert np.allclose(
            np.array(monitor4.grads), np.array(monitor5.grads))
        assert np.allclose(
            np.array(monitor4.objs), np.array(monitor5.objs))
        assert np.allclose(
            np.array(monitor4.objs_test), np.array(monitor5.objs_test))
        assert not np.allclose(
            np.array(monitor4.times), np.array(monitor5.times))


def test_grid_search():
    monitor = Monitor()
    grid_searchCV(
        X_train, y_train, log_alphas, X_test, y_test, X_test, y_test,
        tol, monitor, sk=False)
    monitor_sparse = Monitor()
    grid_searchCV(
        X_train_s, y_train, log_alphas, X_test_s, y_test, X_test_s, y_test, tol,
        monitor_sparse, sk=False)

    assert np.allclose(monitor.objs, monitor_sparse.objs)


if __name__ == '__main__':
    test_beta_jac()
    test_val_grad()
    test_grad_search()
    test_grid_search()
