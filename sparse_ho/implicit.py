import numpy as np
from numpy.linalg import norm
from sklearn.linear_model import Lasso
from scipy.linalg import solve
from scipy.sparse import issparse, identity
from scipy.sparse.linalg import cg
from sparse_ho.utils import init_dbeta0_new
from sparse_ho.forward import get_beta_jac_iterdiff


def get_beta_jac_t_v_implicit(
        X_train, y_train, log_alpha, X_val, y_val,
        mask0=None, dense0=None, jac0=None, tol=1e-3, model="lasso",
        sk=False, maxit=1000, sol_lin_sys=None, criterion="cv", n=1,
        sigma=0, delta=0, epsilon=0):
    alpha = np.exp(log_alpha)
    n_samples, n_features = X_train.shape
    # compute beta using sklearn lasso
    if sk:
        clf = Lasso(
            alpha=alpha, fit_intercept=False, warm_start=True, tol=tol,
            max_iter=10000)
        clf.fit(X_train, y_train)
        coef_ = clf.coef_
        mask = coef_ != 0
        dense = coef_[mask]
    # compute beta using vanilla numba cd lasso
    else:
        mask, dense = get_beta_jac_iterdiff(
            X_train, y_train, log_alpha, mask0=mask0, dense0=dense0,
            maxit=maxit, tol=tol,
            compute_jac=False, jac0=None)

    # v = 2 * X_val[:, mask].T @ (
    #     X_val[:, mask] @ dense - y_val) / X_val.shape[0]

    if criterion == "cv":
        v = 2 * X_val[:, mask].T @ (
            X_val[:, mask] @ dense - y_val) / X_val.shape[0]
    elif criterion == "sure":
        if n == 1:
            v = 2 * X_train[:, mask].T @ (
                X_train[:, mask] @dense -
                y_train - 2 * sigma ** 2 / epsilon * delta)
        elif n == 2:
            v = 2 * sigma ** 2 * X_train[:, mask].T @ delta / epsilon

    is_sparse = issparse(X_train)

    if not alpha.shape:
        alphas = np.ones(n_features) * alpha
    else:
        alphas = alpha.copy()

    if sol_lin_sys is not None:
        sol0 = init_dbeta0_new(sol_lin_sys, mask, mask0)
    else:
        size_mat = mask.sum()
        sol0 = np.zeros(size_mat)

    mat_to_inv = X_train[:, mask].T @ X_train[:, mask]
    size_mat = mask.sum()

    if is_sparse:
        try:
            # reg_amount = 1e-7 * norm(X_train[:, mask].todense(), ord=2) ** 2
            # mat_to_inv += reg_amount * identity(size_mat)
            sol = cg(
                mat_to_inv, - n_samples * v,
                x0=sol0, tol=1e-15, maxiter=1e5)
            # sol = cg(
            #     mat_to_inv, - alpha * n_samples * v,
            #     x0=sol0, atol=1e-3)
            if sol[1] == 0:
                jac = sol[0]
            else:
                raise ValueError('cg did not converge.')
        except Exception:
            print("Matrix to invert was badly conditioned")
            size_mat = mask.sum()
            reg_amount = 1e-7 * norm(X_train[:, mask].todense(), ord=2) ** 2
            sol = cg(
                mat_to_inv + reg_amount * identity(size_mat),
                - n_samples * v, x0=sol0,
                # - alpha * n_samples * v, x0=sol0,
                atol=1e-3)
            jac = sol[0]
    else:
        try:
            jac = solve(
                X_train[:, mask].T @ X_train[:, mask],
                - n_samples * v,
                sym_pos=True, assume_a='pos')
            # import ipdb; ipdb.set_trace()
        except Exception:
            print("Matrix to invert was badly conditioned")
            size_mat = mask.sum()
            reg_amount = 1e-9 * norm(X_train[:, mask], ord=2) ** 2
            jac = solve(
                X_train[:, mask].T @ X_train[:, mask] +
                reg_amount * np.eye(size_mat),
                - n_samples * v,
                sym_pos=True, assume_a='pos')

    if model == "lasso":
        jac_t_v = alpha * np.sign(dense) @ jac
    elif model == "wlasso":
        jac_t_v = np.zeros(n_features)
        jac_t_v[mask] = alphas[mask] * np.sign(dense) * jac

    return mask, dense, jac_t_v, jac
