import numpy as np
from numpy.linalg import norm
from scipy.sparse import issparse, identity
from scipy.sparse.linalg import cg
from sparse_ho.utils import init_dbeta0_new
from sparse_ho.forward import get_beta_jac_iterdiff


class Implicit():
    def __init__(self, criterion):
        self.criterion = criterion

    def get_beta_jac_v(
            self, X, y, log_alpha, model, get_v, mask0=None, dense0=None,
            jac0=None, quantity_to_warm_start=None, max_iter=1000, tol=1e-3,
            compute_jac=False, backward=False, full_jac_v=False):

        mask, dense, jac_v, sol_lin_sys = get_beta_jac_t_v_implicit(
            X, y, log_alpha, self.criterion.X_val, self.criterion.y_val, get_v,
            mask0=mask0, dense0=dense0,
            sol_lin_sys=quantity_to_warm_start, tol=tol, model=model)

        if full_jac_v:
            jac_v = model.get_full_jac_v(mask, jac_v, X.shape[1])

        return mask, dense, jac_v, sol_lin_sys

    def get_val_grad(
            self, log_alpha, mask0=None, dense0=None, beta_star=None,
            jac0=None, max_iter=1000, tol=1e-3, compute_jac=True,
            backward=False):
        return self.criterion.get_val_grad(
            log_alpha, self.get_beta_jac_v, max_iter=max_iter, tol=tol,
            compute_jac=compute_jac, backward=backward)


def get_beta_jac_t_v_implicit(
        X_train, y_train, log_alpha, X_val, y_val, get_v,
        mask0=None, dense0=None, tol=1e-3, model="lasso",
        sk=False, max_iter=1000, sol_lin_sys=None, criterion="cv", n=1,
        sigma=0, delta=0, epsilon=0):
    alpha = np.exp(log_alpha)
    n_samples, n_features = X_train.shape

    mask, dense, _ = get_beta_jac_iterdiff(
        X_train, y_train, log_alpha, mask0=mask0, dense0=dense0,
        tol=tol, max_iter=max_iter, compute_jac=False, model=model)
    v = get_v(mask, dense)

    # TODO: to clean
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

    hessian = model.hessian_f(y_train * (X_train[:, mask] @ dense))
    mat_to_inv = X_train[:, mask].T @ np.diag(hessian) @ X_train[:, mask]
    size_mat = mask.sum()

    try:
        sol = cg(
            # this is shady and may lead to errors to multiply by n_samples
            # here
            mat_to_inv, - n_samples * v,
            # x0=sol0, tol=tol, maxiter=1e5)
            x0=sol0, tol=tol)
        if sol[1] == 0:
            jac = sol[0]
        else:
            raise ValueError('cg did not converge.')
            1 / 0
    except Exception:
        print("Matrix to invert was badly conditioned")
        size_mat = mask.sum()
        if is_sparse:
            reg_amount = 1e-7 * norm(X_train[:, mask].todense(), ord=2) ** 2
            mat_to_inv += reg_amount * identity(size_mat)
        else:
            reg_amount = 1e-7 * norm(X_train[:, mask], ord=2) ** 2
            mat_to_inv += reg_amount * np.eye(size_mat)
        sol = cg(
            mat_to_inv + reg_amount * identity(size_mat),
            - n_samples * v, x0=sol0, atol=1e-3)
        jac = sol[0]

    jac_t_v = model._reduce_jac_t_v(jac, mask, dense, alphas)

    return mask, dense, jac_t_v, sol[0]
