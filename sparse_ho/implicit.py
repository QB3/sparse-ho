import numpy as np
from numpy.linalg import norm
from scipy.sparse import issparse, identity
from scipy.sparse.linalg import cg
from sparse_ho.utils import init_dbeta0_new
from sparse_ho.forward import get_beta_jac_iterdiff


class Implicit():
    """Algorithm that will compute the (hyper)gradient, ie the gradient with respect to the hyperparameter using the implicit differentiation.

    Parameters
    ----------
    criterion: criterion object
        HeldOut, CrossVal or SURE
        max_iter: int
            maximum number of iteration for the inner solver
    """
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

    mat_to_inv = model.get_hessian(mask, dense, log_alpha)
    size_mat = mat_to_inv.shape[0]

    maskp, densep = model.get_primal(mask, dense)
    v = get_v(maskp, densep)

    # TODO: to clean
    is_sparse = issparse(X_train)
    if not alpha.shape:
        alphas = np.ones(n_features) * alpha
    else:
        alphas = alpha.copy()

    if sol_lin_sys is not None:
        sol0 = init_dbeta0_new(sol_lin_sys, mask, mask0)
    else:
        size_mat = mat_to_inv.shape[0]
        sol0 = np.zeros(size_mat)
    try:
        sol = cg(
            mat_to_inv, - model.restrict_full_supp(mask, dense, v),
            # x0=sol0, tol=tol, maxiter=1e5)
            x0=sol0, tol=tol)
        if sol[1] == 0:
            sol_lin_sys = sol[0]
        else:
            raise ValueError('cg did not converge.')
    except Exception:
        print("Matrix to invert was badly conditioned")
        size_mat = mat_to_inv.shape[0]
        if is_sparse:
            reg_amount = 1e-7 * norm(model.reduce_X(mask).todense(), ord=2) ** 2
            mat_to_inv += reg_amount * identity(size_mat)
        else:
            reg_amount = 1e-7 * norm(model.reduce_X(mask), ord=2) ** 2
            mat_to_inv += reg_amount * np.eye(size_mat)
        sol = cg(
            mat_to_inv + reg_amount * identity(size_mat),
            - model.restrict_full_supp(mask, dense, v), x0=sol0, atol=1e-3)
        sol_lin_sys = sol[0]
    jac_t_v = model._get_jac_t_v(sol_lin_sys, mask, dense, alphas, v.copy())

    return mask, dense, jac_t_v, sol[0]
