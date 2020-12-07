import numpy as np
from numpy.linalg import norm
from scipy.sparse import issparse
import scipy.sparse.linalg as slinalg
from sparse_ho.algo.forward import get_beta_jac_iterdiff


class Backward():
    """Algorithm that will compute the (hyper)gradient, ie the gradient with
    respect to the hyperparameter using the backward differentiation.

    Parameters
    ----------
    verbose: bool
    """

    def __init__(self, verbose=False):
        self.verbose = verbose

    def get_beta_jac_v(
            self, X, y, log_alpha, model, get_v, mask0=None, dense0=None,
            quantity_to_warm_start=None, max_iter=1000, tol=1e-3,
            compute_jac=False, full_jac_v=False):
        mask, dense, list_sign = get_beta_jac_iterdiff(
            X, y, log_alpha, model, mask0=mask0, dense0=dense0,
            jac0=None, max_iter=max_iter, tol=tol,
            compute_jac=compute_jac, return_all=True)
        v = np.zeros(X.shape[1])
        v[mask] = get_v(mask, dense)
        jac_v = get_only_jac_backward(
            X, np.exp(log_alpha), list_sign, v, model,
            jac_v0=quantity_to_warm_start)

        if not full_jac_v:
            jac_v = model.get_mask_jac_v(mask, jac_v)
        return mask, dense, jac_v, jac_v


def get_only_jac_backward(X, alpha, list_beta, v, model, jac_v0=None):
    n_samples, n_features = X.shape
    is_sparse = issparse(X)
    if is_sparse:
        L = slinalg.norm(X, axis=0) ** 2 / n_samples
    else:
        L = norm(X, axis=0) ** 2 / n_samples
    v_ = v.copy()
    list_beta = np.asarray(list_beta)
    jac_t_v = model._init_g_backward(None, n_features)
    for k in (np.arange(list_beta.shape[0] - 1, -1, -1)):
        beta = list_beta[k, :]
        if is_sparse:
            jac_t_v = model._update_bcd_jac_backward_sparse(
                X.data, X.indptr, X.indices, n_samples, n_features,
                alpha, jac_t_v, beta, v_, L)
        else:
            jac_t_v = model._update_bcd_jac_backward(
                X, alpha, jac_t_v, beta, v_, L)

    return jac_t_v
