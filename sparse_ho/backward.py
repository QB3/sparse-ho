import numpy as np
from numpy.linalg import norm
from scipy.sparse import issparse
import scipy.sparse.linalg as slinalg
from numba import njit

from sparse_ho.forward import get_beta_jac_iterdiff


def get_beta_jac_backward(
        X_train, y_train, log_alpha, X_val, y_val, mask0=None,
        dense0=None, jac0=None,
        tol=1e-3, maxit=100, model="lasso"):
    n_samples, n_features = X_train.shape

    mask, dense, list_sign = get_beta_jac_iterdiff(
        X_train, y_train, log_alpha, mask0=mask0, dense0=dense0,
        jac0=jac0, tol=tol,
        maxit=maxit, compute_jac=False, model="lasso", backward=True)

    v = np.zeros(n_features)
    v[mask] = 2 * X_val[:, mask].T @ (
        X_val[:, mask] @ dense - y_val) / X_val.shape[0]

    jac = get_only_jac_backward(
        X_train, np.exp(log_alpha), list_sign, v)
    return mask, dense, jac


def get_only_jac_backward(X_train, alpha, list_sign, v):
    n_samples, n_features = X_train.shape
    is_sparse = issparse(X_train)
    if is_sparse:
        L = slinalg.norm(X_train, axis=0) ** 2 / n_samples
    else:
        L = norm(X_train, axis=0) ** 2 / n_samples
    v_t_jac = v.copy()
    list_sign = np.asarray(list_sign)
    grad = 0.0
    for k in (np.arange(list_sign.shape[0] - 1, -1, -1)):
        sign_beta = list_sign[k, :]
        grad = _update_bcd_jac_backward(
            X_train, alpha, grad, sign_beta, v_t_jac, L)
    return grad


@njit
def _update_bcd_jac_backward(X_train, alpha, grad, sign_beta, v_t_jac, L):
    n_samples, n_features = X_train.shape

    for j in (np.arange(sign_beta.shape[0] - 1, -1, -1)):
        grad -= (v_t_jac[j]) * alpha * sign_beta[j] / L[j]
        v_t_jac[j] *= np.abs(sign_beta[j])
        v_t_jac -= v_t_jac[j] / (L[j] * n_samples) * X_train[:, j] @ X_train
    # for j in range(n_features):
    #     grad -= alpha * sign_beta[j] * v_t_jac[j] / L[j]
    #     v_t_jac[j] *= np.abs(sign_beta[j])
    #     v_t_jac -= v_t_jac[j] * X_train[:, j].T @ X_train / (
    #         L[j] * n_samples)
    return grad
