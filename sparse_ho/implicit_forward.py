import numpy as np
from numpy.linalg import norm
from scipy.sparse import issparse
import scipy.sparse.linalg as slinalg
from numba import njit

from sparse_ho.utils import init_dbeta0_new, init_dbeta0_new_p
from sparse_ho.forward import get_beta_jac_iterdiff


def get_beta_jac_fast_iterdiff(
        X, y, log_alpha, X_val, y_val, mask0=None, dense0=None, jac0=None,
        tol=1e-3, maxit=100, niter_jac=1000, tol_jac=1e-6, model="lasso",
        criterion="cv", sigma=1, epsilon=0.1, delta=None, n=1):
    n_samples, n_features = X.shape

    if model == "mcp":
        mask, dense = get_beta_jac_iterdiff(
            X, y, log_alpha, mask0=mask0, dense0=dense0, jac0=jac0, tol=tol,
            maxit=maxit, compute_jac=False, model="mcp")
    else:
        mask, dense = get_beta_jac_iterdiff(
            X, y, log_alpha, mask0=mask0, dense0=dense0, jac0=jac0, tol=tol,
            maxit=maxit, compute_jac=False, model="lasso")

    # TODO this is dirty, to improve and to jit
    size_mat = mask.sum()
    if model == "lasso":
        if jac0 is not None:
            dbeta0_new = init_dbeta0_new(jac0, mask, mask0)
        else:
            dbeta0_new = np.zeros(size_mat)
    elif model == "mcp":
        # TODO add warm start
        if jac0 is None:
            dbeta0_new = np.zeros((size_mat, 2))
        else:
            dbeta0_new = init_dbeta0_mcp(jac0, mask, mask0)
    else:
        if jac0 is None:
            dbeta0_new = np.zeros((size_mat, size_mat))
        else:
            dbeta0_new = init_dbeta0_new_p(jac0, mask, mask0)

    if criterion == "cv":
        v = 2 * X_val[:, mask].T @ (
            X_val[:, mask] @ dense - y_val) / X_val.shape[0]
    elif criterion == "sure":
        if n == 1:
            v = 2 * X[:, mask].T @ (
                X[:, mask] @dense - y - 2 * sigma ** 2 / epsilon * delta)
        elif n == 2:
            v = 2 * sigma ** 2 * X[:, mask].T @ delta / epsilon
    jac = get_only_jac(
        X[:, mask], np.exp(log_alpha), np.sign(dense), v,
        dbeta=dbeta0_new, niter_jac=niter_jac, tol_jac=tol_jac, model=model,
        mask=mask, dense=dense)

    return mask, dense, jac


def init_dbeta0_mcp(jac0, mask, mask0):
    # mask_both = np.logical_and(mask_old, mask)
    size_mat = mask.sum()
    dbeta0_new = np.zeros((size_mat, 2))
    # count = 0
    # count_old = 0
    # n_features = mask.shape[0]
    for j in range(2):
        dbeta0_new[:, j] = init_dbeta0_new(jac0[:, j], mask, mask0)
    return dbeta0_new


def get_only_jac(
        Xs, alpha, sign_beta, v, dbeta=None, niter_jac=100, tol_jac=1e-4,
        model="lasso", mask=None, dense=None):
    n_samples, n_features = Xs.shape

    is_sparse = issparse(Xs)
    if is_sparse:
        L = slinalg.norm(Xs, axis=0) ** 2 / n_samples
    else:
        L = norm(Xs, axis=0) ** 2 / n_samples

    if dbeta is None:
        if model == "lasso":
            dbeta = np.zeros(n_features)
        if model == "mcp":
            dbeta = np.zeros((n_features, 2))
        elif model == "wlasso":
            dbeta = np.zeros((n_features, n_features))
    else:
        dbeta = dbeta.copy()

    dbeta_old = dbeta.copy()

    is_sparse = issparse(Xs)

    tol_crit = tol_jac * norm(v)
    dr = - Xs @ dbeta
    for i in range(niter_jac):
        print("%i -st iterations over %i" % (i, niter_jac))
        if model == "lasso":
            if is_sparse:
                _update_only_jac_sparse(
                    Xs.data, Xs.indptr, Xs.indices, n_samples,
                    n_features, dbeta, dr, L, alpha, sign_beta)
            else:
                _update_only_jac(Xs, dbeta, dr, L, alpha, sign_beta)
        elif model == "mcp":
            if is_sparse:
                _update_only_jac_mcp_sparse(
                    Xs.data, Xs.indptr, Xs.indices, n_samples, n_features,
                    dense, dbeta, dr, alpha[0], alpha[1], L, compute_jac=True)
            else:
                _update_only_jac_mcp(
                    Xs, dense, dbeta, dr, alpha[0], alpha[1],
                    L, compute_jac=True)
        elif model == "wlasso":
            if is_sparse:
                _update_only_jac_sparse_p(
                    Xs.data, Xs.indptr, Xs.indices, n_samples,
                    n_features, dbeta, dr, L, alpha[mask], sign_beta)
                    # n_features, dbeta, dr, L, alpha, sign_beta)
            else:
                _update_only_jac_p(Xs, dbeta, dr, L, alpha[mask], sign_beta)
        print(norm(dbeta - dbeta_old))
        if norm(v @ (dbeta - dbeta_old)) < tol_crit:
            break
        dbeta_old = dbeta.copy()
    return dbeta


@njit
def _update_only_jac_mcp_sparse(
        data, indptr, indices, n_samples, n_features, beta, dbeta, dr, alpha,
        gamma, L, compute_jac=True):
    non_zeros = np.where(L != 0)[0]

    for j in non_zeros:
        # get the j-st column of X in sparse format
        Xjs = data[indptr[j]:indptr[j+1]]
        # get the non zero idices
        idx_nz = indices[indptr[j]:indptr[j+1]]
        # store old beta j for fast update
        dbeta_old = dbeta[j, :].copy()
        dzj = dbeta[j] + Xjs @ dr[idx_nz] / (L[j] * n_samples)
        # TODO compute mcp_dx mcp_dalpha mcp_dgamma before this loop
        dbeta[j:j+1, :] = mcp_dx(beta[j], alpha, gamma) * dzj
        dbeta[j:j+1, 0] += alpha / L[j] * mcp_dalpha(
            beta[j], alpha / L[j], gamma * L[j])
        dbeta[j:j+1, 1] += gamma * L[j] * mcp_dgamma(
            beta[j], alpha / L[j], gamma * L[j])

        dr[idx_nz] -= np.outer(Xjs, (dbeta[j, :] - dbeta_old))


@njit
def _update_only_jac_mcp(
        X, beta, dbeta, dr, alpha, gamma, L, compute_jac=True):
    n_samples, n_features = X.shape
    non_zeros = np.where(L != 0)[0]

    for j in non_zeros:
        dbeta_old = dbeta[j, :]
        dzj = dbeta[j] + X[:, j] @ dr / (L[j] * n_samples)
        # TODO compute mcp_dx mcp_dalpha mcp_dgamma before this loop
        dbeta[j:j+1, :] = mcp_dx(beta[j], alpha, gamma) * dzj
        dbeta[j:j+1, 0] += alpha / L[j] * mcp_dalpha(
            beta[j], alpha / L[j], gamma * L[j])
        dbeta[j:j+1, 1] += gamma * L[j] * mcp_dgamma(
            beta[j], alpha / L[j], gamma * L[j])

        dr -= np.outer(X[:, j], (dbeta[j, :] - dbeta_old))


@njit
def _update_only_jac_sparse(
        data, indptr, indices, n_samples, n_features,
        dbeta, dr, L, alpha, sign_beta):
    for j in range(n_features):
        # get the j-st column of X in sparse format
        Xjs = data[indptr[j]:indptr[j+1]]
        # get the non zero idices
        idx_nz = indices[indptr[j]:indptr[j+1]]
        # store old beta j for fast update
        dbeta_old = dbeta[j]
        # update of the Jacobian dbeta
        dbeta[j] += Xjs @ dr[idx_nz] / (L[j] * n_samples)
        dbeta[j] -= alpha * sign_beta[j] / L[j]
        dr[idx_nz] -= Xjs * (dbeta[j] - dbeta_old)


@njit
def _update_only_jac_sparse_p(
        data, indptr, indices, n_samples, n_features, dbeta, dr, L,
        alpha, sign_beta):
    for j in range(n_features):
        # get the j-st column of X in sparse format
        Xjs = data[indptr[j]:indptr[j+1]]
        # get the non zero idices
        idx_nz = indices[indptr[j]:indptr[j+1]]
        # store old beta j for fast update
        dbeta_old = dbeta[j, :].copy()

        dbeta[j:j+1, :] += Xjs.T @ dr[idx_nz] / (L[j] * n_samples)
        dbeta[j, j] -= alpha[j] * sign_beta[j] / L[j]
        dr[idx_nz, :] -= np.outer(Xjs, (dbeta[j] - dbeta_old))


@njit
def _update_only_jac_p(Xs, dbeta, dr, L, alpha, sign_beta):
    n_samples, n_features = Xs.shape
    for j in range(n_features):
        dbeta_old = dbeta[j, :].copy()
        dbeta[j:j+1, :] = dbeta[j, :] + Xs[:, j] @ dr / (L[j] * n_samples)
        dbeta[j:j+1, j] -= alpha[j] * sign_beta[j] / L[j]
        # update residuals
        dr -= np.outer(Xs[:, j], (dbeta[j, :] - dbeta_old))
        # dbeta[j:j+1, :] += Xs[:, j].T @ dr / (L[j] * n_samples)
        # dbeta[j, j] -= alpha[j] * sign_beta[j] / L[j]
        # dr -= np.outer(Xs[:, j], (dbeta[j] - dbeta_old))


# @njit
# def _update_only_jac_sparse_p(
#         data, indptr, indices, n_samples, n_features, dbeta, dr, L,
#         alpha, sign_beta):
#     # n_samples, n_features = Xs.shape
#     for j in range(n_features):
#         # get the j-st column of X in sparse format
#         Xjs = data[indptr[j]:indptr[j+1]]
#         # get the non zero indices
#         idx_nz = indices[indptr[j]:indptr[j+1]]
#         # store old beta j for fast update
#         dbeta_old = dbeta[j, :].copy()

#         dbeta[j:j+1, :] += Xjs @ dr[idx_nz] / (L[j] * n_samples)
#         dbeta[j, j] -= alpha[j] * sign_beta[j] / L[j]
#         dr[idx_nz] -= np.outer(Xjs, (dbeta[j] - dbeta_old))


@njit
def _update_only_jac(Xs, dbeta, dr, L, alpha, sign_beta):
    n_samples, n_features = Xs.shape
    for j in range(n_features):
        # dbeta_old = dbeta[j].copy()
        dbeta_old = dbeta[j]
        dbeta[j] += Xs[:, j].T @ dr / (L[j] * n_samples)
        dbeta[j] -= alpha * sign_beta[j] / L[j]
        dr -= Xs[:, j] * (dbeta[j] - dbeta_old)
