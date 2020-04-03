import numpy as np
from numpy.linalg import norm
from scipy.sparse import issparse
import scipy.sparse.linalg as slinalg
from numba import njit
from sparse_ho.utils import (
    ST, mcp_pen, mcp_prox, mcp_dx, mcp_dgamma, mcp_dalpha)


def get_beta_jac_iterdiff(
        X, y, log_alpha, mask0=None, dense0=None, jac0=None, maxit=1000,
        tol=1e-3, compute_jac=True, model="lasso", backward=False):
    """
    Parameters
    --------------
    X: np.array, shape (n_samples, n_features)
        design matrix
        It can also be a sparse CSC matrix
    y: np.array, shape (n_samples,)
        observations
    log_alpha: float or np.array, shape (n_features)
        log  of eth coefficient multiplying the penalization
    beta0: np.array, shape (n_features,)
        initial value of the regression coefficients
        beta for warm start
    dbeta0: np.array, shape (n_features,)
        initial value of the jacobian dbeta for warm start
    maxit: int
        number of iterations of the algorithm
    tol: float
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        primal decrease for optimality and continues until it
        is smaller than ``tol``
    compute_jac: bool
        to compute or not the Jacobian along with the regression
        coefficients
    model: string
        model used, "lasso", "wlasso", or "mcp"
    backward: bool
        to store the iterates or not in order to compute the Jacobian in a
        backward way
    """

    n_samples, n_features = X.shape
    is_sparse = issparse(X)
    if is_sparse:
        L = slinalg.norm(X, axis=0) ** 2 / n_samples
    else:
        L = norm(X, axis=0) ** 2 / n_samples

    ############################################
    alpha = np.exp(log_alpha)
    if not alpha.shape:
        alphas = np.ones(n_features) * alpha
    else:
        alphas = alpha.copy()

    ############################################
    # warm start for beta
    beta = np.zeros(n_features)
    if dense0 is None or len(dense0) == 0:
        r = y.copy()
        r = r.astype(np.float)
    else:
        beta[mask0] = dense0.copy()
        r = y - X[:, mask0] @ dense0

    ############################################
    # warm start for dbeta
    if model == "lasso":
        dbeta = np.zeros(n_features)
        if dense0 is None or not compute_jac:
            dr = np.zeros(n_samples)
        else:
            dbeta[mask0] = jac0.copy()
            dr = - X[:, mask0] @ jac0.copy()
    elif model == "wlasso":
        dbeta = np.zeros((n_features, n_features))
        dr = np.zeros((n_samples, n_features))
        if jac0 is not None:
            dbeta[np.ix_(mask0, mask0)] = jac0.copy()
            dr[:, mask0] = - X[:, mask0] @ jac0
    elif model == "mcp":
        dbeta = np.zeros((n_features, 2))
        dr = np.zeros((n_samples, 2))
        alpha = np.exp(log_alpha[0])
        assert np.exp(log_alpha[1]) > 1
        # nz_min = L[L != 0].min()
        # gamma = np.exp(log_alpha[1]) / nz_min
        gamma = np.exp(log_alpha[1])
        if jac0 is not None:
            dbeta[mask0, :] = jac0.copy()
    ############################################
    # store the values of the objective function
    pobj0 = norm(y) ** 2 / (2 * n_samples)
    pobj = []
    ############################################
    # store the iterates if needed
    if backward:
        list_sign = []

    for i in range(maxit):
        print("%i -st iteration over %i" % (i, maxit))
        if model == "lasso":
            if is_sparse:
                _update_bcd_jac_sparse(
                    X.data, X.indptr, X.indices, n_samples, n_features, beta,
                    dbeta, r, dr, alphas, L, compute_jac=compute_jac)
            else:
                _update_bcd_jac(
                    X, beta, dbeta, r, dr, alphas, L, compute_jac=compute_jac)
        elif model == "wlasso":
            if is_sparse:
                _update_bcd_jac_alasso_sparse(
                    X.data, X.indptr, X.indices, n_samples, n_features, beta,
                    dbeta, r, dr, alpha, L)
            else:
                _update_bcd_jac_alasso(
                     X, beta, dbeta, r, dr, alphas, L, compute_jac=compute_jac)
        elif model == "mcp":
            if is_sparse:
                _update_bcd_jac_mcp_sparse(
                    X.data, X.indptr, X.indices, n_samples, n_features,
                    beta, dbeta, r, dr, alpha, gamma, L,
                    compute_jac=compute_jac)
            else:
                _update_bcd_jac_mcp(
                    X, beta, dbeta, r, dr, alpha, gamma, L,
                    compute_jac=compute_jac)

        this_pobj = norm(r) ** 2 / (2 * n_samples)
        if model == "mcp":
            this_pobj += mcp_pen(beta, alpha, gamma=gamma).sum()
        else:
            this_pobj += norm(alphas * beta, 1)
        pobj.append(this_pobj)
        if i > 1:
            assert pobj[-1] - pobj[-2] <= 1e-5 * pobj[0]
            print("relative decrease = ", (pobj[-2] - pobj[-1]) / pobj0)
        if (i > 1) and (pobj[-2] - pobj[-1] <= pobj0 * tol):
            break
        if backward:
            list_sign.append(np.sign(beta).copy())
    else:
        print('did not converge !')
        # raise RuntimeError('did not converge !')

    mask = beta != 0
    dense = beta[mask]
    if model == "lasso":
        jac = dbeta[mask]
    elif model == "wlasso":
        jac = dbeta[np.ix_(mask, mask)]
    elif model == "mcp":
        jac = dbeta[mask, :]
    else:
        raise ValueError('No model called %s' % model)

    if backward:
        return mask, dense, list_sign
    else:
        if compute_jac:
            return mask, dense, jac
        else:
            return mask, dense


@njit
def _update_bcd_jac_mcp_sparse(
        data, indptr, indices, n_samples, n_features, beta, dbeta, r,
        dr, alpha, gamma, L, compute_jac=True):
    non_zeros = np.where(L != 0)[0]

    for j in non_zeros:
        # get the j-st column of X in sparse format
        Xjs = data[indptr[j]:indptr[j+1]]
        # get non zero idices
        idx_nz = indices[indptr[j]:indptr[j+1]]
        beta_old = beta[j]
        if compute_jac:
            dbeta_old = dbeta[j, :]
        zj = beta[j] + Xjs.T @ r[idx_nz] / (L[j] * n_samples)
        beta[j] = mcp_prox(zj, alpha / L[j], gamma * L[j])
        assert not np.isnan(beta[j])
        if compute_jac:
            dzj = dbeta[j] + Xjs.T @ dr[idx_nz, :] / (L[j] * n_samples)
            dbeta[j:j+1, :] = mcp_dx(beta[j], alpha, gamma) * dzj
            dbeta[j:j+1, 0] += alpha / L[j] * mcp_dalpha(
                beta[j], alpha / L[j], gamma * L[j])
            dbeta[j:j+1, 1] += gamma * L[j] * mcp_dgamma(
                beta[j], alpha / L[j], gamma * L[j])
            dr[idx_nz, :] -= np.outer(Xjs, (dbeta[j, :] - dbeta_old))
            # assert not np.isnan(dbeta.min())
        r[idx_nz] -= Xjs * (beta[j] - beta_old)


@njit
def _update_bcd_jac_mcp(
        X, beta, dbeta, r, dr, alpha, gamma, L, compute_jac=True):
    n_samples, n_features = X.shape
    non_zeros = np.where(L != 0)[0]

    for j in non_zeros:
        beta_old = beta[j]
        if compute_jac:
            dbeta_old = dbeta[j, :]
        zj = beta[j] + X[:, j].T @ r / (L[j] * n_samples)
        beta[j] = mcp_prox(zj, alpha / L[j], gamma * L[j])
        assert not np.isnan(beta[j])
        if compute_jac:
            dzj = dbeta[j] + X[:, j] @ dr / (L[j] * n_samples)
            dbeta[j:j+1, :] = mcp_dx(beta[j], alpha, gamma) * dzj
            dbeta[j:j+1, 0] += alpha / L[j] * mcp_dalpha(
                beta[j], alpha / L[j], gamma * L[j])

            dbeta[j:j+1, 1] += gamma * L[j] * mcp_dgamma(
                beta[j], alpha / L[j], gamma * L[j])
            dr -= np.outer(X[:, j], (dbeta[j, :] - dbeta_old))
        r -= X[:, j] * (beta[j] - beta_old)


@njit
def _update_bcd_jac_sparse(
        data, indptr, indices, n_samples, n_features, beta,
        dbeta, r, dr, alphas, L,
        compute_jac=True):

    non_zeros = np.where(L != 0)[0]

    for j in non_zeros:
        # get the j-st column of X in sparse format
        Xjs = data[indptr[j]:indptr[j+1]]
        # get the non zero idices
        idx_nz = indices[indptr[j]:indptr[j+1]]
        beta_old = beta[j]
        if compute_jac:
            dbeta_old = dbeta[j]
        zj = beta[j] + r[idx_nz] @ Xjs / (L[j] * n_samples)
        beta[j:j+1] = ST(zj, alphas[j] / L[j])
        if compute_jac:
            dzj = dbeta[j] + Xjs @ dr[idx_nz] / (L[j] * n_samples)
            dbeta[j:j+1] = np.abs(np.sign(beta[j])) * dzj
            dbeta[j:j+1] -= alphas[j] * np.sign(beta[j]) / L[j]
            # update residuals
            dr[idx_nz] -= Xjs * (dbeta[j] - dbeta_old)
        r[idx_nz] -= Xjs * (beta[j] - beta_old)


@njit
def _update_bcd_jac(
        X, beta, dbeta, r, dr, alpha, L, compute_jac=True):
    n_samples, n_features = X.shape
    non_zeros = np.where(L != 0)[0]

    for j in non_zeros:
        beta_old = beta[j]
        if compute_jac:
            dbeta_old = dbeta[j]
            # compute derivatives
        zj = beta[j] + r @ X[:, j] / (L[j] * n_samples)
        beta[j:j+1] = ST(zj, alpha[j] / L[j])
        if compute_jac:
            dzj = dbeta[j] + X[:, j] @ dr / (L[j] * n_samples)
            dbeta[j:j+1] = np.abs(np.sign(beta[j])) * dzj
            dbeta[j:j+1] -= alpha[j] * np.sign(beta[j]) / L[j]
            # update residuals
            dr -= X[:, j] * (dbeta[j] - dbeta_old)
        r -= X[:, j] * (beta[j] - beta_old)


@njit
def _update_bcd_jac_alasso_sparse(
        data, indptr, indices, n_samples, n_features, beta, dbeta, r, dr,
        alpha, L, compute_jac=True):
    non_zeros = np.where(L != 0)[0]

    for j in non_zeros:
        # get the j-st column of X in sparse format
        Xjs = data[indptr[j]:indptr[j+1]]
        # get non zero idices
        idx_nz = indices[indptr[j]:indptr[j+1]]
        ###########################################
        beta_old = beta[j]
        if compute_jac:
            dbeta_old = dbeta[j, :].copy()
        zj = beta[j] + r[idx_nz] @ Xjs / (L[j] * n_samples)
        beta[j:j+1] = ST(zj, alpha[j] / L[j])
        if compute_jac:
            dzj = dbeta[j, :] + Xjs @ dr[idx_nz, :] / (L[j] * n_samples)
            dbeta[j:j+1, :] = np.abs(np.sign(beta[j])) * dzj
            dbeta[j:j+1, j] -= alpha[j] * np.sign(beta[j]) / L[j]
            # update residuals
            dr[idx_nz, :] -= np.outer(Xjs, (dbeta[j, :] - dbeta_old))
        r[idx_nz] -= Xjs * (beta[j] - beta_old)


@njit
def _update_bcd_jac_alasso(
        X, beta, dbeta, r, dr, alpha, L, compute_jac=True):
    n_samples, n_features = X.shape
    non_zeros = np.where(L != 0)[0]

    for j in non_zeros:
        beta_old = beta[j]
        if compute_jac:
            dbeta_old = dbeta[j, :].copy()
        zj = beta[j] + r @ X[:, j] / (L[j] * n_samples)
        beta[j:j+1] = ST(zj, alpha[j] / L[j])
        if compute_jac:
            dzj = dbeta[j, :] + X[:, j] @ dr / (L[j] * n_samples)
            dbeta[j:j+1, :] = np.abs(np.sign(beta[j])) * dzj
            dbeta[j:j+1, j] -= alpha[j] * np.sign(beta[j]) / L[j]
            # update residuals
            dr -= np.outer(X[:, j], (dbeta[j, :] - dbeta_old))
        r -= X[:, j] * (beta[j] - beta_old)
