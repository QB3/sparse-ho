import numpy as np
from numpy.linalg import norm
from numba import njit
from sparse_ho.utils import init_dbeta0_new
from sparse_ho.utils import proj_box_svm, ind_box, compute_grad_proj
import scipy.sparse.linalg as slinalg
from scipy.sparse import issparse


class SVM():
    r"""Support vector machines.
    The dual optimization problem for the SVM is:

    ..math::

        1/2 w^\top (y * X)(y * X)^\top w - \sum_i w_i
        s.t 0 <= w_i <= C

    Parameters
    ----------
    logC
    max_iter
    tol
    TODO
    """

    def __init__(self, logC, max_iter=100, tol=1e-3):
        self.logC = logC
        self.max_iter = max_iter
        self.tol = tol

    @staticmethod
    def _init_dbeta_dr(X, y, dense0=None,
                       mask0=None, jac0=None, compute_jac=True):
        n_samples, n_features = X.shape
        dbeta = np.zeros(n_samples)
        if jac0 is None or not compute_jac:
            dr = np.zeros(n_features)
        else:
            dbeta[mask0] = jac0.copy()
        if issparse(X):
            dr = (X.T).multiply(y * dbeta)
            dr = np.sum(dr, axis=1)
            dr = np.squeeze(np.array(dr))
        else:
            dr = np.sum(y * dbeta * X.T, axis=1)
        return dbeta, dr

    @staticmethod
    def _init_beta_r(X, y, mask0, dense0):
        beta = np.zeros(X.shape[0])
        if dense0 is None:
            r = np.zeros(X.shape[1])
        else:
            beta[mask0] = dense0
            if issparse(X):
                r = np.sum(X.T.multiply(y * beta), axis=1)
                r = np.squeeze(np.array(r))
            else:
                r = np.sum(y * beta * X.T, axis=1)
        return beta, r

    @staticmethod
    @njit
    def _update_beta_jac_bcd(
            X, y, beta, dbeta, r, dr, C, L, compute_jac=True):
        """
            beta : dual variable of the svm
            r : primal used for cheap updates
            dbeta : jacobian of the dual variables
            dr : jacobian of the primal variable
        """
        C = C[0]
        n_samples = X.shape[0]
        for j in range(n_samples):
            F = y[j] * np.sum(r * X[j, :]) - 1.0
            beta_old = beta[j]
            zj = beta[j] - F / L[j]
            beta[j] = proj_box_svm(zj, C)
            r += (beta[j] - beta_old) * y[j] * X[j, :]
            if compute_jac:
                dF = y[j] * np.sum(dr * X[j, :])
                dbeta_old = dbeta[j]
                dzj = dbeta[j] - dF / L[j]
                dbeta[j] = ind_box(zj, C) * dzj
                dbeta[j] += C * (C <= zj)
                dr += (dbeta[j] - dbeta_old) * y[j] * X[j, :]

    @staticmethod
    @njit
    def _update_beta_jac_bcd_sparse(
            data, indptr, indices, y, n_samples, n_features, beta,
            dbeta, r, dr, C, L, compute_jac=True):
        # data needs to be a row sparse matrix
        non_zeros = np.where(L != 0)[0]
        C = C[0]
        for j in non_zeros:
            # get the i-st row of X in sparse format
            Xis = data[indptr[j]:indptr[j+1]]
            # get the non zero indices
            idx_nz = indices[indptr[j]:indptr[j+1]]

            # compute gradient_i
            G = y[j] * np.sum(r[idx_nz] * Xis) - 1.0

            # compute projected gradient
            PG = compute_grad_proj(beta[j], G, C)

            if np.abs(PG) > 1e-12:
                beta_old = beta[j]
                # update one coefficient SVM
                zj = beta[j] - G / L[j]
                beta[j] = min(max(zj, 0), C)
                r[idx_nz] += (beta[j] - beta_old) * y[j] * Xis
                if compute_jac:
                    dbeta_old = dbeta[j]
                    dG = y[j] * np.sum(dr[idx_nz] * Xis)
                    dzj = dbeta[j] - dG / L[j]
                    dbeta[j:j+1] = ind_box(zj, C) * dzj
                    dbeta[j:j+1] += C * (C <= zj)
                    # update residuals
                    dr[idx_nz] += (dbeta[j] - dbeta_old) * y[j] * Xis

    @staticmethod
    def _get_pobj0(r, beta, C, y):
        C = C[0]
        n_samples = r.shape[0]
        obj_prim = C * np.sum(np.maximum(
            np.ones(n_samples), np.zeros(n_samples)))
        return obj_prim

    @staticmethod
    def _get_pobj(r, X, beta, C, y):
        C = C[0]
        n_samples = X.shape[0]
        obj_prim = 0.5 * norm(r) ** 2 + C * np.sum(np.maximum(
            np.ones(n_samples) - (X @ r) * y, np.zeros(n_samples)))
        obj_dual = 0.5 * r.T @ r - np.sum(beta)
        return (obj_dual + obj_prim)

    @staticmethod
    def _get_jac(dbeta, mask):
        return dbeta[mask]

    @staticmethod
    def _init_dbeta0(mask, mask0, jac0):
        size_mat = mask.sum()
        if jac0 is not None:
            dbeta0_new = init_dbeta0_new(jac0, mask, mask0)
        else:
            dbeta0_new = np.zeros(size_mat)
        return dbeta0_new

    @staticmethod
    def _init_dbeta(n_features):
        dbeta = np.zeros(n_features)
        return dbeta

    @staticmethod
    def _init_dr(dbeta, X, y, sign_beta, C):
        dbeta[sign_beta == 1] = C
        is_sparse = issparse(X)
        if is_sparse:
            res = np.array(np.sum(X.T.multiply(y * dbeta), axis=1))
            return res.reshape((res.shape[0],))
        else:
            return np.sum(y * dbeta * X.T, axis=1)

    @staticmethod
    @njit
    def _update_only_jac(Xs, ys, r, dbeta, dr, L, C, sign_beta):
        for j in np.arange(0, Xs.shape[0])[sign_beta == 0.0]:
            dF = ys[j] * np.sum(dr * Xs[j, :])
            dbeta_old = dbeta[j]
            dzj = dbeta[j] - (dF / L[j])
            dbeta[j] = dzj
            dr += (dbeta[j] - dbeta_old) * ys[j] * Xs[j, :]

    @staticmethod
    @njit
    def _update_only_jac_sparse(
            data, indptr, indices, y, n_samples, n_features,
            dbeta, r, dr, L, C, sign_beta):
        for j in np.arange(0, n_samples)[sign_beta == 0.0]:
            # get the i-st row of X in sparse format
            Xis = data[indptr[j]:indptr[j+1]]
            # get the non zero idices
            idx_nz = indices[indptr[j]:indptr[j+1]]
            # store old beta j for fast update
            dF = y[j] * np.sum(dr[idx_nz] * Xis)
            dbeta_old = dbeta[j]
            dzj = dbeta[j] - (dF / L[j])
            dbeta[j] = dzj
            dr[idx_nz] += ((dbeta[j] - dbeta_old) * y[j] * Xis)

    @staticmethod
    @njit
    def _reduce_alpha(alpha, mask):
        return alpha

    @staticmethod
    def get_L(X, is_sparse=False):
        if is_sparse:
            return slinalg.norm(X, axis=1) ** 2
        else:
            return norm(X, axis=1) ** 2

    @staticmethod
    def reduce_X(X, mask):
        return X[mask, :]

    @staticmethod
    def reduce_y(y, mask):
        return y[mask]

    def sign(self, x, log_C):
        sign = np.zeros(x.shape[0])
        sign[np.isclose(x, 0.0)] = -1.0
        sign[np.isclose(x, np.exp(log_C))] = 1.0
        return sign

    @staticmethod
    def get_jac_v(X, y, mask, dense, jac, v):
        n_samples, n_features = X.shape
        if issparse(X):
            primal_jac = np.sum(X[mask, :].T.multiply(y[mask] * jac), axis=1)
            primal_jac = np.squeeze(np.array(primal_jac))
            primal = np.sum(X[mask, :].T.multiply(y[mask] * dense), axis=1)
            primal = np.squeeze(np.array(primal))
        else:
            primal_jac = np.sum(y[mask] * jac * X[mask, :].T, axis=1)
            primal = np.sum(y[mask] * dense * X[mask, :].T, axis=1)
        mask_primal = np.repeat(True, primal.shape[0])
        dense_primal = primal[mask_primal]
        return primal_jac[primal_jac != 0].T @ v(mask_primal, dense_primal)[primal_jac != 0]

    @staticmethod
    def get_beta(X, y, mask, dense):
        if issparse(X):
            primal = np.sum(X[mask, :].T.multiply(y[mask] * dense), axis=1)
            primal = np.squeeze(np.array(primal))
        else:
            primal = np.sum(y[mask] * dense * X[mask, :].T, axis=1)
        mask_primal = primal != 0
        dense_primal = primal[mask_primal]
        return mask_primal, dense_primal

    @staticmethod
    def get_full_jac_v(mask, jac_v, n_features):
        return jac_v

    @staticmethod
    def get_hessian(X_train, y_train, mask, dense, log_alpha):
        beta = np.zeros(X_train.shape[0])
        beta[mask] = dense
        full_supp = np.logical_and(
            np.logical_not(np.isclose(beta, 0)),
            np.logical_not(np.isclose(beta, np.exp(log_alpha))))

        if issparse(X_train):
            mat = X_train[full_supp, :].multiply(y_train[full_supp, np.newaxis])
        else:
            mat = y_train[full_supp, np.newaxis] * X_train[full_supp, :]
        Q = mat @ mat.T
        return Q

    # @staticmethod
    def _get_jac_t_v(self, X, y, jac, mask, dense, C, v, n_samples):
        # TODO do you think we can improve svm computations?
        # in particular remove the dependency in X and y?
        C = C[0]
        beta = np.zeros(n_samples)
        beta[mask] = dense
        maskC = np.isclose(beta, C)
        full_supp = np.logical_and(
            np.logical_not(np.isclose(beta, 0)),
            np.logical_not(np.isclose(beta, C)))
        full_jac = np.zeros(n_samples)
        if full_supp.sum() != 0:
            full_jac[full_supp] = jac
        full_jac[maskC] = C
        maskp, densep = self.get_beta(X, y, mask, dense)
        # primal dual relation
        jac_primal = (y[mask] * full_jac[mask]) @ X[mask, :]
        return jac_primal[maskp] @ v

    @staticmethod
    def restrict_full_supp(X, y, mask, dense, v, log_alpha):
        C = np.exp(log_alpha)
        n_samples = X.shape[0]
        beta = np.zeros(n_samples)
        beta[mask] = dense
        maskC = np.isclose(beta, C)
        full_supp = np.logical_and(
            np.logical_not(np.isclose(beta, 0)),
            np.logical_not(np.isclose(beta, C)))
        if issparse(X):
            mat = X[full_supp, :].multiply(y[full_supp, np.newaxis])
            Q = mat @ (X[maskC, :].multiply(y[maskC, np.newaxis])).T
        else:
            mat = y[full_supp, np.newaxis] * X[full_supp, :]
            Q = mat @ (y[maskC, np.newaxis] * X[maskC, :]).T

        w = (np.eye(Q.shape[0], Q.shape[1]) - Q) @ (np.ones(maskC.sum()) * C)
        if issparse(X):
            return - np.array(w)[0]
        else:
            return - w

    def proj_hyperparam(self, X, y, log_alpha):
        if log_alpha < -16.0:
            log_alpha = -16.0
        elif log_alpha > 4:
            log_alpha = 4
        return log_alpha

    def get_jac_obj(self, Xs, ys, n_samples, sign_beta, dbeta, r, dr, C):
        full_supp = sign_beta == 0.0
        maskC = sign_beta == 1.0
        if issparse(Xs):
            yXdbeta = (Xs[full_supp, :].multiply(ys[full_supp, np.newaxis])).T @ dbeta[full_supp]
        else:
            yXdbeta = (ys[full_supp, np.newaxis] * Xs[full_supp, :]).T @ dbeta[full_supp]
        q = yXdbeta.T @ yXdbeta
        if issparse(Xs):
            linear_term = yXdbeta.T @ ((Xs[maskC, :].multiply(ys[maskC, np.newaxis])).T @ (np.ones(maskC.sum()) * C))
        else:
            linear_term = yXdbeta.T @ ((ys[maskC, np.newaxis] * Xs[maskC, :]).T @ (np.ones(maskC.sum()) * C))
        res = q + linear_term - C * np.sum(dbeta[full_supp])
        return(
            norm(res))
