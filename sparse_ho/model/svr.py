import numpy as np
from numpy.linalg import norm
from numba import njit
from sparse_ho.utils import init_dbeta0_new
from sparse_ho.utils import proj_box_svm, ind_box, compute_grad_proj
import scipy.sparse.linalg as slinalg
from scipy.sparse import issparse


class SVR():
    """
    Should we remove the SVR?

    Sparse Logistic Regression classifier.
    The objective function is:

    TODO

    Parameters
    ----------
    X: {ndarray, sparse matrix} of (n_samples, n_features)
        Data.
    y: {ndarray, sparse matrix} of (n_samples)
        Target
    TODO: other parameters should be remove
    """

    def __init__(self, X, y, logC, log_epsilon, max_iter=100, tol=1e-3):
        self.hyperparam = np.array([logC, log_epsilon])
        self.max_iter = max_iter
        self.tol = tol
        self.X = X
        self.y = y

    def _init_dbeta_dr(self, X, y, dense0=None,
                       mask0=None, jac0=None, compute_jac=True):
        n_samples, n_features = X.shape
        dbeta = np.zeros((2 * n_samples, 2))
        if jac0 is None or not compute_jac:
            dr = np.zeros((n_features, 2))
        else:
            dbeta[mask0, :] = jac0.copy()
        if issparse(self.X):
            dr = (self.X.T).multiply(y * dbeta)
            dr = np.sum(dr, axis=1)
            dr = np.squeeze(np.array(dr))
        else:
            dr = X.T @ (dbeta[0:n_samples] - dbeta[n_samples:(2 * n_samples)])
        return dbeta, dr

    def _init_beta_r(self, X, y, mask0, dense0):
        n_samples, n_features = X.shape
        beta = np.zeros(2 * n_samples)
        if dense0 is None:
            r = np.zeros(n_features)
        else:
            beta[mask0] = dense0
            if issparse(self.X):
                r = np.sum(self.X.T.multiply(y * beta), axis=1)
                r = np.squeeze(np.array(r))
            else:
                r = X.T @ (beta[0:n_samples] - beta[n_samples:(2 * n_samples)])
        return beta, r

    @staticmethod
    # @njit
    def _update_beta_jac_bcd(
            X, y, beta, dbeta, r, dr, hyperparam, L, compute_jac=True):
        """
            beta : dual variable of the svm
            r : primal used for cheap updates
            dbeta : jacobian of the dual variables
            dr : jacobian of the primal variable
        """
        C = hyperparam[0]
        epsilon = hyperparam[1]
        n_samples = X.shape[0]
        for j in range(2 * n_samples):
            if j < n_samples:
                F = np.sum(r * X[j, :]) + epsilon - y[j]
                beta_old = beta[j]
                zj = beta[j] - F / L[j]
                beta[j] = proj_box_svm(zj, C)
                r += (beta[j] - beta_old) * X[j, :]
                if compute_jac:
                    dF = np.array([np.sum(dr[:, 0].T * X[j, :]), epsilon + np.sum(dr[:, 1].T * X[j, :])])
                    dbeta_old = dbeta[j, :]
                    dzj = dbeta[j, :] - dF / L[j]
                    dbeta[j, :] = ind_box(zj, C) * dzj
                    dbeta[j, 0] += C * (C <= zj)
                    dr[:, 0] += (dbeta[j, 0] - dbeta_old[0]) * X[j, :]
                    dr[:, 1] += (dbeta[j, 1] - dbeta_old[1]) * X[j, :]
            if j >= n_samples:
                F = - np.sum(r * X[j - n_samples, :]) + epsilon + y[j - n_samples]
                beta_old = beta[j]
                zj = beta[j] - F / L[j - n_samples]
                beta[j] = proj_box_svm(zj, C)
                r -= (beta[j] - beta_old) * X[j - n_samples, :]
                if compute_jac:
                    dF = np.array([- np.sum(dr[:, 0].T * X[j - n_samples, :]), - np.sum(dr[:, 1].T * X[j - n_samples, :] + epsilon)])
                    dbeta_old = dbeta[j, :]
                    dzj = dbeta[j, :] - dF / L[j - n_samples]
                    dbeta[j, :] = ind_box(zj, C) * dzj
                    dbeta[j, 0] += C * (C <= zj)
                    dr[:, 0] -= (dbeta[j, 0] - dbeta_old[0]) * X[j - n_samples, :]
                    dr[:, 1] -= (dbeta[j, 1] - dbeta_old[1]) * X[j - n_samples, :]

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

    def _get_pobj0(self, r, beta, hyperparam, y):
        n_samples = self.X.shape[0]
        obj_prim = hyperparam[0] * np.sum(np.maximum(
            np.abs(y) - hyperparam[1], np.zeros(n_samples)))
        return obj_prim

    def _get_pobj(self, r, beta, hyperparam, y):
        # r = y.copy()
        n_samples = self.X.shape[0]
        obj_prim = 0.5 * norm(r) ** 2 + hyperparam[0] * np.sum(np.maximum(
            np.abs(self.X @ r - y) - hyperparam[1], np.zeros(n_samples)))
        obj_dual = 0.5 * r.T @ r + hyperparam[1] * np.sum(beta)
        obj_dual -= np.sum(y * (beta[0:n_samples] - beta[n_samples:(2 * n_samples)]))
        return (obj_dual + obj_prim)

    @staticmethod
    def _get_jac(dbeta, mask):
        return dbeta[mask, :]

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
        dbeta = np.zeros((2 * n_features, 2))
        return dbeta

    @staticmethod
    def _init_dr(dbeta, X, y, sign_beta, alpha):
        is_sparse = issparse(X)
        if is_sparse:
            res = np.array(np.sum(X.T.multiply(y * dbeta), axis=1))
            return res.reshape((res.shape[0],))
        else:
            return X.T @ dbeta

    @staticmethod
    @njit
    def _update_only_jac(Xs, ys, r, dbeta, dr, L, hyperparam, sign_beta):
        supp = np.where(sign_beta == 0.0)
        dbeta[sign_beta == 1.0, :] = np.array([hyperparam[0], 0])
        dr = Xs.T @ dbeta
        n_samples = L.shape[0]
        for j in supp[0]:
            if j < n_samples:
                dF = np.array([np.sum(dr * Xs[j, :]), hyperparam[1]])
                dbeta_old = dbeta[j, :]
                dzj = dbeta[j, :] - dF / L[j]
                dbeta[j, :] = dzj
                dr += (dbeta[j, :] - dbeta_old) * Xs[j, :]
            if j >= n_samples:
                dF = np.array([- np.sum(dr * Xs[j - n_samples, :]), hyperparam[1]])
                dbeta_old = dbeta[j, :]
                dzj = dbeta[j, :] - dF / L[j - n_samples]
                dbeta[j, :] = dzj
                dr -= (dbeta[j, :] - dbeta_old) * Xs[j - n_samples, :]

    @staticmethod
    @njit
    def _update_only_jac_sparse(
            data, indptr, indices, y, n_samples, n_features,
            dbeta, r, dr, L, C, sign_beta):
        supp = np.where(sign_beta == 0.0)
        for j in np.where(sign_beta == 1.0)[0]:
            Xis = data[indptr[j]:indptr[j+1]]
            idx_nz = indices[indptr[j]:indptr[j+1]]
            dr[idx_nz] += ((C - dbeta[j]) * y[j] * Xis)
        dbeta[sign_beta == 1.0] = C
        for j in supp[0]:
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

    def reduce_X(self, mask):
        return self.X[mask, :]

    def reduce_y(self, mask):
        return self.y[mask]

    def sign(self, x, log_alpha):
        sign = np.zeros(x.shape[0])
        sign[np.isclose(x, 0.0)] = -1.0
        sign[np.isclose(x, np.exp(self.logC))] = 1.0
        return sign

    def get_jac_v(self, mask, dense, jac, v):
        n_samples, n_features = self.X.shape
        if issparse(self.X):
            primal_jac = np.sum(self.X[mask, :].T.multiply(self.y[mask] * jac), axis=1)
            primal_jac = np.squeeze(np.array(primal_jac))
            primal = np.sum(self.X[mask, :].T.multiply(self.y[mask] * dense), axis=1)
            primal = np.squeeze(np.array(primal))
        else:
            primal_jac = self.X[mask, :].T @ jac
            primal = self.X[mask, :].T
        mask_primal = np.repeat(True, primal.shape[0])
        dense_primal = primal[mask_primal]
        return primal_jac[primal_jac != 0].T @ v(mask_primal, dense_primal)[primal_jac != 0]

    def get_beta(self, mask, dense):
        if issparse(self.X):
            primal = np.sum(self.X[mask, :].T.multiply(self.y[mask] * dense), axis=1)
            primal = np.squeeze(np.array(primal))
        else:
            primal = np.sum(self.y[mask] * dense * self.X[mask, :].T, axis=1)
        mask_primal = primal != 0
        dense_primal = primal[mask_primal]
        return mask_primal, dense_primal

    @staticmethod
    def get_full_jac_v(mask, jac_v, n_features):
        return jac_v

    def get_hessian(self, X_train, y_train, mask, dense, log_alpha):
        beta = np.zeros(self.X.shape[0])
        beta[mask] = dense
        full_supp = np.logical_and(
            np.logical_not(np.isclose(beta, 0)),
            np.logical_not(np.isclose(beta, np.exp(self.hyperparam[0]))))
        Q = X_train[full_supp, :] @ X_train[full_supp, :].T
        return Q

    def _get_jac_t_v(self, jac, mask, dense, C, v):
        C = C[0]
        n_samples = self.X.shape[0]
        beta = np.zeros(n_samples)
        beta[mask] = dense
        maskC = np.isclose(beta, C)
        full_supp = np.logical_and(np.logical_not(np.isclose(beta, 0)), np.logical_not(np.isclose(beta, C)))

        full_jac = np.zeros(n_samples)
        full_jac[full_supp] = jac
        full_jac[maskC] = C

        # primal dual relation
        jac_primal = (self.y[mask] * full_jac[mask]) @ self.X[mask, :]

        return jac_primal @ v

        # if issparse(self.X):
        #     mat = self.X[full_supp, :].multiply(self.y[full_supp, np.newaxis])
        #     Q = mat @ (self.X[maskC, :].multiply(self.y[maskC, np.newaxis])).T
        # else:
        #     mat = self.y[full_supp, np.newaxis] * self.X[full_supp, :]
        #     Q = mat @ (self.y[maskC, np.newaxis] * self.X[maskC, :]).T

        # u = (np.eye(Q.shape[0], Q.shape[1]) - Q) @ (np.ones(maskC.sum()) * C)
        # if issparse(self.X):
        #     temp = self.X[maskC, :].multiply(self.y[maskC, np.newaxis])
        #     w = temp @ v
        # else:
        #     w = ((self.y[maskC, np.newaxis] * self.X[maskC, :]) @ v)

        # if issparse(self.X):
        #     return np.array(u @ jac + C * np.sum(w))[0]
        # else:
        #     return np.array(u @ jac + C * np.sum(w))

    def restrict_full_supp(self, mask, dense, v):
        C = np.exp(self.logC)
        n_samples = self.X.shape[0]
        beta = np.zeros(n_samples)
        beta[mask] = dense
        maskC = np.isclose(beta, C)
        full_supp = np.logical_and(np.logical_not(np.isclose(beta, 0)), np.logical_not(np.isclose(beta, C)))
        if issparse(self.X):
            mat = self.X[full_supp, :].multiply(self.y[full_supp, np.newaxis])
            Q = mat @ (self.X[maskC, :].multiply(self.y[maskC, np.newaxis])).T
        else:
            mat = self.y[full_supp, np.newaxis] * self.X[full_supp, :]
            Q = mat @ (self.y[maskC, np.newaxis] * self.X[maskC, :]).T

        w = (np.eye(Q.shape[0], Q.shape[1]) - Q) @ (np.ones(maskC.sum()) * C)
        if issparse(self.X):
            return - np.array(w)[0]
        else:
            return - w
        # n_samples = self.X.shape[0]
        # beta = np.zeros(n_samples)
        # beta[mask] = dense
        # full_supp = np.logical_and(np.logical_not(np.isclose(beta, 0)), np.logical_not(np.isclose(beta, np.exp(self.logC))))
        # if issparse(self.X):
        #     temp = self.X[full_supp, :].multiply(self.y[full_supp, np.newaxis])
        #     res = (temp @ v)
        # else:
        #     res = ((self.y[full_supp, np.newaxis] * self.X[full_supp, :]) @ v)
        # return - res

    def proj_hyperparam(self, X, y, log_alpha):
        if log_alpha < -16.0:
            log_alpha = -16.0
        elif log_alpha > 4:
            log_alpha = 4
        return log_alpha

    def get_jac_obj(self, Xs, ys, sign_beta, dbeta, r, dr, C):
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
