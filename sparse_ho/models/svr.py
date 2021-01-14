import numpy as np
from numpy.linalg import norm
import scipy.sparse.linalg as slinalg
from scipy.sparse import issparse

from numba import njit

from sparse_ho.models.base import BaseModel
from sparse_ho.utils import proj_box_svm, ind_box


class SVR(BaseModel):
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

    def __init__(self, max_iter=100, estimator=None):
        self.max_iter = max_iter
        self.estimator = estimator
        self.dual = True
        self.r = None
        self.dr = None

    def _init_dbeta_dr(self, X, y, dense0=None,
                       mask0=None, jac0=None, compute_jac=True):
        n_samples, n_features = X.shape
        dr = np.zeros((2 * n_samples, 2))
        if jac0 is None or not compute_jac or self.dr is None:
            dbeta = np.zeros((n_features, 2))
        else:
            dr = self.dr
            if issparse(X):
                dbeta = (X.T).multiply(
                    dr[0:n_samples, :] - dr[n_samples:(2 * n_samples), :])
                dbeta = np.sum(dbeta, axis=1)
                dbeta = np.squeeze(np.array(dbeta))
            else:
                dbeta = X.T @ (
                    dr[0:n_samples, :] - dr[n_samples:(2 * n_samples), :])
        return dbeta, dr

    def _init_beta_r(self, X, y, mask0, dense0):
        n_samples, n_features = X.shape
        r = np.zeros(2 * n_samples)
        if mask0 is None or self.r is None:
            beta = np.zeros(n_features)
        else:
            r = self.r
            if issparse(X):
                beta = np.sum(X.T.multiply(r[0:n_samples] - r[n_samples:(2 * n_samples)]), axis=1)
                beta = np.squeeze(np.array(beta))
            else:
                beta = X.T @ (r[0:n_samples] - r[n_samples:(2 * n_samples)])
        return beta, r

    @staticmethod
    @njit
    def _update_beta_jac_bcd(
            X, y, beta, dbeta, r, dr, hyperparam, L, compute_jac=True):
        """
            beta : primal variable of the svm
            r : dual used for cheap updates
            dbeta : jacobian of the primal variables
            dr : jacobian of the dual variables
        """
        C = hyperparam[0]
        epsilon = hyperparam[1]
        n_samples = X.shape[0]
        for j in range(2 * n_samples):
            if j < n_samples:
                F = np.sum(beta * X[j, :]) + epsilon - y[j]
                r_old = r[j]
                zj = r[j] - F / L[j]
                r[j] = proj_box_svm(zj, C)
                beta += (r[j] - r_old) * X[j, :]
                if compute_jac:
                    dF = np.array([np.sum(dbeta[:, 0].T * X[j, :]),
                                   epsilon + np.sum(dbeta[:, 1].T * X[j, :])])
                    dr_old = dr[j, :].copy()
                    dzj = dr[j, :] - dF / L[j]
                    dr[j, :] = ind_box(zj, C) * dzj
                    dr[j, 0] += C * (C <= zj)
                    dbeta[:, 0] += (dr[j, 0] - dr_old[0]) * X[j, :]
                    dbeta[:, 1] += (dr[j, 1] - dr_old[1]) * X[j, :]
            if j >= n_samples:
                F = - np.sum(beta * X[j - n_samples, :]) + \
                    epsilon + y[j - n_samples]
                r_old = r[j]
                zj = r[j] - F / L[j - n_samples]
                r[j] = proj_box_svm(zj, C)
                beta -= (r[j] - r_old) * X[j - n_samples, :]
                if compute_jac:
                    dF = np.array([- np.sum(dbeta[:, 0].T * X[j - n_samples, :]),
                                   - np.sum(dbeta[:, 1].T * X[j - n_samples, :]
                                            + epsilon)])
                    dr_old = dr[j, :].copy()
                    dzj = dr[j, :] - dF / L[j - n_samples]
                    dr[j, :] = ind_box(zj, C) * dzj
                    dr[j, 0] += C * (C <= zj)
                    dbeta[:, 0] -= (dr[j, 0] - dr_old[0]) * \
                        X[j - n_samples, :]
                    dbeta[:, 1] -= (dr[j, 1] - dr_old[1]) * \
                        X[j - n_samples, :]

    @staticmethod
    @njit
    def _update_beta_jac_bcd_sparse(
            data, indptr, indices, y, n_samples, n_features, beta,
            dbeta, r, dr, C, L, compute_jac=True):
        # TODO
        return NotImplemented

    def _get_pobj0(self, r, beta, hyperparam, y):
        n_samples = len(y)
        obj_prim = hyperparam[0] * np.sum(np.maximum(
            np.abs(y) - hyperparam[1], np.zeros(n_samples)))
        return obj_prim

    def _get_pobj(self, r, X, beta, hyperparam, y):
        n_samples = X.shape[0]
        obj_prim = 0.5 * norm(beta) ** 2 + hyperparam[0] * np.sum(np.maximum(
            np.abs(X @ beta - y) - hyperparam[1], np.zeros(n_samples)))
        obj_dual = 0.5 * beta.T @ beta + hyperparam[1] * np.sum(r)
        obj_dual -= np.sum(y * (r[0:n_samples] -
                                r[n_samples:(2 * n_samples)]))
        return (obj_dual + obj_prim)

    @staticmethod
    def _get_jac(dbeta, mask):
        return dbeta[mask, :]

    @staticmethod
    def _init_dbeta0(mask, mask0, jac0):
        size_mat = mask.sum()
        if jac0 is not None:
            mask_both = np.logical_and(mask0, mask)
            size_mat = mask.sum()
            dbeta0_new = np.zeros((size_mat, 2))
            count = 0
            count_old = 0
            n_features = mask.shape[0]
            for j in range(n_features):
                if mask_both[j]:
                    dbeta0_new[count, :] = jac0[count_old, :]
                if mask0[j]:
                    count_old += 1
                if mask[j]:
                    count += 1
        else:
            dbeta0_new = np.zeros((size_mat, 2))
        return dbeta0_new

    def _init_dbeta(self, n_features):
        return np.zeros((n_features, 2))

    def _init_dr(self, dbeta, X, y, sign_beta, alpha):
        return np.zeros((2 * X.shape[0], 2))

    @staticmethod
    # @njit
    def _update_only_jac(X, y, r, dbeta, dr, L, hyperparam, sign_beta):
        n_samples = L.shape[0]
        C = hyperparam[0]
        epsilon = hyperparam[1]
        sign = np.zeros(r.shape[0])
        sign[np.isclose(r, 0.0)] = -1.0
        sign[np.isclose(r, C)] = 1.0
        if np.sum(sign == 1.0) != 0:
            dr[sign == 1.0, 0] = np.repeat(C, (sign == 1).sum())
            dr[sign == 1.0, 1] = np.repeat(0, (sign == 1).sum())
        dbeta[0:X.shape[1], :] = X.T @ (
            dr[0:n_samples] - dr[n_samples:(2 * n_samples)])
        for j in np.arange(0, (2 * n_samples))[sign == 0.0]:
            if j < n_samples:
                dF = np.array([np.sum(dbeta[:, 0].T * X[j, :]),
                              epsilon + np.sum(dbeta[:, 1].T * X[j, :])])
                dr_old = dr[j, :].copy()
                dzj = dr[j, :] - dF / L[j]
                dr[j, :] = dzj
                dbeta[:, 0] += (dr[j, 0] - dr_old[0]) * X[j, :]
                dbeta[:, 1] += (dr[j, 1] - dr_old[1]) * X[j, :]
            if j >= n_samples:
                dF = np.array([- np.sum(dbeta[:, 0].T * X[j - n_samples, :]),
                               - np.sum(dbeta[:, 1].T * X[j - n_samples, :]
                               + epsilon)])
                dr_old = dr[j, :].copy()
                dzj = dr[j, :] - dF / L[j - n_samples]
                dr[j, :] = dzj
                dbeta[:, 0] -= (dr[j, 0] - dr_old[0]) * \
                    X[j - n_samples, :]
                dbeta[:, 1] -= (dr[j, 1] - dr_old[1]) * \
                    X[j - n_samples, :]

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

    @staticmethod
    def reduce_X(X, mask):
        return X[:, mask]

    @staticmethod
    def reduce_y(y, mask):
        return y

    def sign(self, x, log_hyperparams):
        sign = np.zeros(x.shape[0])
        sign[np.isclose(x, 0.0)] = -1.0
        sign[np.isclose(x, np.exp(log_hyperparams[0]))] = 1.0
        return sign

    def get_jac_v(self, X, y, mask, dense, jac, v):
        return jac.T @ v(mask, dense)

    def get_beta(self, X, y, mask, dense):
        return mask, dense

    @staticmethod
    def get_full_jac_v(mask, jac_v, n_features):
        return jac_v

    def get_hessian(self, X_train, y_train, mask, dense, log_hyperparam):
        beta = np.zeros(self.X.shape[0])
        beta[mask] = dense
        full_supp = np.logical_and(
            np.logical_not(np.isclose(beta, 0)),
            np.logical_not(np.isclose(beta, np.exp(log_hyperparam[0]))))
        Q = X_train[full_supp, :] @ X_train[full_supp, :].T
        return Q

    def _get_jac_t_v(self, jac, mask, dense, C, v):
        C = C[0]
        n_samples = self.X.shape[0]
        beta = np.zeros(n_samples)
        beta[mask] = dense
        maskC = np.isclose(beta, C)
        full_supp = np.logical_and(np.logical_not(
            np.isclose(beta, 0)), np.logical_not(np.isclose(beta, C)))

        full_jac = np.zeros(n_samples)
        full_jac[full_supp] = jac
        full_jac[maskC] = C

        # primal dual relation
        jac_primal = (self.y[mask] * full_jac[mask]) @ self.X[mask, :]

        return jac_primal @ v

        # if issparse(self.X):
        #     mat = self.X[full_supp, :].multiply(
        #         self.y[full_supp, np.newaxis])
        #     Q = mat @ (self.X[maskC, :].multiply(
        #          self.y[maskC, np.newaxis])).T
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
        full_supp = np.logical_and(np.logical_not(
            np.isclose(beta, 0)), np.logical_not(np.isclose(beta, C)))
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
        # full_supp = np.logical_and(
        #   np.logical_not(np.isclose(beta, 0)),
        #   np.logical_not(np.isclose(beta, np.exp(self.logC))))
        # if issparse(self.X):
        #     temp = self.X[full_supp, :].multiply(
        #       self.y[full_supp, np.newaxis])
        #     res = (temp @ v)
        # else:
        #     res = ((self.y[full_supp, None] * self.X[full_supp, :]) @ v)
        # return - res

    def proj_hyperparam(self, X, y, log_hyperparam):
        if log_hyperparam[0] < -16.0:
            log_hyperparam[0] = -16.0
        elif log_hyperparam[0] > 5:
            log_hyperparam[0] = 5

        if log_hyperparam[1] < -16.0:
            log_hyperparam[1] = -16.0
        elif log_hyperparam[1] > 3:
            log_hyperparam[1] = 3

        return log_hyperparam

    def get_jac_obj(self, Xs, ys, n_samples, sign_beta,
                    dbeta, r, dr, hyperparam):
        # sign = self.sign(
        #     np.abs(r[0:n_samples] - r[n_samples:(2 * n_samples)]), np.log(hyperparam))
        C = hyperparam[0]
        quadratic_term = dbeta[:, 0].T @ dbeta[:, 0]

        temp = np.abs((dr[0:n_samples, 0] - dr[n_samples:(2 * n_samples), 0]))
        temp[np.abs(temp) != C] = 0

        linear_term = temp.T @ (
            dr[0:n_samples, 0] - dr[n_samples:(2 * n_samples), 0])
        res = quadratic_term - linear_term
        return norm(res)

    def _use_estimator(self, X, y, hyperparam, tol, max_iter):
        if self.estimator is None:
            raise ValueError("You did not pass a solver with sklearn API")
        self.estimator.set_params(
            epsilon=hyperparam[1], tol=1e-16, C=hyperparam[0],
            fit_intercept=False, max_iter=1000)
        self.estimator.fit(X, y)
        mask = self.estimator.coef_ != 0
        dense = self.estimator.coef_[mask]
        return mask, dense, None
