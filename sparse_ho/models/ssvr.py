import numpy as np
from numpy.linalg import norm
import scipy.sparse.linalg as slinalg

from numba import njit

from sparse_ho.models.base import BaseModel
from sparse_ho.utils import proj_box_svm, ind_box


class SSVR(BaseModel):
    """The simplex support vector regression without bias
    The optimization problem is solved in the dual.
    It solves the SVR with probability vector constraints:
    sum_i beta_i = 1
    beta_i >= 0. 

    Parameters
    ----------
    log_C : float
        logarithm of the hyperparameter C
    max_iter : int
        maximum number of epochs for the coordinate descent
        algorithm
    estimator: instance of ``sklearn.base.BaseEstimator``
        An estimator that follows the scikit-learn API.
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
        dr = np.zeros((2 * n_samples + n_features + 1, 2))
        if jac0 is None or not compute_jac or self.dr is None:
            dbeta = np.zeros((n_features, 2))
        else:
            dr = self.dr
            dbeta = X.T @ (
                dr[0:n_samples, :] - dr[n_samples:(2 * n_samples), :])
            dbeta += dr[(2 * n_samples):(2 * n_samples + n_features), :]
            dbeta += dr[-1, :]
        return dbeta, dr

    def _init_beta_r(self, X, y, mask0, dense0):
        n_samples, n_features = X.shape
        r = np.zeros(2 * n_samples + n_features + 1)
        if mask0 is None or self.r is None:
            beta = np.zeros(n_features)
        else:
            r = self.r
            beta = X.T @ (r[0:n_samples] - r[n_samples:(2 * n_samples)])
            beta += r[(2 * n_samples):(2 * n_samples + n_features)]
            beta += r[-1]
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
        n_samples, n_features = X.shape
        for j in range(2 * n_samples + n_features + 1):
            if j < n_samples:
                F = np.sum(beta * X[j, :]) + epsilon - y[j]
                r_old = r[j]
                zj = r[j] - F / L[j]
                r[j] = proj_box_svm(zj, C)
                beta += (r[j] - r_old) * X[j, :]
                if compute_jac:
                    dF = np.array([np.sum(dbeta[:, 0].T * X[j, :]),
                                   np.sum(dbeta[:, 1].T * X[j, :])])
                    dr_old = dr[j, :].copy()
                    dzj = dr[j, :] - dF / L[j]
                    dr[j, :] = ind_box(zj, C) * dzj
                    dr[j, 0] += C * (C <= zj)
                    dr[j, 1] -= ind_box(zj, C) * epsilon / L[j]
                    dbeta[:, 0] += (dr[j, 0] - dr_old[0]) * X[j, :]
                    dbeta[:, 1] += (dr[j, 1] - dr_old[1]) * X[j, :]
            elif j >= n_samples and j < (2 * n_samples):
                F = - np.sum(beta * X[j - n_samples, :]) + \
                    epsilon + y[j - n_samples]
                r_old = r[j]
                zj = r[j] - F / L[j - n_samples]
                r[j] = proj_box_svm(zj, C)
                beta -= (r[j] - r_old) * X[j - n_samples, :]
                if compute_jac:
                    dF = np.array([- np.sum(dbeta[:, 0].T *
                                   X[j - n_samples, :]),
                                   - np.sum(dbeta[:, 1].T *
                                   X[j - n_samples, :])])
                    dr_old = dr[j, :].copy()
                    dzj = dr[j, :] - dF / L[j - n_samples]
                    dr[j, :] = ind_box(zj, C) * dzj
                    dr[j, 0] += C * (C <= zj)
                    dr[j, 1] -= ind_box(zj, C) * epsilon / L[j - n_samples]
                    dbeta[:, 0] -= (dr[j, 0] - dr_old[0]) * \
                        X[j - n_samples, :]
                    dbeta[:, 1] -= (dr[j, 1] - dr_old[1]) * \
                        X[j - n_samples, :]
            elif j >= (2 * n_samples) and j < (2 * n_samples + n_features):
                F = beta[j - (2 * n_samples)]
                r_old = r[j]
                zj = r[j] - F
                r[j] = max(0, zj)
                beta[j - (2 * n_samples)] -= (r_old - r[j])
                if compute_jac:
                    dF = dbeta[j - (2 * n_samples)]
                    dr_old = dr[j, :].copy()
                    dzj = dr[j, :] - dF
                    if zj > 0:
                        dr[j, :] = dzj
                    else:
                        dr[j, :] = np.repeat(0.0, 2)
                    dbeta[j - (2 * n_samples), 0] -= (dr_old[0] - dr[j, 0])
                    dbeta[j - (2 * n_samples), 1] -= (dr_old[1] - dr[j, 1])
            elif j == (2 * n_samples + n_features):
                F = np.sum(beta) - 1
                r_old = r[-1]
                zj = r[j] - F / n_features
                r[j] = zj
                beta -= (r_old - r[-1])
                if compute_jac:
                    dF = np.sum(dbeta, axis=0)
                    dr_old = dr[-1, :].copy()
                    dzj = dr[j] - dF / n_features
                    dr[j, :] = dzj
                    dbeta[:, 0] -= (dr_old[0] - dr[-1, 0])
                    dbeta[:, 1] -= (dr_old[1] - dr[-1, 1])

    @staticmethod
    @njit
    def _update_beta_jac_bcd_sparse(
            data, indptr, indices, y, n_samples, n_features, beta,
            dbeta, r, dr, hyperparam, L, compute_jac=True):
        C = hyperparam[0]
        epsilon = hyperparam[1]
        for j in range(2 * n_samples + n_features + 1):
            if j < n_samples:
                Xis = data[indptr[j]:indptr[j+1]]
                # get the non zero indices
                idx_nz = indices[indptr[j]:indptr[j+1]]
                F = np.sum(beta[idx_nz] * Xis) + epsilon - y[j]
                r_old = r[j]
                zj = r[j] - F / L[j]
                r[j] = proj_box_svm(zj, C)
                beta[idx_nz] += (r[j] - r_old) * Xis
                if compute_jac:
                    dF = np.array([np.sum(dbeta[idx_nz, 0].T * Xis),
                                   epsilon + np.sum(dbeta[idx_nz, 1].T * Xis)])
                    dr_old = dr[j, :].copy()
                    dzj = dr[j, :] - dF / L[j]
                    dr[j, :] = ind_box(zj, C) * dzj
                    dr[j, 0] += C * (C <= zj)
                    dr[j, 1] -= ind_box(zj, C) * epsilon / L[j]
                    dbeta[idx_nz, 0] += (dr[j, 0] - dr_old[0]) * Xis
                    dbeta[idx_nz, 1] += (dr[j, 1] - dr_old[1]) * Xis
            elif j >= n_samples and j < (2 * n_samples):
                Xis = data[indptr[j-n_samples]:indptr[j-n_samples+1]]
                # get the non zero indices
                idx_nz = indices[indptr[j-n_samples]:indptr[j-n_samples+1]]
                F = - np.sum(beta[idx_nz] * Xis) + \
                    epsilon + y[j - n_samples]
                r_old = r[j]
                zj = r[j] - F / L[j - n_samples]
                r[j] = proj_box_svm(zj, C)
                beta[idx_nz] -= (r[j] - r_old) * Xis
                if compute_jac:
                    dF = np.array([- np.sum(dbeta[idx_nz, 0].T * Xis),
                                   - np.sum(dbeta[idx_nz, 1].T * Xis
                                            + epsilon)])
                    dr_old = dr[j, :].copy()
                    dzj = dr[j, :] - dF / L[j - n_samples]
                    dr[j, :] = ind_box(zj, C) * dzj
                    dr[j, 0] += C * (C <= zj)
                    dr[j, 1] -= ind_box(zj, C) * epsilon / L[j - n_samples]
                    dbeta[idx_nz, 0] -= (dr[j, 0] - dr_old[0]) * \
                        Xis
                    dbeta[idx_nz, 1] -= (dr[j, 1] - dr_old[1]) * \
                        Xis
            elif j >= (2 * n_samples) and j < (2 * n_samples + n_features):
                F = beta[j - (2 * n_samples)]
                r_old = r[j]
                zj = r[j] - F
                r[j] = max(0, zj)
                beta[j - (2 * n_samples)] -= (r_old - r[j])
                if compute_jac:
                    dF = dbeta[j - (2 * n_samples)]
                    dr_old = dr[j, :].copy()
                    dzj = dr[j, :] - dF
                    if zj > 0:
                        dr[j, :] = dzj
                    else:
                        dr[j, :] = 0
                    dbeta[j - (2 * n_samples), 0] -= (dr_old[0] - dr[j, 0])
                    dbeta[j - (2 * n_samples), 1] -= (dr_old[1] - dr[j, 1])
            elif j == (2 * n_samples + n_features):
                F = np.sum(beta) - 1
                r_old = r[-1]
                zj = r[j] - F / n_features
                r[j] = zj
                beta -= (r_old - r[-1])
                if compute_jac:
                    dF = np.sum(dbeta, axis=0)
                    dr_old = dr[-1, :].copy()
                    dzj = dr[j, :] - dF / n_features
                    dr[j, :] = dzj
                    dbeta[:, 0] -= (dr_old[0] - dr[-1, 0])
                    dbeta[:, 1] -= (dr_old[1] - dr[-1, 1])

    def _get_pobj0(self, r, beta, hyperparam, y):
        n_samples = len(y)
        obj_prim = hyperparam[0] * np.sum(np.maximum(
            np.abs(y) - hyperparam[1], 0))
        return obj_prim

    def _get_pobj(self, r, X, beta, hyperparam, y):
        n_samples = X.shape[0]
        obj_prim = 0.5 * norm(beta) ** 2 + hyperparam[0] * np.sum(np.maximum(
            np.abs(X @ beta - y) - hyperparam[1], 0))
        obj_dual = 0.5 * beta.T @ beta + hyperparam[1] * np.sum(r)
        obj_dual -= np.sum(y * (r[0:n_samples] -
                                r[n_samples:(2 * n_samples)]))
        obj_dual -= 1
        
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
        if self.dbeta is not None:
            return self.dbeta
        else:
            return np.zeros((n_features, 2))

    def _init_dr(self, dbeta, X, y, sign_beta, hyperparam):
        r = self.r
        C = hyperparam[0]
        n_samples, n_features = X.shape
        sign = np.zeros(r.shape[0])
        bool_temp = r[0:(2 * n_samples + n_features)] == 0.0
        sign[0:(2 * n_samples + n_features)][bool_temp] = -1.0
        sign[0:(2 * n_samples)][r[0:(2 * n_samples)] == C] = 1.0
        dr = np.zeros((2 * n_samples + n_features + 1, 2))
        if np.sum(sign == 1.0) != 0:
            dr[sign == 1.0, 0] = np.repeat(C, (sign == 1).sum())
            dr[sign == 1.0, 1] = np.repeat(0, (sign == 1).sum())
        self.dr = dr
        self.dbeta = X.T @ (
            dr[0:n_samples, :] - dr[n_samples:(2 * n_samples), :])
        self.dbeta += dr[(2 * n_samples):(2 * n_samples + n_features), :]
        self.dbeta += dr[-1, :]
        return dr

    @staticmethod
    @njit
    def _update_only_jac(X, y, r, dbeta, dr, L, hyperparam, sign_beta):
        n_samples, n_features = X.shape
        C = hyperparam[0]
        epsilon = hyperparam[1]
        sign = np.zeros(r.shape[0])
        bool_temp = r[0:(2 * n_samples + n_features)] == 0.0
        sign[0:(2 * n_samples + n_features)][bool_temp] = -1.0
        sign[0:(2 * n_samples)][r[0:(2 * n_samples)] == C] = 1.0

        for j in np.arange(0, (2 * n_samples + n_features + 1))[sign == 0.0]:
            if j < n_samples:
                dF = np.array([np.sum(dbeta[:, 0].T * X[j, :]),
                               np.sum(dbeta[:, 1].T * X[j, :])])
                dr_old = dr[j, :].copy()
                dzj = dr[j, :] - dF / L[j]
                dr[j, :] = dzj
                dr[j, 1] -= epsilon / L[j]
                dbeta[:, 0] += (dr[j, 0] - dr_old[0]) * X[j, :]
                dbeta[:, 1] += (dr[j, 1] - dr_old[1]) * X[j, :]
            elif j >= n_samples and j < (2 * n_samples):
                dF = np.array([- np.sum(dbeta[:, 0].T * X[j - n_samples, :]),
                               - np.sum(dbeta[:, 1].T * X[j - n_samples, :])])
                dr_old = dr[j, :].copy()
                dzj = dr[j, :] - dF / L[j - n_samples]
                dr[j, :] = dzj
                dr[j, 1] -= epsilon / L[j - n_samples]
                dbeta[:, 0] -= (dr[j, 0] - dr_old[0]) * \
                    X[j - n_samples, :]
                dbeta[:, 1] -= (dr[j, 1] - dr_old[1]) * \
                    X[j - n_samples, :]
            elif j >= (2 * n_samples) and j < (2 * n_samples + n_features):
                dF = dbeta[j - (2 * n_samples)]
                dr_old = dr[j, :].copy()
                dzj = dr[j, :] - dF
                dr[j, :] = dzj
                dbeta[j - (2 * n_samples), 0] -= (dr_old[0] - dr[j, 0])
                dbeta[j - (2 * n_samples), 1] -= (dr_old[1] - dr[j, 1])
            elif j == (2 * n_samples + n_features):
                dF = np.sum(dbeta, axis=0)
                dr_old = dr[-1, :].copy()
                dzj = dr[j, :] - dF / n_features
                dr[j, :] = dzj
                dbeta[:, 0] -= (dr_old[0] - dr[-1, 0])
                dbeta[:, 1] -= (dr_old[1] - dr[-1, 1])

    @staticmethod
    @njit
    def _update_only_jac_sparse(
            data, indptr, indices, y, n_samples, n_features,
            dbeta, r, dr, L, hyperparam, sign_beta):
        C = hyperparam[0]
        epsilon = hyperparam[1]

        sign = np.zeros(r.shape[0])
        bool_temp = r[0:(2 * n_samples + n_features)] == 0.0
        sign[0:(2 * n_samples + n_features)][bool_temp] = -1.0
        sign[0:(2 * n_samples)][r[0:(2 * n_samples)] == C] = 1.0

        for j in np.arange(0, (2 * n_samples + n_features + 1))[sign == 0.0]:
            if j < n_samples:
                Xis = data[indptr[j]:indptr[j+1]]
                # get the non zero indices
                idx_nz = indices[indptr[j]:indptr[j+1]]
                dF = np.array([np.sum(dbeta[idx_nz, 0].T * Xis),
                               np.sum(dbeta[idx_nz, 1].T * Xis)])
                dr_old = dr[j, :].copy()
                dzj = dr[j, :] - dF / L[j]
                dr[j, :] = dzj
                dr[j, 1] -= epsilon / L[j]
                dbeta[idx_nz, 0] += (dr[j, 0] - dr_old[0]) * Xis
                dbeta[idx_nz, 1] += (dr[j, 1] - dr_old[1]) * Xis
            elif j >= n_samples and j < (2 * n_samples):
                Xis = data[indptr[j-n_samples]:indptr[j-n_samples+1]]
                # get the non zero indices
                idx_nz = indices[indptr[j-n_samples]:indptr[j-n_samples+1]]
                dF = np.array([- np.sum(dbeta[idx_nz, 0].T * Xis),
                               - np.sum(dbeta[idx_nz, 1].T * Xis)])
                dr_old = dr[j, :].copy()
                dzj = dr[j, :] - dF / L[j - n_samples]
                dr[j, :] = dzj
                dr[j, 1] -= epsilon / L[j]
                dbeta[idx_nz, 0] -= (dr[j, 0] - dr_old[0]) * \
                    Xis
                dbeta[idx_nz, 1] -= (dr[j, 1] - dr_old[1]) * \
                    Xis
            elif j >= (2 * n_samples) and j < (2 * n_samples + n_features):
                dF = dbeta[j - (2 * n_samples)]
                dr_old = dr[j, :].copy()
                dzj = dr[j, :] - dF
                dr[j, :] = dzj
                dbeta[j - (2 * n_samples), 0] -= (dr_old[0] - dr[j, 0])
                dbeta[j - (2 * n_samples), 1] -= (dr_old[1] - dr[j, 1])
            elif j == (2 * n_samples + n_features):
                dF = np.sum(dbeta, axis=0)
                dr_old = dr[-1, :].copy()
                dzj = dr[j, :] - dF / n_features
                dr[j, :] = dzj
                dbeta[:, 0] -= (dr_old[0] - dr[-1, 0])
                dbeta[:, 1] -= (dr_old[1] - dr[-1, 1])

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

    @staticmethod
    def get_full_jac_v(mask, jac_v, n_features):
        return jac_v

    def get_hessian(self, X, y, mask, dense, log_hyperparam):
        C = np.exp(log_hyperparam[0])
        n_samples, n_features = X.shape
        alpha = self.r[0:n_samples] - self.r[n_samples:(2 * n_samples)]
        gamma = self.r[(2 * n_samples):(2 * n_samples + n_features)]
        mask0 = gamma != 0
        full_supp = np.logical_and(alpha != 0, np.abs(alpha) != C)
        sub_id = np.zeros((mask0.sum(), n_features))
        sub_id[:, mask0] = 1.0
        hessian = np.concatenate((X[full_supp, :],
                                  -sub_id, -np.ones((1, n_features))), axis=0)
        return hessian @ hessian.T

    def get_dual_v(self, X, y, v, log_hyperparam):
        if v.shape[0] != 0:
            return np.hstack((-X @ v, v, np.sum(v)))
        else:
            return np.zeros(X.shape[0] + X.shape[1] + 1)

    def _get_jac_t_v(self, X, y, jac, mask, dense, hyperparam, v, n_samples):
        C = hyperparam[0]
        epsilon = hyperparam[1]
        alpha = self.r[0:n_samples] - self.r[n_samples:(2 * n_samples)]
        n_features = X.shape[1]
        gamma = self.r[(2 * n_samples):(2 * n_samples + n_features)]
        mask0 = gamma != 0
        full_supp = np.logical_and(alpha != 0, np.abs(alpha) != C)
        maskC = np.abs(alpha) == C
        sub_id = np.zeros((mask0.sum(), n_features))
        sub_id[:, mask0] = 1.0

        hessian = np.concatenate((X[full_supp, :], -sub_id,
                                  -np.ones((1, n_features))), axis=0)
        hessian_vec = hessian @ X[maskC, :].T @ alpha[maskC]
        jac_t_v = -hessian_vec.T @ jac
        jac_t_v -= alpha[maskC].T @ v[0:n_samples][maskC]
        jac_t_v2 = -epsilon * np.sign(alpha[full_supp]) @ \
            jac[0:full_supp.sum()]
        return np.array([jac_t_v, jac_t_v2])

    def generalized_supp(self, X, v, log_hyperparam):
        n_samples, n_features = X.shape
        C = np.exp(log_hyperparam[0])
        alpha = self.r[0:n_samples] - self.r[n_samples:(2 * n_samples)]
        full_supp = np.logical_and(alpha != 0, np.abs(alpha) != C)
        mask0 = self.r[(2 * n_samples):(2 * n_samples + n_features)] != 0
        return v[np.hstack((full_supp, mask0, True))]

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
        n_features = dbeta.shape[0]
        C = hyperparam[0]
        alpha = r[0:n_samples] - r[n_samples:(2 * n_samples)]
        dalpha = dr[0:n_samples, 0] - dr[n_samples:(2 * n_samples), 0]
        dgamma = dr[(2 * n_samples):(2 * n_samples + n_features), 0]
        dmu = dr[-1, 0]
        maskC = np.abs(alpha) == C

        full_supp = np.logical_and(alpha != 0, np.abs(alpha) != C)

        vecX = dalpha[full_supp].T @ Xs[full_supp, :]
        vecX += dgamma + np.repeat(dmu, n_features)
        quadratic_term = vecX.T @ vecX
        linear_term = vecX.T @ Xs[maskC, :].T @ alpha[maskC]
        return norm(quadratic_term + linear_term)

    def _use_estimator(self, X, y, hyperparam, tol, max_iter):
        if self.estimator is None:
            raise ValueError("You did not pass a solver with sklearn API")
        self.estimator.set_params(
            epsilon=hyperparam[1], tol=tol, C=hyperparam[0],
            fit_intercept=False, max_iter=max_iter)
        self.estimator.fit(X, y)
        mask = self.estimator.coef_ != 0
        dense = self.estimator.coef_[mask]
        return mask, dense, None
