import numpy as np
from numpy.linalg import norm
import scipy.sparse.linalg as slinalg

from numba import njit

from sparse_ho.models.base import BaseModel
from sparse_ho.utils import proj_box_svm, ind_box


class SVR(BaseModel):
    """The support vector regression without bias
    The optimization problem is solved in the dual.

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
        self.residuals = None
        self.dresiduals = None

    def _init_dbeta_dresiduals(
            self, X, y, dense0=None, mask0=None, jac0=None, compute_jac=True):
        n_samples, n_features = X.shape
        dresiduals = np.zeros((2 * n_samples, 2))
        if jac0 is None or not compute_jac or self.dresiduals is None:
            dbeta = np.zeros((n_features, 2))
        else:
            if self.dresiduals.shape[0] != (2 * n_samples):
                dbeta = np.zeros((n_features, 2))
            else:
                dresiduals = self.dresiduals.copy()
                dbeta = X.T @ (
                    dresiduals[0:n_samples, :] -
                    dresiduals[n_samples:(2 * n_samples), :])
        return dbeta, dresiduals

    def _init_beta_residuals(self, X, y, mask0, dense0):
        n_samples, n_features = X.shape
        residuals = np.zeros(2 * n_samples)
        if mask0 is None or self.residuals is None:
            beta = np.zeros(n_features)
        else:
            if self.residuals.shape[0] != (2 * n_samples):
                beta = np.zeros(n_features)
            else:
                residuals = self.residuals
                beta = X.T @ (residuals[0:n_samples] -
                              residuals[n_samples:(2 * n_samples)])
        return beta, residuals

    @staticmethod
    @njit
    def _update_beta_jac_bcd(
            X, y, beta, dbeta, residuals, dresiduals,
            hyperparam, L, compute_jac=True):
        """
            beta : primal variable of the svr
            residuals : dual used for cheap updates
            dbeta : jacobian of the primal variables
            dresiduals : jacobian of the dual variables
        """
        C = hyperparam[0]
        epsilon = hyperparam[1]
        n_samples = X.shape[0]

        for j in range(2 * n_samples):
            if j < n_samples:
                F = np.sum(beta * X[j, :]) + epsilon - y[j]
                residuals_old = residuals[j]
                zj = residuals[j] - F / L[j]
                residuals[j] = proj_box_svm(zj, C)
                beta += (residuals[j] - residuals_old) * X[j, :]
                if compute_jac:
                    dF = np.array([np.sum(dbeta[:, 0].T * X[j, :]),
                                   np.sum(dbeta[:, 1].T * X[j, :])])
                    dresiduals_old = dresiduals[j, :].copy()
                    dzj = dresiduals[j, :] - dF / L[j]
                    dresiduals[j, :] = ind_box(zj, C) * dzj
                    dresiduals[j, 0] += C * (C <= zj)
                    dresiduals[j, 1] -= epsilon * ind_box(zj, C) / L[j]
                    dbeta[:, 0] += (dresiduals[j, 0] -
                                    dresiduals_old[0]) * X[j, :]
                    dbeta[:, 1] += (dresiduals[j, 1] -
                                    dresiduals_old[1]) * X[j, :]
            if j >= n_samples:
                F = - np.sum(beta * X[j - n_samples, :]) + \
                    epsilon + y[j - n_samples]
                residuals_old = residuals[j]
                zj = residuals[j] - F / L[j - n_samples]
                residuals[j] = proj_box_svm(zj, C)
                beta -= (residuals[j] - residuals_old) * X[j - n_samples, :]

                if compute_jac:
                    dF = np.array([- np.sum(dbeta[:, 0].T *
                                   X[j - n_samples, :]),
                                   - np.sum(dbeta[:, 1].T *
                                   X[j - n_samples, :])])
                    dresiduals_old = dresiduals[j, :].copy()
                    dzj = dresiduals[j, :] - dF / L[j - n_samples]
                    dresiduals[j, :] = ind_box(zj, C) * dzj
                    dresiduals[j, 0] += C * (C <= zj)
                    dresiduals[j, 1] -= epsilon * ind_box(zj, C) / \
                        L[j - n_samples]
                    dbeta[:, 0] -= (dresiduals[j, 0] - dresiduals_old[0]) * \
                        X[j - n_samples, :]
                    dbeta[:, 1] -= (dresiduals[j, 1] - dresiduals_old[1]) * \
                        X[j - n_samples, :]

    @staticmethod
    @njit
    def _update_beta_jac_bcd_sparse(
            data, indptr, indices, y, n_samples, n_features, beta,
            dbeta, residuals, dresiduals, hyperparam, L, compute_jac=True):
        C = hyperparam[0]
        epsilon = hyperparam[1]
        for j in range(2 * n_samples):
            if j < n_samples:
                # get the i-st row of X in sparse format
                Xis = data[indptr[j]:indptr[j+1]]
                # get the non zero indices
                idx_nz = indices[indptr[j]:indptr[j+1]]
                F = np.sum(beta[idx_nz] * Xis) + epsilon - y[j]
                residuals_old = residuals[j]
                zj = residuals[j] - F / L[j]
                residuals[j] = proj_box_svm(zj, C)
                beta[idx_nz] += (residuals[j] - residuals_old) * Xis

                if compute_jac:
                    dF = np.array([np.sum(dbeta[idx_nz, 0].T * Xis),
                                   np.sum(dbeta[idx_nz, 1].T * Xis)])
                    dresiduals_old = dresiduals[j, :].copy()
                    dzj = dresiduals[j, :] - dF / L[j]
                    dresiduals[j, :] = ind_box(zj, C) * dzj
                    dresiduals[j, 0] += C * (C <= zj)
                    dresiduals[j, 1] -= epsilon * ind_box(zj, C) / L[j]
                    dbeta[idx_nz, 0] += (dresiduals[j, 0] -
                                         dresiduals_old[0]) * Xis
                    dbeta[idx_nz, 1] += (dresiduals[j, 1] -
                                         dresiduals_old[1]) * Xis
            if j >= n_samples:
                # get the i-st row of X in sparse format
                Xis = data[indptr[j-n_samples]:indptr[j-n_samples+1]]
                # get the non zero indices
                idx_nz = indices[indptr[j-n_samples]:indptr[j-n_samples+1]]
                F = - np.sum(beta[idx_nz] * Xis) + \
                    epsilon + y[j - n_samples]
                residuals_old = residuals[j]
                zj = residuals[j] - F / L[j - n_samples]
                residuals[j] = proj_box_svm(zj, C)
                beta[idx_nz] -= (residuals[j] - residuals_old) * Xis

                if compute_jac:
                    dF = np.array([- np.sum(dbeta[idx_nz, 0].T * Xis),
                                   - np.sum(dbeta[idx_nz, 1].T * Xis)])
                    dresiduals_old = dresiduals[j, :].copy()
                    dzj = dresiduals[j, :] - dF / L[j - n_samples]
                    dresiduals[j, :] = ind_box(zj, C) * dzj
                    dresiduals[j, 0] += C * (C <= zj)
                    dresiduals[j, 1] -= epsilon * ind_box(zj, C) / \
                        L[j - n_samples]
                    dbeta[idx_nz, 0] -= (dresiduals[j, 0] -
                                         dresiduals_old[0]) * Xis
                    dbeta[idx_nz, 1] -= (dresiduals[j, 1] -
                                         dresiduals_old[1]) * Xis

    def _get_pobj0(self, residuals, beta, hyperparam, y):
        n_samples = len(y)
        obj_prim = hyperparam[0] * np.sum(np.maximum(
            np.abs(y) - hyperparam[1], np.zeros(n_samples)))
        return obj_prim

    def _get_pobj(self, residuals, X, beta, hyperparam, y):
        n_samples = X.shape[0]
        obj_prim = 0.5 * norm(beta) ** 2 + hyperparam[0] * np.sum(np.maximum(
            np.abs(X @ beta - y) - hyperparam[1], np.zeros(n_samples)))
        obj_dual = 0.5 * beta.T @ beta + hyperparam[1] * np.sum(residuals)
        obj_dual -= np.sum(y * (residuals[0:n_samples] -
                                residuals[n_samples:(2 * n_samples)]))
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

    def _init_dresiduals(self, dbeta, X, y, sign_beta, hyperparam):
        residuals = self.residuals
        C = hyperparam[0]
        n_samples = X.shape[0]
        sign = np.zeros(residuals.shape[0])
        sign[residuals == 0.0] = -1.0
        sign[residuals == C] = 1.0
        dresiduals = np.zeros((2 * X.shape[0], 2))
        if np.sum(sign == 1.0) != 0:
            dresiduals[sign == 1.0, 0] = np.repeat(C, (sign == 1).sum())
            dresiduals[sign == 1.0, 1] = np.repeat(0, (sign == 1).sum())
        self.dresiduals = dresiduals
        self.dbeta = X.T @ (
            dresiduals[0:n_samples, :] -
            dresiduals[n_samples:(2 * n_samples), :])
        return dresiduals

    @staticmethod
    @njit
    def _update_only_jac(X, y, residuals, dbeta, dresiduals,
                         L, hyperparam, sign_beta):
        n_samples = L.shape[0]
        C = hyperparam[0]
        epsilon = hyperparam[1]
        sign = np.zeros(residuals.shape[0])
        sign[residuals == 0.0] = -1.0
        sign[residuals == C] = 1.0
        for j in np.arange(0, (2 * n_samples))[sign == 0.0]:
            if j < n_samples:
                dF = np.array([np.sum(dbeta[:, 0].T * X[j, :]),
                              np.sum(dbeta[:, 1].T * X[j, :])])
                dresiduals_old = dresiduals[j, :].copy()
                dzj = dresiduals[j, :] - dF / L[j]
                dresiduals[j, :] = dzj
                dresiduals[j, 1] -= epsilon / L[j]
                dbeta[:, 0] += (dresiduals[j, 0] - dresiduals_old[0]) * X[j, :]
                dbeta[:, 1] += (dresiduals[j, 1] - dresiduals_old[1]) * X[j, :]
            if j >= n_samples:
                dF = np.array([- np.sum(dbeta[:, 0].T * X[j - n_samples, :]),
                               - np.sum(dbeta[:, 1].T * X[j - n_samples, :])])
                dresiduals_old = dresiduals[j, :].copy()
                dzj = dresiduals[j, :] - dF / L[j - n_samples]
                dresiduals[j, :] = dzj
                dresiduals[j, 1] -= epsilon / L[j - n_samples]
                dbeta[:, 0] -= (dresiduals[j, 0] - dresiduals_old[0]) * \
                    X[j - n_samples, :]
                dbeta[:, 1] -= (dresiduals[j, 1] - dresiduals_old[1]) * \
                    X[j - n_samples, :]

    @staticmethod
    @njit
    def _update_only_jac_sparse(
            data, indptr, indices, y, n_samples, n_features,
            dbeta, residuals, dresiduals, L, hyperparam, sign_beta):
        n_samples = L.shape[0]
        C = hyperparam[0]
        epsilon = hyperparam[1]
        # non_zeros = np.where(L != 0)[0]
        sign = np.zeros(residuals.shape[0])
        sign[residuals == 0.0] = -1.0
        sign[residuals == C] = 1.0

        for j in np.arange(0, (2 * n_samples))[sign == 0.0]:
            if j < n_samples:
                # get the i-st row of X in sparse format
                Xis = data[indptr[j]:indptr[j+1]]
                # get the non zero indices
                idx_nz = indices[indptr[j]:indptr[j+1]]
                dF = np.array([np.sum(dbeta[idx_nz, 0].T * Xis),
                               np.sum(dbeta[idx_nz, 1].T * Xis)])
                dresiduals_old = dresiduals[j, :].copy()
                dzj = dresiduals[j, :] - dF / L[j]
                dresiduals[j, :] = dzj
                dresiduals[j, 1] -= epsilon / L[j]
                dbeta[idx_nz, 0] += (dresiduals[j, 0] -
                                     dresiduals_old[0]) * Xis
                dbeta[idx_nz, 1] += (dresiduals[j, 1] -
                                     dresiduals_old[1]) * Xis
            if j >= n_samples:
                # get the i-st row of X in sparse format
                Xis = data[indptr[j-n_samples]:indptr[j-n_samples+1]]
                # get the non zero indices
                idx_nz = indices[indptr[j-n_samples]:indptr[j-n_samples+1]]
                dF = np.array([- np.sum(dbeta[idx_nz, 0].T * Xis),
                               - np.sum(dbeta[idx_nz, 1].T * Xis)])
                dresiduals_old = dresiduals[j, :].copy()
                dzj = dresiduals[j, :] - dF / L[j - n_samples]
                dresiduals[j, :] = dzj
                dresiduals[j, 1] -= epsilon / L[j - n_samples]
                dbeta[idx_nz, 0] -= (dresiduals[j, 0] - dresiduals_old[0]) * \
                    Xis
                dbeta[idx_nz, 1] -= (dresiduals[j, 1] - dresiduals_old[1]) * \
                    Xis

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
        n_samples = X.shape[0]
        alpha = self.residuals[0:n_samples] - \
            self.residuals[n_samples:(2 * n_samples)]
        full_supp = np.logical_and(alpha != 0, np.abs(alpha) != C)

        return X[full_supp, :] @ X[full_supp, :].T

    def get_dual_v(self, X, y, v, log_hyperparam):
        if v.shape[0] != 0:
            return X @ v
        else:
            return np.zeros(X.shape[0])

    def _get_jac_t_v(self, X, y, jac, mask, dense, hyperparam, v, n_samples):
        C = hyperparam[0]
        epsilon = hyperparam[1]
        alpha = self.residuals[0:n_samples] - \
            self.residuals[n_samples:(2 * n_samples)]
        full_supp = np.logical_and(alpha != 0, np.abs(alpha) != C)
        maskC = np.abs(alpha) == C
        hessian = X[full_supp, :] @ X[maskC, :].T
        hessian_vec = hessian @ alpha[maskC]
        jac_t_v = hessian_vec.T @ jac
        jac_t_v += alpha[maskC].T @ v[maskC]
        jac_t_v2 = epsilon * np.sign(alpha[full_supp]) @ jac
        return np.array([jac_t_v, jac_t_v2])

    def generalized_supp(self, X, v, log_hyperparam):
        n_samples = int(self.residuals.shape[0] / 2)
        C = np.exp(log_hyperparam[0])
        alpha = self.residuals[0:n_samples] - \
            self.residuals[n_samples:(2 * n_samples)]
        full_supp = np.logical_and(alpha != 0, np.abs(alpha) != C)
        return v[full_supp]

    def proj_hyperparam(self, X, y, log_hyperparam):
        if log_hyperparam[0] < -16.0:
            log_hyperparam[0] = -16.0
        elif log_hyperparam[0] > 2:
            log_hyperparam[0] = 5

        if log_hyperparam[1] < -16.0:
            log_hyperparam[1] = -16.0
        elif log_hyperparam[1] > 2:
            log_hyperparam[1] = 2

        return log_hyperparam

    def get_jac_obj(self, Xs, ys, n_samples, sign_beta,
                    dbeta, residuals, dresiduals, hyperparam):
        C = hyperparam[0]
        alpha = residuals[0:n_samples] - residuals[n_samples:(2 * n_samples)]
        dalpha = dresiduals[0:n_samples, 0] - \
            dresiduals[n_samples:(2 * n_samples), 0]

        maskC = np.abs(alpha) == C
        full_supp = np.logical_and(alpha != 0, np.abs(alpha) != C)

        alphaX = dalpha[full_supp].T @ Xs[full_supp, :]
        quadratic_term = alphaX.T @ alphaX

        linear_term = alphaX.T @ Xs[maskC, :].T @ alpha[maskC]
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
