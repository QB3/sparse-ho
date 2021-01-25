import numpy as np
from numpy.linalg import norm
import scipy.sparse.linalg as slinalg
from scipy.sparse import issparse

from numba import njit

from sparse_ho.models.base import BaseModel
from sparse_ho.utils import proj_box_svm, ind_box
from sparse_ho.utils import init_dbeta0_new


class SVM(BaseModel):
    """The support vector Machine classifier without bias
    The optimization problem is solved in the dual:
    1/2 r^T(y * X)(y * X)^T r - sum_i^n r_i
    s.t 0 <= r_i <= C

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

    def __init__(self, estimator=None, max_iter=100):
        self.estimator = estimator
        self.max_iter = max_iter
        self.dual = True
        self.dresiduals = None

    def _init_dbeta_dresiduals(self, X, y, dense0=None,
                       mask0=None, jac0=None, compute_jac=True):
        n_samples, n_features = X.shape
        dresiduals = np.ones(n_samples)
        if self.dresiduals is not None and self.dresiduals.shape[0] == n_samples:
            dresiduals = self.dresiduals.copy()
        if issparse(X):
            dbeta = (X.T).multiply(y * dresiduals)
            dbeta = np.sum(dbeta, axis=1)
            dbeta = np.squeeze(np.array(dbeta))
        else:
            dbeta = np.sum(y * dresiduals * X.T, axis=1)
        return dbeta, dresiduals

    def _init_beta_residuals(self, X, y, mask0, dense0):
        n_samples, n_features = X.shape
        residuals = np.zeros(n_samples)
        if mask0 is None or self.residuals is None:
            beta = np.zeros(n_features)
        else:
            residuals = self.residuals
            if issparse(X):
                beta = (X.T).multiply(y * residuals)
                beta = np.sum(beta, axis=1)
                beta = np.squeeze(np.array(beta))
            else:
                beta = np.sum(y * residuals * X.T, axis=1)
        return beta, residuals

    @staticmethod
    @njit
    def _update_beta_jac_bcd(
            X, y, beta, dbeta, residuals, dresiduals, C, L, compute_jac=True):
        """
            beta : dual variable of the svm
            r : primal used for cheap updates
            dbeta : jacobian of the dual variables
            dr : jacobian of the primal variable
        """
        C = C[0]
        n_samples = X.shape[0]
        for j in range(n_samples):
            F = y[j] * np.sum(beta * X[j, :]) - 1.0
            residuals_old = residuals[j]
            zj = residuals[j] - F / L[j]
            residuals[j] = proj_box_svm(zj, C)
            beta += (residuals[j] - residuals_old) * y[j] * X[j, :]
            if compute_jac:
                dF = y[j] * np.sum(dbeta * X[j, :])
                dresiduals_old = dresiduals[j]
                dzj = dresiduals[j] - dF / L[j]
                dresiduals[j] = ind_box(zj, C) * dzj
                dresiduals[j] += C * (C <= zj)
                dbeta += (dresiduals[j] - dresiduals_old) * y[j] * X[j, :]

    @staticmethod
    @njit
    def _update_beta_jac_bcd_sparse(
            data, indptr, indices, y, n_samples, n_features, beta,
            dbeta, residuals, dresiduals, C, L, compute_jac=True):
        # data needs to be a row sparse matrix
        C = C[0]
        for j in range(n_samples):
            # get the i-st row of X in sparse format
            Xis = data[indptr[j]:indptr[j+1]]
            # get the non zero indices
            idx_nz = indices[indptr[j]:indptr[j+1]]
            # Compute the gradient
            F = y[j] * np.sum(beta[idx_nz] * Xis) - 1.0
            residuals_old = residuals[j]
            zj = residuals[j] - F / L[j]
            residuals[j] = proj_box_svm(zj, C)
            beta[idx_nz] += (residuals[j] - residuals_old) * y[j] * Xis
            if compute_jac:
                dF = y[j] * np.sum(dbeta[idx_nz] * Xis)
                dresiduals_old = dresiduals[j]
                dzj = dresiduals[j] - dF / L[j]
                dresiduals[j] = ind_box(zj, C) * dzj
                dresiduals[j] += C * (C <= zj)
                dbeta[idx_nz] += (dresiduals[j] - dresiduals_old) * y[j] * Xis

    @staticmethod
    def _get_pobj0(residuals, beta, C, y):
        C = C[0]
        n_samples = residuals.shape[0]
        obj_prim = C * np.sum(np.maximum(
            np.ones(n_samples), np.zeros(n_samples)))
        return obj_prim

    @staticmethod
    def _get_pobj(residuals, X, beta, C, y):
        C = C[0]
        n_samples = X.shape[0]
        obj_prim = 0.5 * norm(beta) ** 2 + C * np.sum(np.maximum(
            np.ones(n_samples) - (X @ beta) * y, np.zeros(n_samples)))
        obj_dual = 0.5 * beta.T @ beta - np.sum(residuals)
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

    def _init_dbeta(self, n_features):
        if self.dbeta is not None:
            return self.dbeta
        else:
            return np.zeros(n_features)

    def _init_dresiduals(self, dbeta, X, y, sign_beta, C):
        is_sparse = issparse(X)
        sign = np.zeros(self.residuals.shape[0])
        sign[self.residuals == 0.0] = -1.0
        sign[self.residuals == C] = 1.0
        dresiduals = np.zeros(X.shape[0])
        self.dresiduals = dresiduals
        if np.sum(sign == 1.0) != 0:
            dresiduals[sign == 1.0] = np.repeat(C, (sign == 1).sum())
        if is_sparse:
            self.dbeta = np.array(np.sum(X.T.multiply(y * dresiduals), axis=1))
        else:
            self.dbeta = np.sum(y * dresiduals * X.T, axis=1)
        return dresiduals

    @staticmethod
    @njit
    def _update_only_jac(Xs, ys, residuals, dbeta, dresiduals, L, C, sign_beta):
        sign = np.zeros(residuals.shape[0])
        sign[residuals == 0.0] = -1.0
        sign[residuals == C] = 1.0
        for j in np.arange(0, Xs.shape[0])[sign == 0.0]:
            dF = ys[j] * np.sum(dbeta * Xs[j, :])
            dresiduals_old = dresiduals[j]
            dzj = dresiduals[j] - (dF / L[j])
            dresiduals[j] = dzj
            dbeta += (dresiduals[j] - dresiduals_old) * ys[j] * Xs[j, :]

    @staticmethod
    @njit
    def _update_only_jac_sparse(
            data, indptr, indices, y, n_samples, n_features,
            dbeta, residuals, dresiduals, L, C, sign_beta):
        sign = np.zeros(residuals.shape[0])
        sign[residuals == 0.0] = -1.0
        sign[residuals == C] = 1.0
        for j in np.arange(0, n_samples)[sign == 0.0]:
            # get the i-st row of X in sparse format
            Xis = data[indptr[j]:indptr[j+1]]
            # get the non zero idices
            idx_nz = indices[indptr[j]:indptr[j+1]]
            # store old beta j for fast update
            dF = y[j] * np.sum(dbeta[idx_nz] * Xis)
            dresiduals_old = dresiduals[j]
            dzj = dresiduals[j] - (dF / L[j])
            dresiduals[j] = dzj
            dbeta[idx_nz] += ((dresiduals[j] - dresiduals_old) * y[j] * Xis)

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

    def sign(self, x, log_C):
        sign = np.zeros(x.shape[0])
        sign[np.isclose(x, 0.0)] = -1.0
        sign[np.isclose(x, np.exp(log_C))] = 1.0
        return sign

    def get_dual_v(self, X, y, v, log_C):
        if issparse(X):
            v_dual = v * (X.T).multiply(y)
            v_dual = np.sum(v_dual, axis=1)
            v_dual = np.squeeze(np.array(v_dual))
        else:
            v_dual = (y * X.T).T @ v
        return v_dual

    @staticmethod
    def get_jac_v(X, y, mask, dense, jac, v):
        return jac[mask].T @ v(mask, dense)

    @staticmethod
    def get_full_jac_v(mask, jac_v, n_features):
        return jac_v

    def get_hessian(self, X, y, mask, dense, log_C):
        C = np.exp(log_C)
        full_supp = np.logical_and(self.residuals != 0, self.residuals != C)
        if issparse(X):
            Xy = X[full_supp, :].multiply(y[full_supp, np.newaxis])
            return Xy @ Xy.T
        else:
            Xy = (y[full_supp] * X[full_supp, :].T)
            return Xy.T @ Xy

    def _get_jac_t_v(self, X, y, jac, mask, dense, C, v, n_samples):
        C = C[0]
        full_supp = np.logical_and(self.residuals != 0, self.residuals != C)
        maskC = self.residuals == C
        hessian = (y[full_supp] * X[full_supp, :].T).T @ \
            (y[maskC] * X[maskC, :].T)
        hessian_vec = hessian @ np.repeat(C, maskC.sum())
        jac_t_v = hessian_vec.T @ jac
        jac_t_v += np.repeat(C, maskC.sum()).T @ v[maskC]
        return jac_t_v

    def generalized_supp(self, X, v, log_C):
        full_supp = np.logical_and(self.residuals != 0, self.residuals != np.exp(log_C))
        return v[full_supp]

    def proj_hyperparam(self, X, y, log_alpha):
        if log_alpha < -16.0:
            log_alpha = -16.0
        elif log_alpha > 4:
            log_alpha = 4
        return log_alpha

    def get_jac_obj(self, Xs, ys, n_samples, sign_beta, dbeta, residuals,
                    dresiduals, C):
        maskC = residuals == C
        full_supp = np.logical_and(residuals != 0, residuals != C)
        if issparse(Xs):
            dryX = dresiduals[full_supp].T @ \
                (Xs[full_supp, :].T).multiply(ys[full_supp]).T
        else:
            dryX = dresiduals[full_supp].T @ (ys[full_supp] * Xs[full_supp, :].T).T
        quadratic_term = dryX.T @ dryX
        if maskC.sum() != 0:
            if issparse(Xs):
                linear_term = dryX.T @ (Xs[maskC, :].T).multiply(ys[maskC]) @ \
                    residuals[maskC]
            else:
                linear_term = dryX.T @ (ys[maskC] * Xs[maskC, :].T) @ residuals[maskC]
        else:
            linear_term = 0
        res = quadratic_term + linear_term
        return norm(res)

    def _use_estimator(self, X, y, C, tol, max_iter):
        if self.estimator is None:
            raise ValueError("You did not pass a solver with sklearn API")
        self.estimator.set_params(
            tol=tol, C=C,
            fit_intercept=False, max_iter=max_iter)
        self.estimator.fit(X, y)
        mask = self.estimator.coef_ != 0
        dense = self.estimator.coef_[mask]
        return mask, dense, None
