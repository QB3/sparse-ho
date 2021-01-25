import numpy as np
from numpy.linalg import norm
import scipy.sparse.linalg as slinalg

from numba import njit

from sparse_ho.models.base import BaseModel
from sparse_ho.utils import ST, init_dbeta0_new_p


class WeightedLasso(BaseModel):
    r"""Linear Model trained with weighted L1 regularizer (aka weighted Lasso)

    The optimization objective for weighted Lasso is:

    ..math::

        ||y - Xw||^2_2 / (2 * n_samples) + \sum_i^{n_features} \alpha_i |wi|

    Parameters
    ----------
    X: {ndarray, sparse matrix} of (n_samples, n_features)
        Data.
    y: {ndarray, sparse matrix} of (n_samples)
        Target
    estimator: instance of ``sklearn.base.BaseEstimator``
        An estimator that follows the scikit-learn API.
    log_alpha_max: float
        logarithm of alpha_max if already precomputed
    """

    def __init__(
            self, max_iter=1000, estimator=None, log_alpha_max=None):
        self.max_iter = max_iter
        self.estimator = estimator
        self.log_alpha_max = log_alpha_max

    def _init_dbeta_dresiduals(self, X, y, mask0=None, jac0=None,
                       dense0=None, compute_jac=True):
        n_samples, n_features = X.shape
        dbeta = np.zeros((n_features, n_features))
        dresiduals = np.zeros((n_samples, n_features))
        if jac0 is not None:
            dbeta[np.ix_(mask0, mask0)] = jac0.copy()
            dresiduals[:, mask0] = - X[:, mask0] @ jac0
        return dbeta, dresiduals

    def _init_beta_residuals(self, X, y, mask0=None, dense0=None):
        beta = np.zeros(X.shape[1])
        if dense0 is None or len(dense0) == 0:
            residuals = y.copy()
            residuals = residuals.astype(np.float)
        else:
            beta[mask0] = dense0.copy()
            residuals = y - X[:, mask0] @ dense0
        return beta, residuals

    @staticmethod
    @njit
    def _update_beta_jac_bcd(
            X, y, beta, dbeta, residuals, dresiduals, alpha, L, compute_jac=True):
        n_samples, n_features = X.shape
        non_zeros = np.where(L != 0)[0]

        for j in non_zeros:
            beta_old = beta[j]
            if compute_jac:
                dbeta_old = dbeta[j, :].copy()
            zj = beta[j] + residuals @ X[:, j] / (L[j] * n_samples)
            beta[j:j+1] = ST(zj, alpha[j] / L[j])
            if compute_jac:
                dzj = dbeta[j, :] + X[:, j] @ dresiduals / (L[j] * n_samples)
                dbeta[j:j+1, :] = np.abs(np.sign(beta[j])) * dzj
                dbeta[j:j+1, j] -= alpha[j] * np.sign(beta[j]) / L[j]
                # update residuals
                dresiduals -= np.outer(X[:, j], (dbeta[j, :] - dbeta_old))
            residuals -= X[:, j] * (beta[j] - beta_old)

    @staticmethod
    @njit
    def _update_beta_jac_bcd_sparse(
            data, indptr, indices, y, n_samples, n_features, beta,
            dbeta, residuals, dresiduals, alphas, L, compute_jac=True):
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
            zj = beta[j] + residuals[idx_nz] @ Xjs / (L[j] * n_samples)
            beta[j:j+1] = ST(zj, alphas[j] / L[j])
            if compute_jac:
                dzj = dbeta[j, :] + Xjs @ dresiduals[idx_nz, :] / (L[j] * n_samples)
                dbeta[j:j+1, :] = np.abs(np.sign(beta[j])) * dzj
                dbeta[j:j+1, j] -= alphas[j] * np.sign(beta[j]) / L[j]
                # update residuals
                dresiduals[idx_nz, :] -= np.outer(Xjs, (dbeta[j, :] - dbeta_old))
            residuals[idx_nz] -= Xjs * (beta[j] - beta_old)

    @staticmethod
    @njit
    def _update_bcd_jac_backward(
            X, alpha, jac_t_v, beta, v_, L):
        n_samples, n_features = X.shape
        sign_beta = np.sign(beta)
        for j in (np.arange(sign_beta.shape[0] - 1, -1, -1)):
            jac_t_v[j] = jac_t_v[j] - (v_[j]) * alpha[j] * sign_beta[j] / L[j]
            v_[j] *= np.abs(sign_beta[j])
            v_ -= v_[j] / (L[j] * n_samples) * X[:, j] @ X
        return jac_t_v

    @staticmethod
    def _get_pobj(residuals, X, beta, alphas, y=None):
        n_samples = residuals.shape[0]
        return (
            norm(residuals) ** 2 / (2 * n_samples) + norm(alphas * beta, 1))

    @staticmethod
    def _get_pobj0(residuals, beta, alphas, y=None):
        n_samples = residuals.shape[0]
        return norm(y) ** 2 / (2 * n_samples)

    @staticmethod
    def _get_jac(dbeta, mask):
        return dbeta[np.ix_(mask, mask)]

    @staticmethod
    def _init_dbeta0(mask, mask0, jac0):
        size_mat = mask.sum()
        if jac0 is None:
            dbeta0_new = np.zeros((size_mat, size_mat))
        else:
            dbeta0_new = init_dbeta0_new_p(jac0, mask, mask0)
        return dbeta0_new

    @staticmethod
    def _init_dbeta(n_features):
        dbeta = np.zeros((n_features, n_features))
        return dbeta

    @staticmethod
    def _init_dresiduals(dbeta, X, y, sign_beta, alpha):
        return - X @ dbeta

    @staticmethod
    def _init_g_backward(jac_v0, n_features):
        if jac_v0 is None:
            return np.zeros(n_features)
        else:
            return jac_v0

    @staticmethod
    @njit
    def _update_only_jac(Xs, y, residuals, dbeta, dresiduals, L, alpha, sign_beta):
        n_samples, n_features = Xs.shape
        for j in range(n_features):
            dbeta_old = dbeta[j, :].copy()
            dbeta[j:j+1, :] = dbeta[j, :] + Xs[:, j] @ dresiduals / (L[j] * n_samples)
            dbeta[j:j+1, j] -= alpha[j] * sign_beta[j] / L[j]
            # update residuals
            dresiduals -= np.outer(Xs[:, j], (dbeta[j, :] - dbeta_old))

    @staticmethod
    @njit
    def _update_only_jac_sparse(
            data, indptr, indices, y, n_samples, n_features, dbeta, residuals,
            dresiduals, L, alpha, sign_beta):
        for j in range(n_features):
            # get the j-st column of X in sparse format
            Xjs = data[indptr[j]:indptr[j+1]]
            # get the non zero idices
            idx_nz = indices[indptr[j]:indptr[j+1]]
            # store old beta j for fast update
            dbeta_old = dbeta[j, :].copy()

            dbeta[j:j+1, :] += Xjs @ dresiduals[idx_nz] / (L[j] * n_samples)
            dbeta[j, j] -= alpha[j] * sign_beta[j] / L[j]
            dresiduals[idx_nz] -= np.outer(Xjs, (dbeta[j] - dbeta_old))

    # @njit
    @staticmethod
    def _reduce_alpha(alpha, mask):
        return alpha[mask]

    @staticmethod
    def get_full_jac_v(mask, jac_v, n_features):
        res = np.zeros(n_features)
        res[mask] = jac_v
        return res

    @staticmethod
    def get_mask_jac_v(mask, jac_v):
        return jac_v[mask]

    @staticmethod
    def _get_jac_t_v(X, y, jac, mask, dense, alphas, v, n_samples):
        size_supp = mask.sum()
        jac_t_v = np.zeros(size_supp)
        jac_t_v = n_samples * alphas[mask] * np.sign(dense) * jac
        return jac_t_v

    def proj_hyperparam(self, X, y, log_alpha):
        """Maybe we could do this in place.
        """
        if self.log_alpha_max is None:
            alpha_max = np.max(np.abs(X.T @ y))
            alpha_max /= X.shape[0]
            self.log_alpha_max = np.log(alpha_max)
        proj_log_alpha = log_alpha.copy()
        proj_log_alpha[proj_log_alpha < -12] = -12
        if np.max(proj_log_alpha > self.log_alpha_max):
            proj_log_alpha[
                proj_log_alpha > self.log_alpha_max] = self.log_alpha_max
        return proj_log_alpha

    @staticmethod
    def get_L(X, is_sparse=False):
        if is_sparse:
            return slinalg.norm(X, axis=0) ** 2 / (X.shape[0])
        else:
            return norm(X, axis=0) ** 2 / (X.shape[0])

    @staticmethod
    def get_hessian(X, y, mask, dense, log_alpha):
        X_m = X[:, mask]
        hessian = X_m.T @ X_m
        return hessian

    def _use_estimator(self, X, y, alpha, tol, max_iter):
        self.estimator.set_params(tol=tol)
        self.estimator.weights = alpha
        self.estimator.fit(X, y)
        mask = self.estimator.coef_ != 0
        dense = (self.estimator.coef_)[mask]
        return mask, dense, None

    @staticmethod
    def reduce_X(X, mask):
        return X[:, mask]

    @staticmethod
    def reduce_y(y, mask):
        return y

    def sign(self, x, log_alpha):
        return np.sign(x)

    def get_beta(self, X, y, mask, dense):
        return mask, dense

    def get_jac_v(self, X, y, mask, dense, jac, v):
        return jac.T @ v(mask, dense)

    def generalized_supp(self, X, v, log_alpha):
        return v

    def get_jac_obj(self, Xs, ys, n_samples, sign_beta, dbeta, residuals, 
                    dresiduals, alpha):
        return(
            norm(dresiduals.T @ dresiduals + n_samples * alpha * sign_beta @ dbeta))
