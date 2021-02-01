import numpy as np
from numpy.linalg import norm
import scipy.sparse.linalg as slinalg
from numba import njit

from sparse_ho.utils import init_dbeta0_new, ST
from sparse_ho.utils import sparse_scalar_product
from sparse_ho.models.base import BaseModel


class Lasso(BaseModel):
    """Linear Model trained with L1 prior as regularizer (aka the Lasso)
    The optimization objective for Lasso is:
    (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

    Parameters
    ----------
    log_alpha : float
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

    def _init_dbeta_ddual_var(self, X, y, mask0=None, jac0=None,
                              dense0=None, compute_jac=True):
        n_samples, n_features = X.shape
        dbeta = np.zeros(n_features)
        if jac0 is None or not compute_jac:
            ddual_var = np.zeros(n_samples)
        else:
            dbeta[mask0] = jac0.copy()
            ddual_var = - X[:, mask0] @ jac0.copy()
        return dbeta, ddual_var

    def _init_beta_dual_var(self, X, y, mask0=None, dense0=None):
        beta = np.zeros(X.shape[1])
        if dense0 is None or len(dense0) == 0:
            dual_var = y.copy()
            dual_var = dual_var.astype(np.float)
        else:
            beta[mask0] = dense0.copy()
            dual_var = y - X[:, mask0] @ dense0
        return beta, dual_var

    @staticmethod
    @njit
    def _update_beta_jac_bcd(
            X, y, beta, dbeta, dual_var, ddual_var,
            alpha, L, compute_jac=True):
        n_samples, n_features = X.shape
        non_zeros = np.where(L != 0)[0]

        for j in non_zeros:
            beta_old = beta[j]
            if compute_jac:
                dbeta_old = dbeta[j]
                # compute derivatives
            zj = beta[j] + dual_var @ X[:, j] / (L[j] * n_samples)
            beta[j] = ST(zj, alpha[j] / L[j])
            # beta[j:j+1] = ST(zj, alpha[j] / L[j])
            if compute_jac:
                dzj = dbeta[j] + X[:, j] @ ddual_var / (L[j] * n_samples)
                dbeta[j:j+1] = np.abs(np.sign(beta[j])) * dzj
                dbeta[j:j+1] -= alpha[j] * np.sign(beta[j]) / L[j]
                # update residuals
                ddual_var -= X[:, j] * (dbeta[j] - dbeta_old)
            dual_var -= X[:, j] * (beta[j] - beta_old)

    @staticmethod
    @njit
    def _update_beta_jac_bcd_sparse(
            data, indptr, indices, y, n_samples, n_features, beta,
            dbeta, dual_var, ddual_var, alphas, L, compute_jac=True):

        non_zeros = np.where(L != 0)[0]

        for j in non_zeros:
            # get the j-st column of X in sparse format
            Xjs = data[indptr[j]:indptr[j+1]]
            # get the non zero indices
            idx_nz = indices[indptr[j]:indptr[j+1]]
            beta_old = beta[j]
            if compute_jac:
                dbeta_old = dbeta[j]
            zj = beta[j] + dual_var[idx_nz] @ Xjs / (L[j] * n_samples)
            beta[j:j+1] = ST(zj, alphas[j] / L[j])
            if compute_jac:
                dzj = dbeta[j] + Xjs @ ddual_var[idx_nz] / (L[j] * n_samples)
                dbeta[j:j+1] = np.abs(np.sign(beta[j])) * dzj
                dbeta[j:j+1] -= alphas[j] * np.sign(beta[j]) / L[j]
                # update residuals
                ddual_var[idx_nz] -= Xjs * (dbeta[j] - dbeta_old)
            dual_var[idx_nz] -= Xjs * (beta[j] - beta_old)

    @staticmethod
    @njit
    def _update_bcd_jac_backward(X, alpha, grad, beta, v_t_jac, L):
        sign_beta = np.sign(beta)
        n_samples, n_features = X.shape
        for j in (np.arange(sign_beta.shape[0] - 1, -1, -1)):
            grad -= (v_t_jac[j]) * alpha * sign_beta[j] / L[j]
            v_t_jac[j] *= np.abs(sign_beta[j])
            v_t_jac -= v_t_jac[j] / (L[j] * n_samples) * X[:, j] @ X

        return grad

    @staticmethod
    @njit
    def _update_bcd_jac_backward_sparse(
            data, indptr, indices, n_samples, n_features,
            alpha, grad, beta, v_t_jac, L):
        sign_beta = np.sign(beta)
        for j in (np.arange(sign_beta.shape[0] - 1, -1, -1)):
            if L[j] != 0:
                Xjs = data[indptr[j]:indptr[j+1]]
                idx_nz = indices[indptr[j]:indptr[j+1]]
                grad -= (v_t_jac[j]) * alpha * sign_beta[j] / L[j]
                v_t_jac[j] *= np.abs(sign_beta[j])
                cste = v_t_jac[j] / (L[j] * n_samples)
                for i in (np.arange(sign_beta.shape[0] - 1, -1, -1)):
                    Xis = data[indptr[i]:indptr[i+1]]
                    idx = indices[indptr[i]:indptr[i+1]]
                    product = sparse_scalar_product(Xjs, idx_nz, Xis, idx)
                    v_t_jac[i] -= cste * product

        return grad

    @staticmethod
    def _get_pobj0(dual_var, beta, alphas, y=None):
        n_samples = dual_var.shape[0]
        return norm(y) ** 2 / (2 * n_samples)

    @staticmethod
    def _get_pobj(dual_var, X, beta, alphas, y=None):
        n_samples = dual_var.shape[0]
        return (
            norm(dual_var) ** 2 / (2 * n_samples) +
            np.abs(alphas * beta).sum())

    @staticmethod
    def _get_dobj(dual_var, X, beta, alpha, y=None):
        # the dual variable is theta = (y - X beta) / (alpha n_samples)
        n_samples = X.shape[0]
        theta = dual_var / (alpha * n_samples)
        norm_inf_XTtheta = np.max(np.abs(X.T @ theta))
        if norm_inf_XTtheta > 1:
            theta /= norm_inf_XTtheta
        dobj = alpha * y @ theta
        dobj -= alpha ** 2 * n_samples / 2 * (theta ** 2).sum()
        return dobj

    @staticmethod
    def _get_jac(dbeta, mask):
        return dbeta[mask]

    @staticmethod
    def get_full_jac_v(mask, jac_v, n_features):
        return jac_v

    @staticmethod
    def get_mask_jac_v(mask, jac_v):
        return jac_v

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
    def _init_ddual_var(dbeta, X, y, sign_beta, alpha):
        return - X @ dbeta

    @staticmethod
    def _init_g_backward(jac_v0, n_features):
        if jac_v0 is None:
            return 0.0
        else:
            return jac_v0

    @staticmethod
    @njit
    def _update_only_jac(Xs, y, dual_var, dbeta, ddual_var,
                         L, alpha, sign_beta):
        n_samples, n_features = Xs.shape
        for j in range(n_features):
            # dbeta_old = dbeta[j].copy()
            dbeta_old = dbeta[j]
            dbeta[j] += Xs[:, j].T @ ddual_var / (L[j] * n_samples)
            dbeta[j] -= alpha * sign_beta[j] / L[j]
            ddual_var -= Xs[:, j] * (dbeta[j] - dbeta_old)

    @staticmethod
    @njit
    def _update_only_jac_sparse(
            data, indptr, indices, y, n_samples, n_features,
            dbeta, dual_var, ddual_var, L, alpha, sign_beta):
        for j in range(n_features):
            # get the j-st column of X in sparse format
            Xjs = data[indptr[j]:indptr[j+1]]
            # get the non zero idices
            idx_nz = indices[indptr[j]:indptr[j+1]]
            # store old beta j for fast update
            dbeta_old = dbeta[j]
            # update of the Jacobian dbeta
            dbeta[j] += Xjs @ ddual_var[idx_nz] / (L[j] * n_samples)
            dbeta[j] -= alpha * sign_beta[j] / L[j]
            ddual_var[idx_nz] -= Xjs * (dbeta[j] - dbeta_old)

    @staticmethod
    @njit
    def _reduce_alpha(alpha, mask):
        return alpha

    @staticmethod
    def _get_jac_t_v(X, y, jac, mask, dense, alphas, v, n_samples):
        return n_samples * alphas[mask] * np.sign(dense) @ jac

    def proj_hyperparam(self, X, y, log_alpha):
        if self.log_alpha_max is None:
            alpha_max = np.max(np.abs(X.T @ y))
            alpha_max /= X.shape[0]
            self.log_alpha_max = np.log(alpha_max)
        if log_alpha < self.log_alpha_max - 12:
            return self.log_alpha_max - 12
        elif log_alpha > self.log_alpha_max + np.log(0.9):
            return self.log_alpha_max + np.log(0.9)
        else:
            return log_alpha

    @staticmethod
    def get_L(X, is_sparse=False):
        # print(is_sparse)
        if is_sparse:
            return slinalg.norm(X, axis=0) ** 2 / (X.shape[0])
        else:
            return norm(X, axis=0) ** 2 / (X.shape[0])

    def _use_estimator(self, X, y, alpha, tol, max_iter):
        if self.estimator is None:
            raise ValueError("You did not pass a solver with sklearn API")
        self.estimator.set_params(tol=tol, alpha=alpha)
        self.estimator.fit(X, y)
        mask = self.estimator.coef_ != 0
        dense = self.estimator.coef_[mask]
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

    @staticmethod
    def get_hessian(X_train, y_train, mask, dense, log_alpha):
        X_m = X_train[:, mask]
        hessian = X_m.T @ X_m
        return hessian

    def generalized_supp(self, X, v, log_alpha):
        return v

    def compute_alpha_max(self):
        if self.log_alpha_max is None:
            alpha_max = np.max(np.abs(self.X.T @ self.y))
            alpha_max /= self.X.shape[0]
            self.log_alpha_max = np.log(alpha_max)
        return self.log_alpha_max

    def get_jac_obj(self, Xs, ys, n_samples, sign_beta, dbeta,
                    dual_var, ddual_var, alpha):
        return norm(ddual_var.T @ ddual_var +
                    n_samples * alpha * sign_beta @ dbeta)
