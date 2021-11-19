import numpy as np
from numba import njit
from numpy.linalg import norm
from scipy.sparse import issparse
import scipy.sparse.linalg as slinalg
from scipy.sparse.linalg import LinearOperator

from sparse_ho.models.base import BaseModel
from sparse_ho.utils import ST, init_dbeta0_new_p


class WeightedLasso(BaseModel):
    r"""Linear Model trained with weighted L1 regularizer (aka weighted Lasso).

    The optimization objective for weighted Lasso is:

    ..math::

        ||y - Xw||^2_2 / (2 * n_samples) + \sum_i^{n_features} \alpha_i |wi|

    Parameters
    ----------
    estimator: instance of ``sklearn.base.BaseEstimator``
        An estimator that follows the scikit-learn API.
    """

    def __init__(self, estimator=None):
        self.estimator = estimator

    def _init_dbeta_ddual_var(self, X, y, mask0=None, jac0=None,
                              dense0=None, compute_jac=True):
        n_samples, n_features = X.shape
        dbeta = np.zeros((n_features, n_features))
        ddual_var = np.zeros((n_samples, n_features))
        if jac0 is not None:
            dbeta[np.ix_(mask0, mask0)] = jac0.copy()
            ddual_var[:, mask0] = - X[:, mask0] @ jac0
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
                dbeta_old = dbeta[j, :].copy()
            zj = beta[j] + dual_var @ X[:, j] / (L[j] * n_samples)
            beta[j:j+1] = ST(zj, alpha[j] / L[j])
            if compute_jac:
                dzj = dbeta[j, :] + X[:, j] @ ddual_var / (L[j] * n_samples)
                dbeta[j:j+1, :] = np.abs(np.sign(beta[j])) * dzj
                dbeta[j:j+1, j] -= alpha[j] * np.sign(beta[j]) / L[j]
                # update residuals
                ddual_var -= np.outer(X[:, j], (dbeta[j, :] - dbeta_old))
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
            # get non zero idices
            idx_nz = indices[indptr[j]:indptr[j+1]]
            ###########################################
            beta_old = beta[j]
            if compute_jac:
                dbeta_old = dbeta[j, :].copy()
            zj = beta[j] + dual_var[idx_nz] @ Xjs / (L[j] * n_samples)
            beta[j:j+1] = ST(zj, alphas[j] / L[j])
            if compute_jac:
                dzj = dbeta[j, :] + Xjs @ ddual_var[idx_nz, :] / \
                    (L[j] * n_samples)
                dbeta[j:j+1, :] = np.abs(np.sign(beta[j])) * dzj
                dbeta[j:j+1, j] -= alphas[j] * np.sign(beta[j]) / L[j]
                # update residuals
                ddual_var[idx_nz, :] -= np.outer(
                    Xjs, (dbeta[j, :] - dbeta_old))
            dual_var[idx_nz] -= Xjs * (beta[j] - beta_old)

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
    def _get_pobj(dual_var, X, beta, alphas, y=None):
        n_samples = dual_var.shape[0]
        return (
            norm(dual_var) ** 2 / (2 * n_samples) + norm(alphas * beta, 1))

    @staticmethod
    def _get_pobj0(dual_var, beta, alphas, y=None):
        n_samples = dual_var.shape[0]
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
    def _init_ddual_var(dbeta, X, y, sign_beta, alpha):
        return - X @ dbeta

    @staticmethod
    def _init_g_backward(jac_v0, n_features):
        if jac_v0 is None:
            return np.zeros(n_features)
        else:
            return jac_v0

    @staticmethod
    @njit
    def _update_only_jac(Xs, y, dual_var, dbeta, ddual_var, L,
                         alpha, sign_beta):
        n_samples, n_features = Xs.shape
        for j in range(n_features):
            dbeta_old = dbeta[j, :].copy()
            dbeta[j:j+1, :] = dbeta[j, :] + Xs[:, j] @ ddual_var / \
                (L[j] * n_samples)
            dbeta[j:j+1, j] -= alpha[j] * sign_beta[j] / L[j]
            # update residuals
            ddual_var -= np.outer(Xs[:, j], (dbeta[j, :] - dbeta_old))

    @staticmethod
    @njit
    def _update_only_jac_sparse(
            data, indptr, indices, y, n_samples, n_features, dbeta, dual_var,
            ddual_var, L, alpha, sign_beta):
        for j in range(n_features):
            # get the j-st column of X in sparse format
            Xjs = data[indptr[j]:indptr[j+1]]
            # get the non zero idices
            idx_nz = indices[indptr[j]:indptr[j+1]]
            # store old beta j for fast update
            dbeta_old = dbeta[j, :].copy()

            dbeta[j:j+1, :] += Xjs @ ddual_var[idx_nz] / (L[j] * n_samples)
            dbeta[j, j] -= alpha[j] * sign_beta[j] / L[j]
            ddual_var[idx_nz] -= np.outer(Xjs, (dbeta[j] - dbeta_old))

    # @njit
    @staticmethod
    def _reduce_alpha(alpha, mask):
        return alpha[mask]

    @staticmethod
    def get_full_jac_v(mask, jac_v, n_features):
        """TODO

        Parameters
        ----------
        mask: TODO
        jac_v: TODO
        n_features: int
            Number of features.
        """
        # MM sorry I don't get what this does
        # TODO n_features should be n_hyperparams, right ?
        res = np.zeros(n_features)
        res[mask] = jac_v
        return res

    @staticmethod
    def get_mask_jac_v(mask, jac_v):
        """TODO

        Parameters
        ----------
        mask: TODO
        jac_v: TODO
        """
        return jac_v[mask]

    @staticmethod
    def _get_grad(X, y, jac, mask, dense, alphas, v):
        size_supp = mask.sum()
        jac_t_v = np.zeros(size_supp)
        jac_t_v = alphas[mask] * np.sign(dense) * jac
        return jac_t_v

    def proj_hyperparam(self, X, y, log_alpha):
        """Project hyperparameter on an admissible range of values.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Design matrix.
        y: ndarray, shape (n_samples,)
            Observation vector.
        log_alpha: ndarray, shape (n_features,)
            Logarithm of hyperparameter.

        Returns
        -------
        log_alpha: ndarray, shape (n_features,)
            Logarithm of projected hyperparameter.
        """
        if not hasattr(self, "log_alpha_max"):
            alpha_max = np.max(np.abs(X.T @ y)) / X.shape[0]
            self.log_alpha_max = np.log(alpha_max)
        log_alpha = np.clip(log_alpha, self.log_alpha_max - 5,
                            self.log_alpha_max + np.log(0.9))
        return log_alpha

    @staticmethod
    def get_L(X):
        """Compute Lipschitz constant of datafit.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Design matrix.

        Returns
        -------
        L: float
            The Lipschitz constant.
        """
        if issparse(X):
            return slinalg.norm(X, axis=0) ** 2 / (X.shape[0])
        else:
            return norm(X, axis=0) ** 2 / (X.shape[0])

    @staticmethod
    def get_mat_vec(X, y, mask, dense, log_alpha):
        """Returns a LinearOperator computing the matrix vector product
        with the Hessian of datafit. It is necessary to avoid storing a
        potentially large matrix, and keep advantage of the sparsity of X.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Design matrix.
        y: ndarray, shape (n_samples,)
            Observation vector.
        mask: ndarray, shape (n_features,)
            Mask corresponding to non zero entries of beta.
        dense: ndarray, shape (mask.sum(),)
            Non zero entries of beta.
        log_alpha: ndarray
            Logarithm of hyperparameter.
        """
        X_m = X[:, mask]
        n_samples, size_supp = X_m.shape

        def mv(v):
            return X_m.T @ (X_m @ v) / n_samples
        return LinearOperator((size_supp, size_supp), matvec=mv)

    def _use_estimator(self, X, y, alpha, tol):
        self.estimator.set_params(tol=tol)
        self.estimator.weights = alpha
        self.estimator.fit(X, y)
        mask = self.estimator.coef_ != 0
        dense = (self.estimator.coef_)[mask]
        return mask, dense, None

    @staticmethod
    def reduce_X(X, mask):
        """Reduce design matrix to generalized support.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Design matrix.
        mask : ndarray, shape (n_features,)
            Generalized support.
        """
        return X[:, mask]

    @staticmethod
    def reduce_y(y, mask):
        """Reduce observation vector to generalized support.

        Parameters
        ----------
        y : ndarray, shape (n_samples,)
            Observation vector.
        mask : ndarray, shape (n_features,)  TODO shape n_samples right?
            Generalized support.
        """
        return y

    def sign(self, x, log_alpha):
        """Get sign of iterate.

        Parameters
        ----------
        x : ndarray, shape TODO
        log_alpha : ndarray, shape TODO
            Logarithm of hyperparameter.
        """
        return np.sign(x)

    def get_beta(self, X, y, mask, dense):
        """Return primal iterate.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Design matrix.
        y: ndarray, shape (n_samples,)
            Observation vector.
        mask: ndarray, shape (n_features,)
            Mask corresponding to non zero entries of beta.
        dense: ndarray, shape (mask.sum(),)
            Non zero entries of beta.
        """
        # TODO what's the use of this function? it does nothing for all models
        return mask, dense

    def get_jac_v(self, X, y, mask, dense, jac, v):
        """Compute hypergradient.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Design matrix.
        y: ndarray, shape (n_samples,)
            Observation vector.
        mask: ndarray, shape (n_features,)
            Mask corresponding to non zero entries of beta.
        dense: ndarray, shape (mask.sum(),)
            Non zero entries of beta.
        jac: TODO
        v: TODO
        """
        # TODO this is the same for Lasso, Enet, Wlasso. Maybe inherit from
        # a common class, LinearModelPrimal or something?
        return jac.T @ v(mask, dense)

    def generalized_supp(self, X, v, log_alpha):
        """Generalized support of iterate.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Design matrix.
        v : TODO
        log_alpha : float
            Log of hyperparameter.

        Returns
        -------
        TODO
        """
        return v

    def get_jac_residual_norm(self, Xs, ys, n_samples, sign_beta,
                              dbeta, dual_var, ddual_var, alpha):
        return(
            norm(ddual_var.T @ ddual_var +
                 n_samples * alpha * sign_beta @ dbeta))
