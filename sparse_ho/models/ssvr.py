import numpy as np
from numpy.core.fromnumeric import size
from numpy.linalg import norm
import scipy.sparse.linalg as slinalg
from numba import njit
from scipy.sparse.linalg import LinearOperator
from sparse_ho.models.base import BaseModel
from sparse_ho.utils import proj_box_svm, ind_box
from scipy.sparse import issparse


@njit
def _compute_jac_aux(X, epsilon, dbeta, ddual_var, zj, L, C, j1, j2, sign):
    dF = sign * np.array([np.sum(dbeta[:, 0].T * X[j1, :]),
                           np.sum(dbeta[:, 1].T * X[j1, :])])
    ddual_var_old = ddual_var[j2, :].copy()
    dzj = ddual_var[j2, :] - dF / L[j1]
    ddual_var[j2, :] = ind_box(zj, C) * dzj
    ddual_var[j2, 0] += C * (C <= zj)
    ddual_var[j2, 1] -= epsilon * ind_box(zj, C) / L[j1]

    dbeta[:, 0] += sign * (ddual_var[j2, 0] -
                           ddual_var_old[0]) * X[j1, :]
    dbeta[:, 1] += sign * (ddual_var[j2, 1] -
                           ddual_var_old[1]) * X[j1, :]


@njit
def _update_beta_jac_bcd_aux(X, y, epsilon, beta, dbeta, dual_var, ddual_var,
                             L, C, j1, j2, sign, compute_jac):
    n_samples = X.shape[0]
    F = sign * np.sum(beta * X[j1, :]) + epsilon - sign * y[j1]
    dual_var_old = dual_var[j2]
    zj = dual_var[j2] - F / L[j1]
    dual_var[j2] = proj_box_svm(zj, C / n_samples)
    beta += sign * ((dual_var[j2] - dual_var_old) * X[j1, :])
    if compute_jac:
        _compute_jac_aux(X, epsilon, dbeta, ddual_var, zj, L, C / n_samples, j1, j2, sign)


@njit
def _compute_jac_aux_sparse(
        data, indptr, indices, epsilon, dbeta, ddual_var, zj, L, C,
        j1, j2, sign):
    # get the i-st row of X in sparse format
    Xjs = data[indptr[j1]:indptr[j1+1]]
    # get the non zero indices
    idx_nz = indices[indptr[j1]:indptr[j1+1]]

    dF = sign * np.array([np.sum(dbeta[idx_nz, 0].T * Xjs),
                          np.sum(dbeta[idx_nz, 1].T * Xjs)])
    ddual_var_old = ddual_var[j2, :].copy()
    dzj = ddual_var[j2, :] - dF / L[j1]
    ddual_var[j2, :] = ind_box(zj, C) * dzj
    ddual_var[j2, 0] += C * (C <= zj)
    ddual_var[j2, 1] -= epsilon * ind_box(zj, C) / L[j1]

    dbeta[idx_nz, 0] += sign * (ddual_var[j2, 0] -
                                ddual_var_old[0]) * Xjs
    dbeta[idx_nz, 1] += sign * (ddual_var[j2, 1] -
                                ddual_var_old[1]) * Xjs


@njit
def _update_beta_jac_bcd_aux_sparse(data, indptr, indices, y, epsilon, beta,
                                    dbeta, dual_var, ddual_var,
                                    L, C, j1, j2, sign, compute_jac):
    n_samples = (dual_var.shape[0] - beta.shape[0] - 1) // 2
    # get the i-st row of X in sparse format
    Xjs = data[indptr[j1]:indptr[j1+1]]
    # get the non zero indices
    idx_nz = indices[indptr[j1]:indptr[j1+1]]

    F = sign * np.sum(beta[idx_nz] * Xjs) + epsilon - sign * y[j1]
    dual_var_old = dual_var[j2]
    zj = dual_var[j2] - F / L[j1]
    dual_var[j2] = proj_box_svm(zj, C / n_samples)
    beta[idx_nz] += sign * (dual_var[j2] - dual_var_old) * Xjs
    if compute_jac:
        _compute_jac_aux_sparse(
            data, indptr, indices, epsilon, dbeta, ddual_var, zj,
            L, C / n_samples, j1, j2, sign)


class SimplexSVR(BaseModel):
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

    def __init__(self, estimator=None):
        self.estimator = estimator
        self.dual = True
        self.dual_var = None
        self.ddual_var = None

    def _init_dbeta_ddual_var(self, X, y, dense0=None,
                              mask0=None, jac0=None, compute_jac=True):
        n_samples, n_features = X.shape
        self.n_samples = n_samples
        ddual_var = np.zeros((2 * n_samples + n_features + 1, 2))

        if jac0 is None or not compute_jac or self.ddual_var is None:
            dbeta = np.zeros((n_features, 2))
        else:
            if self.ddual_var.shape[0] != 2 * n_samples + n_features + 1:
                dbeta = np.zeros((n_features, 2))
            else:
                ddual_var = self.ddual_var
                dbeta = X.T @ (
                    ddual_var[0:n_samples, :] -
                    ddual_var[n_samples:(2 * n_samples), :])
                dbeta += ddual_var[(2 * n_samples):
                                   (2 * n_samples + n_features), :]
                dbeta += ddual_var[-1, :]
        return dbeta, ddual_var

    def _init_beta_dual_var(self, X, y, mask0, dense0):
        n_samples, n_features = X.shape
        dual_var = np.zeros(2 * n_samples + n_features + 1)
        if mask0 is None or self.dual_var is None:
            beta = np.zeros(n_features)
        else:
            if self.dual_var.shape[0] != (2 * n_samples + n_features + 1):
                beta = np.zeros(n_features)
            else:
                dual_var = self.dual_var
                beta = X.T @ (dual_var[0:n_samples] -
                            dual_var[n_samples:(2 * n_samples)])
                beta += dual_var[(2 * n_samples):(2 * n_samples + n_features)]
                beta += dual_var[-1]
        return beta, dual_var

    @staticmethod
    @njit
    def _update_beta_jac_bcd(
            X, y, beta, dbeta, dual_var, ddual_var,
            hyperparam, L, compute_jac=True):
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
            if j < (2 * n_samples):
                if j < n_samples:
                    j1, j2, sign = j, j, 1
                elif j >= n_samples:
                    j1, j2, sign = j - n_samples, j, -1

                _update_beta_jac_bcd_aux(
                    X, y, epsilon, beta, dbeta, dual_var,
                    ddual_var, L, C, j1, j2, sign, compute_jac)
            else:
                if j < (2 * n_samples + n_features):
                    F = beta[j - (2 * n_samples)]
                    dual_var_old = dual_var[j]
                    zj = dual_var[j] - F
                    dual_var[j] = max(0, zj)
                    beta[j - (2 * n_samples)] -= (dual_var_old - dual_var[j])
                    if compute_jac:
                        dF = dbeta[j - (2 * n_samples)]
                        ddual_var_old = ddual_var[j, :].copy()
                        dzj = ddual_var[j, :] - dF
                        if zj > 0:
                            ddual_var[j, :] = dzj
                        else:
                            ddual_var[j, :] = np.repeat(0.0, 2)
                        dbeta[j - (2 * n_samples), 0] -= (ddual_var_old[0] -
                                                          ddual_var[j, 0])
                        dbeta[j - (2 * n_samples), 1] -= (ddual_var_old[1] -
                                                          ddual_var[j, 1])
                else:
                    F = np.sum(beta) - 1
                    dual_var_old = dual_var[-1]
                    zj = dual_var[j] - F / n_features
                    dual_var[j] = zj
                    beta -= (dual_var_old - dual_var[-1])
                    if compute_jac:
                        dF = np.sum(dbeta, axis=0)
                        ddual_var_old = ddual_var[-1, :].copy()
                        dzj = ddual_var[j] - dF / n_features
                        ddual_var[j, :] = dzj
                        dbeta[:, 0] -= (ddual_var_old[0] - ddual_var[-1, 0])
                        dbeta[:, 1] -= (ddual_var_old[1] - ddual_var[-1, 1])

    @staticmethod
    @njit
    def _update_beta_jac_bcd_sparse(
            data, indptr, indices, y, n_samples, n_features, beta,
            dbeta, dual_var, ddual_var, hyperparam, L, compute_jac=True):
        C = hyperparam[0]
        epsilon = hyperparam[1]
        for j in range(2 * n_samples + n_features + 1):
            if j < (2 * n_samples):
                if j < n_samples:
                    j1, j2, sign = j, j, 1
                elif j >= n_samples and j < (2 * n_samples):
                    j1, j2, sign = j - n_samples, j, -1

                _update_beta_jac_bcd_aux_sparse(
                    data, indptr, indices, y, epsilon,
                    beta, dbeta, dual_var, ddual_var,
                    L, C, j1, j2, sign, compute_jac)

            else:
                if j >= (2 * n_samples) and j < (2 * n_samples + n_features):
                    F = beta[j - (2 * n_samples)]
                    dual_var_old = dual_var[j]
                    zj = dual_var[j] - F
                    dual_var[j] = max(0, zj)
                    beta[j - (2 * n_samples)] -= (dual_var_old - dual_var[j])
                    if compute_jac:
                        dF = dbeta[j - (2 * n_samples)]
                        ddual_var_old = ddual_var[j, :].copy()
                        dzj = ddual_var[j, :] - dF
                        if zj > 0:
                            ddual_var[j, :] = dzj
                        else:
                            ddual_var[j, :] = 0
                        dbeta[j - (2 * n_samples), 0] -= (ddual_var_old[0] -
                                                          ddual_var[j, 0])
                        dbeta[j - (2 * n_samples), 1] -= (ddual_var_old[1] -
                                                          ddual_var[j, 1])
                else:
                    F = np.sum(beta) - 1
                    dual_var_old = dual_var[-1]
                    zj = dual_var[j] - F / n_features
                    dual_var[j] = zj
                    beta -= (dual_var_old - dual_var[-1])
                    if compute_jac:
                        dF = np.sum(dbeta, axis=0)
                        ddual_var_old = ddual_var[-1, :].copy()
                        dzj = ddual_var[j, :] - dF / n_features
                        ddual_var[j, :] = dzj
                        dbeta[:, 0] -= (ddual_var_old[0] - ddual_var[-1, 0])
                        dbeta[:, 1] -= (ddual_var_old[1] - ddual_var[-1, 1])

    def _get_pobj0(self, dual_var, beta, hyperparam, y):
        obj_prim = hyperparam[0] / self.n_samples * np.sum(np.maximum(
            np.abs(y) - hyperparam[1], 0))
        return obj_prim

    def _get_pobj(self, dual_var, X, beta, hyperparam, y):
        n_samples = X.shape[0]
        obj_prim = 0.5 * norm(beta) ** 2 + hyperparam[0] / n_samples * np.sum(np.maximum(
            np.abs(X @ beta - y) - hyperparam[1], 0))
        return obj_prim
    
    @staticmethod
    def _get_dobj(dual_var, X, beta, hyperparam, y):
        n_samples = X.shape[0]
        obj_dual = 0.5 * beta.T @ beta
        obj_dual += hyperparam[1] * np.sum(dual_var[0:(2 * n_samples)])
        obj_dual -= np.sum(y * (dual_var[0:n_samples] -
                                dual_var[n_samples:(2 * n_samples)]))
        obj_dual -= dual_var[-1]
        return -obj_dual

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

    def _init_ddual_var(self, dbeta, X, y, sign_beta, hyperparam):
        dual_var = self.dual_var
        C = hyperparam[0]
        n_samples, n_features = X.shape
        sign = np.zeros(dual_var.shape[0])
        bool_temp = np.isclose(dual_var[0:(2 * n_samples + n_features)], 0.0)
        sign[0:(2 * n_samples + n_features)][bool_temp] = -1.0
        sign[0:(2 * n_samples)][np.isclose(dual_var[0:(2 * n_samples)], C / n_samples)] = 1.0
        ddual_var = np.zeros((dual_var.shape[0], 2))
        if np.any(sign == 1.0):
            ddual_var[sign == 1.0, 0] = np.repeat(C / n_samples, (sign == 1).sum())
            ddual_var[sign == 1.0, 1] = np.repeat(0, (sign == 1).sum())
        self.ddual_var = ddual_var
        self.dbeta = X.T @ (
            ddual_var[0:n_samples, :] -
            ddual_var[n_samples:(2 * n_samples), :])
        self.dbeta += (
            ddual_var[(2 * n_samples):(2 * n_samples + n_features), :])
        self.dbeta += ddual_var[-1, :]
        return ddual_var

    @staticmethod
    @njit
    def _update_only_jac(X, y, dual_var, dbeta, ddual_var,
                         L, hyperparam, sign_beta):
        n_samples, n_features = X.shape
        length_dual = dual_var.shape[0]
        C = hyperparam[0]
        epsilon = hyperparam[1]
        gen_supp = np.zeros(length_dual)
        bool_temp = dual_var[0:(2 * n_samples + n_features)] == 0.0
        gen_supp[0:(2 * n_samples + n_features)][bool_temp] = -1.0
        gen_supp[0:(2 * n_samples)][dual_var[0:(2 * n_samples)] == C / n_samples] = 1.0
        for j in np.arange(0, length_dual)[gen_supp == 0.0]:
            if j < (2 * n_samples):
                if j < n_samples:
                    j1, j2, sign = j, j, 1
                elif j >= n_samples:
                    j1, j2, sign = j - n_samples, j, -1

                _compute_jac_aux(
                    X, epsilon, dbeta, ddual_var, dual_var[j2], L,
                    C, j1, j2, sign)
            else:
                if j < (2 * n_samples + n_features):
                    dF = dbeta[j - (2 * n_samples)]
                    ddual_var_old = ddual_var[j, :].copy()
                    dzj = ddual_var[j, :] - dF
                    ddual_var[j, :] = dzj
                    dbeta[j - (2 * n_samples), 0] -= (ddual_var_old[0] -
                                                      ddual_var[j, 0])
                    dbeta[j - (2 * n_samples), 1] -= (ddual_var_old[1] -
                                                      ddual_var[j, 1])
                else:
                    dF = np.sum(dbeta, axis=0)
                    ddual_var_old = ddual_var[-1, :].copy()
                    dzj = ddual_var[j, :] - dF / n_features
                    ddual_var[j, :] = dzj
                    dbeta[:, 0] -= (ddual_var_old[0] - ddual_var[-1, 0])
                    dbeta[:, 1] -= (ddual_var_old[1] - ddual_var[-1, 1])

    @staticmethod
    @njit
    def _update_only_jac_sparse(
            data, indptr, indices, y, n_samples, n_features,
            dbeta, dual_var, ddual_var, L, hyperparam, sign_beta):
        C = hyperparam[0]
        epsilon = hyperparam[1]

        gen_supp = np.zeros(dual_var.shape[0])
        bool_temp = dual_var[0:(2 * n_samples + n_features)] == 0.0
        gen_supp[0:(2 * n_samples + n_features)][bool_temp] = -1.0
        gen_supp[0:(2 * n_samples)][dual_var[0:(2 * n_samples)] == C / n_samples] = 1.0

        iter = np.arange(0, (2 * n_samples + n_features + 1))[gen_supp == 0.0]
        for j in iter:
            if j < (2 * n_samples):
                if j < n_samples:
                    j1, j2, sign = j, j, 1
                elif j >= n_samples:
                    j1, j2, sign = j - n_samples, j, -1

                _compute_jac_aux_sparse(
                    data, indptr, indices, epsilon, dbeta, ddual_var,
                    dual_var[j2], L, C, j1, j2, sign)

            else:
                if j >= (2 * n_samples) and j < (2 * n_samples + n_features):
                    dF = dbeta[j - (2 * n_samples)]
                    ddual_var_old = ddual_var[j, :].copy()
                    dzj = ddual_var[j, :] - dF
                    ddual_var[j, :] = dzj
                    dbeta[j - (2 * n_samples), 0] -= (ddual_var_old[0] -
                                                      ddual_var[j, 0])
                    dbeta[j - (2 * n_samples), 1] -= (ddual_var_old[1] -
                                                      ddual_var[j, 1])
                else:
                    dF = np.sum(dbeta, axis=0)
                    ddual_var_old = ddual_var[-1, :].copy()
                    dzj = ddual_var[j, :] - dF / n_features
                    ddual_var[j, :] = dzj
                    dbeta[:, 0] -= (ddual_var_old[0] - ddual_var[-1, 0])
                    dbeta[:, 1] -= (ddual_var_old[1] - ddual_var[-1, 1])

    @staticmethod
    @njit
    def _reduce_alpha(alpha, mask):
        return alpha

    @staticmethod
    def get_L(X):
        if issparse(X):
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
        sign[np.isclose(x, np.exp(log_hyperparams[0]) / self.n_samples)] = 1.0
        return sign

    def get_jac_v(self, X, y, mask, dense, jac, v):
        return jac.T @ v(mask, dense)

    @staticmethod
    def get_full_jac_v(mask, jac_v, n_features):
        return jac_v

    def get_dual_v(self, mask, dense, X, y, v, log_hyperparam):
        full_v = np.zeros(X.shape[1])
        full_v[mask] = v
        if v.shape[0] != 0:
            return np.hstack((-X @ full_v, full_v, np.sum(full_v)))
        else:
            return np.zeros(X.shape[0] + X.shape[1] + 1)

    def _get_grad(self, X, y, jac, mask, dense, hyperparam, v):
        C = hyperparam[0]
        n_samples = X.shape[0]
        epsilon = hyperparam[1]
        alpha = self.dual_var[0:n_samples] - \
            self.dual_var[n_samples:(2 * n_samples)]
        n_features = X.shape[1]
        gamma = self.dual_var[(2 * n_samples):(2 * n_samples + n_features)]
        mask0 = np.logical_not(np.isclose(gamma, 0))
        full_supp = np.logical_not(
            np.logical_or(
                np.isclose(alpha, 0),
                np.isclose(np.abs(alpha), C / n_samples)))
        maskC = np.isclose(np.abs(alpha), C / n_samples)
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
        alpha = self.dual_var[0:n_samples] -\
            self.dual_var[n_samples:(2 * n_samples)]
        full_supp = np.logical_not(
            np.logical_or(
                np.isclose(alpha, 0),
                np.isclose(np.abs(alpha), C / n_samples)))
        mask0 = np.logical_not(np.isclose(self.dual_var[(2 * n_samples):
                              (2 * n_samples + n_features)], 0))
        return v[np.hstack((full_supp, mask0, True))]

    def proj_hyperparam(self, X, y, log_hyperparam):
        return np.clip(log_hyperparam, -16, [10, 10.0])

    def get_jac_obj(self, Xs, ys, n_samples, sign_beta,
                    dbeta, dual_var, ddual_var, hyperparam):
        n_features = dbeta.shape[0]
        C = hyperparam[0]
        alpha = dual_var[0:n_samples] - dual_var[n_samples:(2 * n_samples)]
        dalpha = ddual_var[0:n_samples, 0] - \
            ddual_var[n_samples:(2 * n_samples), 0]
        dgamma = ddual_var[(2 * n_samples):(2 * n_samples + n_features), 0]
        dmu = ddual_var[-1, 0]
        maskC = np.isclose(np.abs(alpha), C / n_samples)

        full_supp = np.logical_not(
            np.logical_or(
                np.isclose(alpha, 0),
                np.isclose(np.abs(alpha), C / n_samples)))

        vecX = dalpha[full_supp].T @ Xs[full_supp, :]
        vecX += dgamma + np.repeat(dmu, n_features)
        quadratic_term = vecX.T @ vecX
        linear_term = vecX.T @ Xs[maskC, :].T @ alpha[maskC]
        return norm(quadratic_term + linear_term)

    def get_mat_vec(self, X, y, mask, dense, log_C):
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
        log_C: ndarray
            Logarithm of hyperparameter.
        """
        C = np.exp(log_C)[0]
        n_samples, n_features = X.shape
        alpha = self.dual_var[0:n_samples] - \
            self.dual_var[n_samples:(2 * n_samples)]
        gamma = self.dual_var[(2 * n_samples):(2 * n_samples + n_features)]
        mask0 = np.logical_not(np.isclose(gamma, 0))
        full_supp = np.logical_not(
            np.logical_or(
                np.isclose(alpha, 0),
                np.isclose(np.abs(alpha), C / n_samples)))
        sub_id = np.zeros((mask0.sum(), n_features))
        sub_id[:, mask0] = 1.0
        X_m = np.concatenate((X[full_supp, :],
                                  -sub_id, -np.ones((1, n_features))), axis=0)
        size_supp = X_m.shape[0]
        def mv(v):
            return X_m @ (X_m.T @ v)
        return LinearOperator((size_supp, size_supp), matvec=mv)

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

