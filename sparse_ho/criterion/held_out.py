import numpy as np
from numpy.linalg import norm
from scipy.sparse import issparse

from sparse_ho.utils import sigma, smooth_hinge
from sparse_ho.utils import derivative_smooth_hinge
from sparse_ho.algo.forward import compute_beta
from sparse_ho.criterion.base import BaseCriterion


class HeldOutMSE(BaseCriterion):
    """Held out loss for quadratic datafit.

    Parameters
    ----------
    idx_train: ndarray
        indices of the training set
    idx_val: ndarray
        indices of the validation set
    """
    # XXX : this code should be the same as CrossVal as you can pass
    # cv as [(train, test)] ie directly the indices of the train
    # and test splits.

    def __init__(self, idx_train, idx_val):
        self.idx_train = idx_train
        self.idx_val = idx_val

        self.mask0 = None
        self.dense0 = None
        self.quantity_to_warm_start = None

    def get_val_outer(self, X, y, mask, dense):
        """Compute the MSE on the validation set.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Design matrix.
        y: ndarray, shape (n_samples,)
            Observation vector.
        mask: array-like, shape (n_features,)
            Boolean array corresponding to the non-zeros coefficients.
        dense: ndarray
            Values of the non-zeros coefficients.
        """
        return norm(y - X[:, mask] @ dense) ** 2 / len(y)

    def get_val(self, model, X, y, log_alpha, monitor=None, tol=1e-3):
        """Get value of criterion.

        Parameters
        ----------
        model: instance of ``sparse_ho.base.BaseModel``
            A model that follows the sparse_ho API.
        X: array-like, shape (n_samples, n_features)
            Design matrix.
        y: ndarray, shape (n_samples,)
            Observation vector.
        log_alpha: float or np.array
            Logarithm of hyperparameter.
        monitor: instance of Monitor.
            Monitor.
        tol: float, optional (default=1e-3)
            Tolerance for the inner problem.
        """
        mask, dense, _ = compute_beta(
            X[self.idx_train], y[self.idx_train], log_alpha, model,
            mask0=self.mask0, dense0=self.dense0, tol=tol,
            compute_jac=False)
        value_outer = self.get_val_outer(
            X[self.idx_val, :], y[self.idx_val], mask, dense)

        self.mask0 = mask
        self.dense0 = dense

        if monitor is not None:
            monitor(value_outer, None, alpha=np.exp(log_alpha))
        return value_outer

    def get_val_grad(
            self, model, X, y, log_alpha, compute_beta_grad, max_iter=10000,
            tol=1e-5, monitor=None):
        """Get value and gradient of criterion.

        Parameters
        ----------
        model: instance of ``sparse_ho.base.BaseModel``
            A model that follows the sparse_ho API.
        X: array-like, shape (n_samples, n_features)
            Design matrix.
        y: ndarray, shape (n_samples,)
            Observation vector.
        log_alpha: float or np.array
            Logarithm of hyperparameter.
        compute_beta_grad: callable
            Returns the regression coefficients beta and the hypergradient.
        max_iter: int
            Maximum number of iteration for the inner problem.
        tol: float, optional (default=1e-3)
            Tolerance for the inner problem.
        monitor: instance of Monitor.
            Monitor.
        """

        X_train, X_val = X[self.idx_train, :], X[self.idx_val, :]
        y_train, y_val = y[self.idx_train], y[self.idx_val]

        def get_grad_outer(mask, dense):
            X_val_m = X_val[:, mask]
            return 2 * (X_val_m.T @ (X_val_m @ dense - y_val)) / len(y_val)
        mask, dense, grad, quantity_to_warm_start = compute_beta_grad(
            X_train, y_train, log_alpha, model,
            get_grad_outer, mask0=self.mask0, dense0=self.dense0,
            quantity_to_warm_start=self.quantity_to_warm_start,
            max_iter=max_iter, tol=tol, full_jac_v=True)

        self.mask0 = mask
        self.dense0 = dense
        self.quantity_to_warm_start = quantity_to_warm_start
        val = self.get_val_outer(X_val, y_val, mask, dense)
        if monitor is not None:
            monitor(val, grad, mask, dense, alpha=np.exp(log_alpha))
        return val, grad

    def proj_hyperparam(self, model, X, y, log_alpha):
        """Project hyperparameter on a range of admissible values.

        Parameters
        ----------
        model: instance of ``sparse_ho.base.BaseModel``
            A model that follows the sparse_ho API.
        X: array-like, shape (n_samples, n_features)
            Design matrix.
        y: ndarray, shape (n_samples,)
            Observation vector.
        log_alpha: float
            Logarithm of hyperparameter.
        """
        return model.proj_hyperparam(
            X[self.idx_train, :], y[self.idx_train], log_alpha)


class HeldOutLogistic(BaseCriterion):
    """Logistic loss on held out data

    Parameters
    ----------
    idx_train: ndarray
        indices of the training set
    idx_val: ndarray
        indices of the validation set
    """

    def __init__(self, idx_train, idx_val):
        self.idx_train = idx_train
        self.idx_val = idx_val

        self.mask0 = None
        self.dense0 = None
        self.quantity_to_warm_start = None

    @staticmethod
    def get_val_outer(X, y, mask, dense):
        """Compute the logistic loss on the validation set.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Design matrix.
        y: ndarray, shape (n_samples,)
            Observation vector.
        mask: array-like, shape (n_features,)
            Boolean array corresponding to the non-zeros coefficients.
        dense: ndarray
            Values of the non-zeros coefficients.
        """
        val = np.sum(np.log(1 + np.exp(-y * (X[:, mask] @ dense))))
        val /= X.shape[0]
        return val

    def get_val(self, model, X, y, log_alpha, monitor=None, tol=1e-3):
        """Get value of criterion.

        Parameters
        ----------
        model: instance of ``sparse_ho.base.BaseModel``
            A model that follows the sparse_ho API.
        X: array-like, shape (n_samples, n_features)
            Design matrix.
        y: ndarray, shape (n_samples,)
            Observation vector.
        log_alpha: float or np.array
            Logarithm of hyperparameter.
        monitor: instance of Monitor.
            Monitor.
        tol: float, optional (default=1e-3)
            Tolerance for the inner problem.
        """
        mask, dense, _ = compute_beta(
            X[self.idx_train], y[self.idx_train], log_alpha, model,
            mask0=self.mask0, dense0=self.dense0, tol=tol, compute_jac=False)
        val = self.get_val_outer(
            X[self.idx_val, :], y[self.idx_val], mask, dense)

        self.mask0 = mask
        self.dense0 = dense

        if monitor is not None:
            monitor(val, None, mask, dense, alpha=np.exp(log_alpha))
        return val

    def get_val_grad(
            self, model, X, y, log_alpha, compute_beta_grad, max_iter=10000,
            tol=1e-5, monitor=None):
        """Get value and gradient of criterion.

        Parameters
        ----------
        model: instance of ``sparse_ho.base.BaseModel``
            A model that follows the sparse_ho API.
        X: array-like, shape (n_samples, n_features)
            Design matrix.
        y: ndarray, shape (n_samples,)
            Observation vector.
        log_alpha: float or np.array
            Logarithm of hyperparameter.
        compute_beta_grad: callable
            Returns the regression coefficients beta and the hypergradient.
        max_iter: int
            Maximum number of iteration for the inner problem.
        tol: float, optional (default=1e-3)
            Tolerance for the inner problem.
        monitor: instance of Monitor.
            Monitor.
        """

        X_train, X_val = X[self.idx_train, :], X[self.idx_val, :]
        y_train, y_val = y[self.idx_train], y[self.idx_val]

        def get_grad_outer(mask, dense):
            X_val_m = X_val[:, mask]
            temp = sigma(y_val * (X_val_m @ dense))
            v = X_val_m.T @ (y_val * (temp - 1))
            v /= len(y_val)
            return v

        mask, dense, grad, quantity_to_warm_start = compute_beta_grad(
            X_train, y_train, log_alpha, model, get_grad_outer, mask0=self.
            mask0, dense0=self.dense0,
            quantity_to_warm_start=self.quantity_to_warm_start,
            max_iter=max_iter, tol=tol, full_jac_v=True)

        self.mask0 = mask
        self.dense0 = dense
        self.quantity_to_warm_start = quantity_to_warm_start
        val = self.get_val_outer(X_val, y_val, mask, dense)
        if monitor is not None:
            monitor(val, grad, mask, dense, alpha=np.exp(log_alpha))

        return val, grad

    def proj_hyperparam(self, model, X, y, log_alpha):
        """Project hyperparameter on a range of admissible values.

        Parameters
        ----------
        model: instance of ``sparse_ho.base.BaseModel``
            A model that follows the sparse_ho API.
        X: array-like, shape (n_samples, n_features)
            Design matrix.
        y: ndarray, shape (n_samples,)
            Observation vector.
        log_alpha: float
            Logarithm of hyperparameter.
        """
        return model.proj_hyperparam(
            X[self.idx_train, :], y[self.idx_train], log_alpha)


class HeldOutSmoothedHinge(BaseCriterion):
    """Smooth Hinge loss.

    Parameters
    ----------
    idx_train: ndarray
        indices of the training set
    idx_val: ndarray
        indices of the validation set
    """

    def __init__(self, idx_train, idx_val):
        """
        Parameters:
        ----------
        idx_train: ndarray
            indices of the training set
        idx_val: ndarray
            indices of the validation set
        """
        self.idx_train = idx_train
        self.idx_val = idx_val

        self.mask0 = None
        self.dense0 = None
        self.quantity_to_warm_start = None

    def get_val_outer(self, X, y, mask, dense):
        """Compute the smoothed Hinge on the validation set.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Design matrix.
        y: ndarray, shape (n_samples,)
            Observation vector.
        mask: array-like, shape (n_features,)
            Boolean array corresponding to the non-zeros coefficients.
        dense: ndarray
            Values of the non-zeros coefficients.
        """

        if issparse(X):
            Xbeta_y = (X[:, mask].T).multiply(y).T @ dense
        else:
            Xbeta_y = y * (X[:, mask] @ dense)
        return np.sum(smooth_hinge(Xbeta_y)) / len(y)

    def get_val_grad(
            self, model, X, y, log_alpha, compute_beta_grad, max_iter=10000,
            tol=1e-5, monitor=None):
        """Get value and gradient of criterion.

        Parameters
        ----------
        model: instance of ``sparse_ho.base.BaseModel``
            A model that follows the sparse_ho API.
        X: array-like, shape (n_samples, n_features)
            Design matrix.
        y: ndarray, shape (n_samples,)
            Observation vector.
        log_alpha: float or np.array
            Logarithm of hyperparameter.
        compute_beta_grad: callable
            Returns the regression coefficients beta and the hypergradient.
        max_iter: int
            Maximum number of iteration for the inner problem.
        tol: float, optional (default=1e-3)
            Tolerance for the inner problem.
        monitor: instance of Monitor.
            Monitor.
        """

        X_train, X_val = X[self.idx_train, :], X[self.idx_val, :]
        y_train, y_val = y[self.idx_train], y[self.idx_val]

        def get_grad_outer(mask, dense):
            X_val_m = X_val[:, mask]
            Xbeta_y = y_val * (X_val_m @ dense)
            deriv = derivative_smooth_hinge(Xbeta_y)
            if issparse(X):
                v = X_val_m.T.multiply(deriv * y_val)
                v = np.array(np.sum(v, axis=1))
                v = np.squeeze(v)
            else:
                v = (deriv * y_val)[:, np.newaxis] * X_val_m
                v = np.sum(v, axis=0)
            v /= len(self.idx_val)
            return v

        mask, dense, grad, quantity_to_warm_start = compute_beta_grad(
            X_train, y_train, log_alpha, model, get_grad_outer,
            mask0=self.mask0, dense0=self.dense0,
            quantity_to_warm_start=self.quantity_to_warm_start,
            max_iter=max_iter, tol=tol, full_jac_v=True)

        self.mask0 = mask
        self.dense0 = dense
        self.quantity_to_warm_start = quantity_to_warm_start

        val = self.get_val_outer(X_val, y_val, mask, dense)

        if monitor is not None:
            monitor(val, grad, mask, dense, alpha=np.exp(log_alpha))

        return val, grad

    def get_val(self, model, X, y, log_alpha, tol=1e-3):
        """Get value of criterion.

        Parameters
        ----------
        model: instance of ``sparse_ho.base.BaseModel``
            A model that follows the sparse_ho API.
        X: array-like, shape (n_samples, n_features)
            Design matrix.
        y: ndarray, shape (n_samples,)
            Observation vector.
        log_alpha: float or np.array
            Logarithm of hyperparameter.
        tol: float, optional (default=1e-3)
            Tolerance for the inner problem.
        """
        # TODO add maxiter param for all get_val
        mask, dense, _ = compute_beta(
            X, y, log_alpha, model,
            tol=tol, compute_jac=False)
        val = self.get_val_outer(
            X[self.idx_val], y[self.idx_val], mask, dense)
        return val

    def proj_hyperparam(self, model, X, y, log_alpha):
        """Project hyperparameter on a range of admissible values.

        Parameters
        ----------
        model: instance of ``sparse_ho.base.BaseModel``
            A model that follows the sparse_ho API.
        X: array-like, shape (n_samples, n_features)
            Design matrix.
        y: ndarray, shape (n_samples,)
            Observation vector.
        log_alpha: float
            Logarithm of hyperparameter.
        """
        return model.proj_hyperparam(
            X[self.idx_train, :], y[self.idx_train], log_alpha)
