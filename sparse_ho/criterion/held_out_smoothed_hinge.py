import numpy as np
from sparse_ho.algo.forward import get_beta_jac_iterdiff
from scipy.sparse import issparse
from sparse_ho.utils import smooth_hinge
from sparse_ho.utils import derivative_smooth_hinge


class HeldOutSmoothedHinge():
    """Smooth Hinge loss.

    Attributes
    ----------
    TODO
    """

    def __init__(self, idx_train, idx_val):
        """
        Parameters:
        ----------
        idx_train: np.array
            indices of the training set
        idx_val: np.array
            indices of the validation set
        """
        self.idx_train = idx_train
        self.idx_val = idx_val

        self.mask0 = None
        self.dense0 = None
        self.quantity_to_warm_start = None
        self.rmse = None

    def get_val_outer(self, X, y, mask, dense):
        if X is None or y is None:
            return None

        if issparse(X):
            Xbeta_y = (X[:, mask].T).multiply(y).T @ dense
        else:
            Xbeta_y = y * (X[:, mask] @ dense)
        return np.sum(smooth_hinge(Xbeta_y)) / len(y)

    def get_val_grad(
            self, model, X, y, log_alpha, get_beta_jac_v, max_iter=10000,
            tol=1e-5, compute_jac=True, monitor=None):

        X_train, X_val = X[self.idx_train, :], X[self.idx_val, :]
        y_train, y_val = y[self.idx_train], y[self.idx_val]

        def get_v(mask, dense):
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

        mask, dense, grad, quantity_to_warm_start = get_beta_jac_v(
            X_train, y_train, log_alpha, model, get_v,
            mask0=self.mask0, dense0=self.dense0,
            quantity_to_warm_start=self.quantity_to_warm_start,
            max_iter=max_iter, tol=tol, compute_jac=compute_jac,
            full_jac_v=True)

        self.mask0 = mask
        self.dense0 = dense
        self.quantity_to_warm_start = quantity_to_warm_start
        mask, dense = model.get_beta(
            X_train, y_train, mask, dense)
        val = self.get_val_outer(X_val, y_val, mask, dense)

        if monitor is not None:
            monitor(val, grad, mask, dense, log_alpha)

        return val, grad

    def get_val(self, model, X, y, log_alpha, tol=1e-3):
        mask, dense, _ = get_beta_jac_iterdiff(
            X, y, log_alpha, model,  # TODO max_iter
            max_iter=model.max_iter, tol=tol, compute_jac=False)
        mask, dense = model.get_beta(mask, dense)
        val = self.get_val_outer(
            X[self.idx_val], y[self.idx_val], mask, dense)
        return val

    def proj_hyperparam(self, model, X, y, log_alpha):
        return model.proj_hyperparam(
            X[self.idx_train, :], y[self.idx_train], log_alpha)
