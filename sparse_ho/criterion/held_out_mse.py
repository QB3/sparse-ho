from numpy.linalg import norm
from ..algo.forward import get_beta_jac_iterdiff


class HeldOutMSE():
    """Held out loss for quadratic datafit.

    Attributes
    ----------
    TODO
    """
    # XXX : this code should be the same as CrossVal as you can pass
    # cv as [(train, test)] ie directly the indices of the train
    # and test splits.

    def __init__(self, idx_train, idx_val):
        """
        Parameters
        ----------
        idx_train: np.array
            indices of the training set
        idx_test: np.array
            indices of the testing set
        """
        self.idx_train = idx_train
        self.idx_val = idx_val

        self.mask0 = None
        self.dense0 = None
        self.quantity_to_warm_start = None
        self.rmse = None

    def get_val_outer(self, X, y, mask, dense):
        """Compute the MSE on the validation set."""
        return norm(y - X[:, mask] @ dense) ** 2 / len(y)

    def get_val(self, model, X, y, log_alpha, tol=1e-3):
        # TODO add warm start
        mask, dense, _ = get_beta_jac_iterdiff(
            X[self.idx_train], y[self.idx_train], log_alpha, model, tol=tol,
            compute_jac=False)
        return self.get_val_outer(mask, dense)

    def get_val_grad(
            self, model, X, y, log_alpha, get_beta_jac_v, max_iter=10000,
            tol=1e-5, compute_jac=True, monitor=None):

        X_train, X_val = X[self.idx_train, :], X[self.idx_val, :]
        y_train, y_val = y[self.idx_train], y[self.idx_val]

        def get_v(mask, dense):
            X_val_m = X_val[:, mask]
            return 2 * (X_val_m.T @ (X_val_m @ dense - y_val)) / len(y_val)

        mask, dense, grad, quantity_to_warm_start = get_beta_jac_v(
            X_train, y_train, log_alpha, model,
            get_v, mask0=self.mask0, dense0=self.dense0,
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

    def proj_hyperparam(self, model, X, y, log_alpha):
        return model.proj_hyperparam(
            X[self.idx_train, :], y[self.idx_train], log_alpha)
