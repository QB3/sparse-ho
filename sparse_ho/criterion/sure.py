from numpy.linalg import norm
from sklearn.utils import check_random_state

from sparse_ho.algo.forward import get_beta_jac_iterdiff
from sparse_ho.criterion.base import BaseCriterion


class FiniteDiffMonteCarloSure(BaseCriterion):
    """Smoothed version of the Stein Unbiased Risk Estimator (SURE).

    Implements the iterative Finite-Difference Monte-Carlo approximation of the
    SURE. By default, the approximation is ruled by a power law heuristic [1].

    Parameters
    ----------
    sigma: float
        Noise level
    finite_difference_step: float, optional
        Finite difference step used in the approximation of the SURE.
        By default, use a power law heuristic.
    random_state : int, RandomState instance, default=42
        The seed of the pseudo random number generator.
        Pass an int for reproducible output across multiple function calls.

    Attributes
    ----------
    TODO

    References
    ----------
    .. [1] C.-A. Deledalle, Stein Unbiased GrAdient estimator of the Risk
    (SUGAR) for multiple parameter selection.
    SIAM J. Imaging Sci., 7(4), 2448-2487.
    """

    def __init__(self, sigma, finite_difference_step=None,
                 random_state=42):
        self.sigma = sigma
        self.random_state = random_state
        self.finite_difference_step = finite_difference_step
        self.init_delta_epsilon = False

        self.mask0 = None
        self.dense0 = None
        self.quantity_to_warm_start = None

        self.mask02 = None
        self.dense02 = None
        self.quantity_to_warm_start2 = None

        self.rmse = None

    def get_val_outer(self, X, y, mask, dense, mask2, dense2):
        X_m = X[:, mask]  # avoid multiple calls to X[:, mask]
        dof = ((X[:, mask2] @ dense2 -
                X_m @ dense) @ self.delta)
        dof /= self.epsilon
        # compute the value of the sure
        val = norm(y - X_m @ dense) ** 2
        val -= len(y) * self.sigma ** 2
        val += 2 * self.sigma ** 2 * dof
        return val

    def get_val(self, model, X, y, log_alpha, monitor=None, tol=1e-3):
        # TODO add warm start
        if not self.init_delta_epsilon:
            self._init_delta_epsilon(X)
        mask, dense, _ = get_beta_jac_iterdiff(
            X, y, log_alpha, model,
            tol=tol, mask0=self.mask0, dense0=self.dense0, compute_jac=False)
        mask2, dense2, _ = get_beta_jac_iterdiff(
            X, y + self.epsilon * self.delta,
            log_alpha, model, tol=tol, compute_jac=False)

        val = self.get_val_outer(X, y, mask, dense, mask2, dense2)
        if monitor is not None:
            monitor(val, None, mask, dense, log_alpha)
        return val

    def _init_delta_epsilon(self, X):
        if self.finite_difference_step:
            self.epsilon = self.finite_difference_step
        else:
            # Use Deledalle et al. 2014 heuristic
            self.epsilon = 2.0 * self.sigma / (X.shape[0]) ** 0.3
        rng = check_random_state(self.random_state)
        self.delta = rng.randn(X.shape[0])  # sample random noise for MCMC step
        self.init_delta_epsilon = True

    def get_val_grad(
            self, model, X, y, log_alpha, get_beta_jac_v,
            mask0=None, dense0=None,
            jac0=None, max_iter=1000, tol=1e-3, compute_jac=True,
            monitor=None):
        if not self.init_delta_epsilon:
            self._init_delta_epsilon(X)

        def v(mask, dense):
            X_m = X[:, mask]  # avoid multiple calls to X[:, mask]
            return (2 * X_m.T @ (
                    X_m @ dense - y -
                    self.delta * self.sigma ** 2 / self.epsilon))

        def v2(mask, dense):
            return ((2 * self.sigma ** 2 *
                     X[:, mask].T @ self.delta / self.epsilon))

        mask, dense, jac_v, quantity_to_warm_start = get_beta_jac_v(
            X, y, log_alpha, model, v,
            mask0=self.mask0, dense0=self.dense0,
            quantity_to_warm_start=self.quantity_to_warm_start,
            max_iter=max_iter, tol=tol, compute_jac=compute_jac,
            full_jac_v=True)
        mask2, dense2, jac_v2, quantity_to_warm_start2 = get_beta_jac_v(
            X, y + self.epsilon * self.delta,
            log_alpha, model, v2, mask0=self.mask02,
            dense0=self.dense02,
            quantity_to_warm_start=self.quantity_to_warm_start2,
            max_iter=max_iter, tol=tol, compute_jac=compute_jac,
            full_jac_v=True)
        val = self.get_val_outer(X, y, mask, dense, mask2, dense2)
        self.mask0 = mask
        self.dense0 = dense
        self.quantity_to_warm_start = quantity_to_warm_start

        self.mask02 = mask2
        self.dense02 = dense2
        self.quantity_to_warm_start2 = quantity_to_warm_start2

        if jac_v is not None and jac_v2 is not None:
            grad = jac_v + jac_v2
        else:
            grad = None
        if monitor is not None:
            monitor(val, grad, mask, dense, log_alpha)

        return val, grad
