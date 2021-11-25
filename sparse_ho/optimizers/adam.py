import numpy as np
from numpy.linalg import norm

from sparse_ho.optimizers.base import BaseOptimizer


class Adam(BaseOptimizer):
    """ADAM optimizer for the outer problem.

    This Adam code is taken from
    https://github.com/sagarvegad/Adam-optimizer/blob/master/Adam.py

    Parameters
    ----------
    n_outer: int, optional (default=100).
        Number of maximum updates of alpha.
    epsilon: float, optional (default=1e-3)
    lr: float, optional (default=1e-2)
        Learning rate
    beta_1: float, optional (default=0.9)
    beta_2: float, optional (default=0.999)
    verbose: bool, optional (default=False)
        Indicates whether information about hyperparameter
        optimization process is printed or not.
    tol : float, optional (default=1e-5)
        Tolerance for the inner optimization solver.
    t_max: float, optional (default=10_000)
        Maximum running time threshold in seconds.
    """

    def __init__(
            self, n_outer=100, epsilon=1e-3, lr=0.01, beta_1=0.9, beta_2=0.999,
            verbose=False, tol=1e-5, t_max=10000):
        self.n_outer = n_outer
        self.epsilon = epsilon
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.verbose = verbose
        self.tol = tol
        self.t_max = t_max

    def _grad_search(
            self, _get_val_grad, proj_hyperparam, log_alpha0, monitor):

        log_alpha = log_alpha0
        # log_alpha0 = 0  # initialize the vector
        m_t = 0
        v_t = 0
        t = 0

        for i in range(self.n_outer):
            t += 1
            value_outer, grad = _get_val_grad(log_alpha, self.tol, monitor)

            if self.verbose:
                print(
                    "Iteration %i/%i || " % (i+1, self.n_outer) +
                    "Value outer criterion: %.2e || " % value_outer +
                    "norm grad %.2e" % norm(grad))

            if (i > 1) and (monitor.objs[-1] > monitor.objs[-2]):
                break
            # updates the moving averages of the gradient
            m_t = self.beta_1*m_t + (1 - self.beta_1) * grad
            # updates the moving averages of the squared gradient
            v_t = self.beta_2*v_t + (1 - self.beta_2) * (grad * grad)
            m_cap = m_t/(1-(self.beta_1**t))
            # calculates the bias-corrected estimates
            v_cap = v_t/(1-(self.beta_2**t))
            # calculates the bias-corrected estimates
            logh_alpha_prev = log_alpha
            # updates the parameters
            log_alpha = log_alpha - (self.lr*m_cap) / (
                np.sqrt(v_cap) + self.epsilon)
            # checks if it is converged or not
            if np.allclose(log_alpha, logh_alpha_prev):
                break
