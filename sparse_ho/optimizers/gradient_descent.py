
import numpy as np
from numpy.linalg import norm

from sparse_ho.optimizers.base import BaseOptimizer


class GradientDescent(BaseOptimizer):
    """Gradient descent for the outer problem.

    n_outer: int, optional (default=100).
        number of maximum updates of alpha.
    step_size: float
        stepsize of the gradient descent
    verbose: bool, optional (default=False)
        Indicates whether information about hyperparameter
        optimization process is printed or not.
    tol : float, optional (default=1e-5)
        Tolerance for the inner optimization solver.
    t_max: float, optional (default=10000)
        Maximum running time threshold in seconds.
    """
    def __init__(
            self, n_outer=100, step_size=None, p_grad0=1,
            verbose=False, tol=1e-5,
            t_max=10_000):
        self.n_outer = n_outer
        self.step_size = step_size
        self.verbose = verbose
        self.tol = tol
        self.t_max = t_max
        self.p_grad0 = p_grad0

    def _grad_search(
            self, _get_val_grad, proj_hyperparam, log_alpha0, monitor):
        is_multiparam = isinstance(log_alpha0, np.ndarray)
        if is_multiparam:
            log_alphak = log_alpha0.copy()
        else:
            log_alphak = log_alpha0

        for i in range(self.n_outer):
            value_outer, grad_outer = _get_val_grad(
                log_alphak, self.tol, monitor)
            if self.step_size is None or i < 10:
                self.step_size = self.p_grad0 / (
                    np.linalg.norm(grad_outer) + 1e-2)
            log_alphak -= self.step_size * grad_outer

            if self.verbose:
                print(
                    "Iteration %i/%i ||" % (i+1, self.n_outer) +
                    "Value outer criterion: %.2e ||" % value_outer +
                    "norm grad %.2e" % norm(grad_outer))
            if len(monitor.times) > 0 and monitor.times[-1] > self.t_max:
                break

            if i > 0 and (monitor.objs[-1] > monitor.objs[-2]):
                self.step_size /= 10
        return log_alphak, value_outer, grad_outer
