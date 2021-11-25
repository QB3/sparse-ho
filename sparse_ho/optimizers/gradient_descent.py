
import numpy as np
from numpy.linalg import norm

from sparse_ho.optimizers.base import BaseOptimizer


class GradientDescent(BaseOptimizer):
    """Gradient descent for the outer problem.
    This gradient descent scheme uses a (heuristic) adaptive stepsize:

    log_alphak = log_alphak - p_grad_norm * grad_outer / norm(grad_outer)

    Parameters
    ----------
    n_outer: int, optional (default=100).
        number of maximum updates of alpha.
    step_size: float
        stepsize of the gradient descent
    p_grad_norm: float
        Coefficient multiplying grad_outer / norm(grad_outer) in the gradient
        descent.
    verbose: bool, optional (default=False)
        Indicates whether information about hyperparameter
        optimization process is printed or not.
    tol : float, optional (default=1e-5)
        Tolerance for the inner optimization solver.
    tol_decrease: bool
        To use or not a tolerance decrease strategy in the gradient descent.
    t_max: float, optional (default=10000)
        Maximum running time threshold in seconds.
    """

    def __init__(
            self, n_outer=100, step_size=None, p_grad_norm=1,
            verbose=False, tol=1e-5, tol_decrease=None, t_max=10_000):
        self.n_outer = n_outer
        self.step_size = step_size
        self.verbose = verbose
        self.tol = tol
        self.t_max = t_max
        self.p_grad_norm = p_grad_norm
        self.has_gone_up = False
        self.tol_decrease = tol_decrease

    def _grad_search(
            self, _get_val_grad, proj_hyperparam, log_alpha0, monitor):
        is_multiparam = isinstance(log_alpha0, np.ndarray)
        if is_multiparam:
            log_alphak = log_alpha0.copy()
        else:
            log_alphak = log_alpha0

        if self.tol_decrease is not None:
            tols = np.geomspace(1e-2, self.tol, num=self.n_outer)
        else:
            tols = np.ones(self.n_outer) * self.tol

        for i, tol in enumerate(tols):
            value_outer, grad_outer = _get_val_grad(
                log_alphak, tol, monitor)
            if (self.step_size is None or i < 20) and not self.has_gone_up:
                self.step_size = self.p_grad_norm / (
                    np.linalg.norm(grad_outer) + 1e-12)
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
                self.has_gone_up = True
        return log_alphak, value_outer, grad_outer
