
import numpy as np


class GradientDescent():
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
            self, n_outer=100, step_size=None, verbose=False, tol=1e-5,
            t_max=10_000):
        self.n_outer = n_outer
        self.step_size = step_size
        self.verbose = verbose
        self.tol = tol
        self.t_max = t_max

    def _grad_search(
            self, _get_val_grad, proj_hyperparam, log_alpha0, monitor):
        is_multiparam = isinstance(log_alpha0, np.ndarray)
        if is_multiparam:
            log_alphak = log_alpha0.copy()
        else:
            log_alphak = log_alpha0

        for _ in range(self.n_outer):
            value_outer, grad_outer = _get_val_grad(
                log_alphak, self.tol, monitor)
            log_alphak -= self.step_size * grad_outer

            if self.verbose:
                print("Value outer criterion: %f" % value_outer)
            if len(monitor.times) > 0 and monitor.times[-1] > self.t_max:
                break
        return log_alphak, value_outer, grad_outer
