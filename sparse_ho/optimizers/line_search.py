import numpy as np
from numpy.linalg import norm

from sparse_ho.optimizers.base import BaseOptimizer


class LineSearch(BaseOptimizer):
    """Gradient descent with line search for the outer problem.

    The code is taken from here:
    https://github.com/fabianp/hoag/blob/master/hoag/hoag.py

    Parameters
    ----------
    n_outer: int, optional (default=100).
        number of maximum updates of alpha.
    verbose: bool, optional (default=False)
        Verbosity.
    tolerance_decrease: string, optional (default="constant")
        Tolerance decrease strategy for approximate gradient.
    tol : float, optional (default=1e-5)
        Tolerance for the inner optimization solver.
    t_max: float, optional (default=10000)
        Maximum running time threshold in seconds.
    """

    def __init__(
            self, n_outer=100, verbose=False, tolerance_decrease='constant',
            tol=1e-5, t_max=10000):
        self.n_outer = n_outer
        self.verbose = verbose
        self.tolerance_decrease = tolerance_decrease
        self.tol = tol
        self.t_max = t_max

    def _grad_search(
            self, _get_val_grad, proj_hyperparam, log_alpha0, monitor):

        is_multiparam = isinstance(log_alpha0, np.ndarray)
        if is_multiparam:
            log_alphak = log_alpha0.copy()
            old_log_alphak = log_alphak.copy()
        else:
            log_alphak = log_alpha0
            old_log_alphak = log_alphak

        grad_norms = []

        L_log_alpha = None
        value_outer_old = np.inf

        if self.tolerance_decrease == 'exponential':
            seq_tol = np.geomspace(1e-2, self.tol, self.n_outer)
        else:
            seq_tol = self.tol * np.ones(self.n_outer)

        for i in range(self.n_outer):
            tol = seq_tol[i]
            try:
                old_tol = seq_tol[i - 1]
            except Exception:
                old_tol = seq_tol[0]
            value_outer, grad_outer = _get_val_grad(
                log_alphak, tol=tol, monitor=monitor)

            grad_norms.append(norm(grad_outer))
            if np.isnan(grad_norms[-1]):
                print("Nan present in gradient")
                break

            if L_log_alpha is None:
                if grad_norms[-1] > 1e-3:
                    # make sure we are not selecting a step size
                    # that is too small
                    if is_multiparam:
                        L_log_alpha = grad_norms[-1] / np.sqrt(len(log_alphak))
                    else:
                        L_log_alpha = grad_norms[-1]
                else:
                    L_log_alpha = 1
            step_size = (1. / L_log_alpha)
            try:
                old_log_alphak = log_alphak.copy()
            except Exception:
                old_log_alphak = log_alphak
            log_alphak -= step_size * grad_outer

            incr = norm(step_size * grad_outer)
            C = 0.25
            factor_L_log_alpha = 1.0
            if value_outer <= value_outer_old + C * tol + \
                    old_tol * (C + factor_L_log_alpha) * incr - \
                    factor_L_log_alpha * (L_log_alpha) * incr * incr:
                L_log_alpha *= 0.95
                if self.verbose > 1:
                    print('increased step size')
                log_alphak -= step_size * grad_outer

            elif value_outer >= 1.2 * value_outer_old:
                if self.verbose > 1:
                    print('decrease step size')
                # decrease step size
                L_log_alpha *= 2
                if is_multiparam:
                    log_alphak = old_log_alphak.copy()
                else:
                    log_alphak = old_log_alphak
                print('!!step size rejected!!', value_outer, value_outer_old)
                value_outer, grad_outer = _get_val_grad(
                    log_alphak, tol=tol, monitor=monitor)

                tol *= 0.5
            else:
                old_log_alphak = log_alphak.copy()
                log_alphak -= step_size * grad_outer

            log_alphak = proj_hyperparam(log_alphak)
            value_outer_old = value_outer

            if self.verbose:
                print(
                    "Iteration %i/%i || " % (i+1, self.n_outer) +
                    "Value outer criterion: %.2e || " % value_outer +
                    "norm grad %.2e" % norm(grad_outer))
            if monitor.times[-1] > self.t_max:
                break
        return log_alphak, value_outer, grad_outer
