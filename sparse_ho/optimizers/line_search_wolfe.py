from numpy.linalg import norm


class LineSearchWolfe():

    def __init__(
            self, n_outer=100, maxit_ln=5, verbose=False,
            tol=1e-5, t_max=10000):
        self.n_outer = n_outer
        self.maxit_ln = maxit_ln
        self.verbose = verbose
        self.tol = tol
        self.t_max = t_max

    def grad_search_wolfe(
            self, algo, criterion, model, log_alpha0, monitor):

        def _get_val_grad(log_alpha, tol=self.tol):
            return criterion.get_val_grad(
                model, log_alpha, algo.get_beta_jac_v, tol=tol)

        def _get_val(log_alpha, tol=self.tol):
            return criterion.get_val(model, log_alpha, tol=tol)

        log_alphak = log_alpha0
        for _ in range(self.n_outer):
            val, grad = _get_val_grad(log_alphak)

            monitor(val.copy(), criterion.val_test, log_alphak,
                    grad, criterion.rmse)

            step_size = self.wolfe(
                log_alphak, -grad, val, _get_val, _get_val_grad,
                maxit_ln=self.maxit_ln)
            log_alphak -= step_size * grad

    def wolfe(x_k, p_k, val, fun, fun_grad, maxit_ln=5):

        alpha_low = 0
        alpha_high = 1000
        alpha = 1 / (10 * norm(p_k))
        # alpha = 1 / (10 * norm(p_k))
        c1 = 0.1
        c2 = 0.7

        k = 0
        while k < maxit_ln:
            if (fun(x_k + alpha * p_k) > val - c1 * (alpha * norm(p_k) ** 2)):
                alpha_high = alpha
                alpha = (alpha_high+alpha_low) / 2
            elif fun_grad(
                    x_k + alpha * p_k)[1].T * p_k < - c2 * norm(p_k) ** 2:
                alpha_low = alpha
                if alpha_high > 100:
                    alpha = 2 * alpha_low
                else:
                    alpha = (alpha_high + alpha_low) / 2
            else:
                break
            k += 1

        return alpha
