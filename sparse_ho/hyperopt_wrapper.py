from hyperopt import hp
from hyperopt import fmin, tpe, rand
from sklearn.utils import check_random_state


def hyperopt_wrapper(
        algo, log_alpha_min, log_alpha_max, monitor, max_evals=50,
        tol=1e-5, beta_star=None, random_state=42, t_max=1000,
        method='bayesian'):

    def objective(log_alpha):
        val_func, _ = algo.get_val_grad(
            log_alpha, tol=tol, beta_star=beta_star, compute_jac=False)
        monitor(
            val_func, algo.criterion.val_test, log_alpha, None,
            algo.criterion.rmse)
        return val_func

    space = hp.uniform(
        'log_alpha', log_alpha_min, log_alpha_max)

    rng = check_random_state(random_state)

    if method == "bayesian":
        fmin(
            objective, space, algo=tpe.suggest, max_evals=max_evals,
            timeout=t_max, rstate=rng)
    elif method == "random":
        fmin(
            objective, space, algo=rand.suggest, max_evals=max_evals,
            timeout=t_max, rstate=rng)
    return monitor
