# Authors: Quentin Bertrand <quentin.bertrand@inria.fr>
#          Quentin Klopfenstein <quentin.klopfenstein@u-bourgogne.fr>
#
# License: BSD (3-clause)


from hyperopt import hp
from hyperopt import fmin, tpe, rand
from sklearn.utils import check_random_state


def hyperopt_wrapper(
        algo, criterion, log_alpha_min, log_alpha_max, monitor, max_evals=50,
        tol=1e-5, beta_star=None, random_state=42, t_max=1000,
        method='bayesian'):

    def objective(log_alpha):
        val_func, _ = criterion.get_val_grad(
            log_alpha, algo.get_beta_jac_v, tol=tol, beta_star=beta_star, compute_jac=False)
        monitor(
            val_func, criterion.val_test, log_alpha, None,
            criterion.rmse)
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
