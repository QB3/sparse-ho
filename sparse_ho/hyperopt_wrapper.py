# Authors: Quentin Bertrand <quentin.bertrand@inria.fr>
#          Quentin Klopfenstein <quentin.klopfenstein@u-bourgogne.fr>
#
# License: BSD (3-clause)


import numpy as np
from hyperopt import hp
from hyperopt import fmin, tpe, rand
from sklearn.utils import check_random_state


def hyperopt_wrapper(
        algo, criterion, model, X, y, log_alpha_min, log_alpha_max, monitor,
        max_evals=50, tol=1e-5, random_state=42, t_max=1000,
        method='bayesian', size_space=1):

    def objective(log_alpha):
        log_alpha = np.array(log_alpha)
        val_func = criterion.get_val(
            model, X, y, log_alpha, algo.get_beta_jac_v, monitor, tol=tol)
        return val_func

    space = [
        hp.uniform(str(dim), log_alpha_min, log_alpha_max) for dim in range(
            size_space)]

    # space = hp.uniform(
    #     'log_alpha', log_alpha_min, log_alpha_max)

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
