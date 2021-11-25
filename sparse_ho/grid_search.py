# This files contains the functions to perform zero order descent for HO
# hyperparameter setting
import numpy as np


def grid_search(
        criterion, model, X, y, alpha_min, alpha_max, monitor,
        max_evals=50, tol=1e-5, nb_hyperparam=1,
        random_state=42, samp="grid", alphas=None,
        t_max=100_000, reverse=True):
    if alphas is None and samp == "grid":
        if reverse:
            alphas = np.geomspace(alpha_max, alpha_min, max_evals)
        else:
            alphas = np.linspace(alpha_min, alpha_max, max_evals)
        if nb_hyperparam == 2:
            alphas = np.array(np.meshgrid(
                alphas, alphas)).T.reshape(-1, 2)

    elif samp == "random":
        rng = np.random.RandomState(random_state)
        # sample uniformly on log scale
        alphas = np.exp(rng.uniform(
            np.log(alpha_min), np.log(alpha_max), size=max_evals))
        if reverse:
            alphas = np.sort(alphas)[::-1]
        else:
            alphas = np.sort(alphas)
        if nb_hyperparam == 2:
            alphas2 = np.exp(rng.uniform(
                np.log(alpha_min), np.log(alpha_max), size=max_evals))
            if reverse:
                alphas2 = np.sort(alphas2)[::-1]
            else:
                alphas2 = np.sort(alphas2)
            alphas = np.array(np.meshgrid(
                alphas, alphas2)).T.reshape(-1, 2)

    min_g_func = np.inf
    alpha_opt = alphas[0]

    for i, alpha in enumerate(alphas):
        print("Iteration %i / %i" % (i+1, len(alphas)))

        g_func = criterion.get_val(
            model, X, y, np.log(alpha), monitor, tol=tol)

        if g_func < min_g_func:
            min_g_func = g_func
            alpha_opt = alpha

        if monitor.times[-1] > t_max:
            break
    return alpha_opt, min_g_func
