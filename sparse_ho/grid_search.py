# This files contains the functions to perform zero order descent for HO
# hyperparameter setting
import numpy as np
try:
    from smt.sampling_methods import LHS
except Exception:
    print("could import smt.sampling_methods")


def grid_search(
        algo, criterion, model, X, y, log_alpha_min, log_alpha_max, monitor,
        max_evals=50, tol=1e-5, nb_hyperparam=1, beta_star=None,
        random_state=42, samp="grid", log_alphas=None, t_max=100_000,
        reverse=True, verbose=False):
    if log_alphas is None and samp == "grid":
        if reverse:
            log_alphas = np.linspace(log_alpha_max, log_alpha_min, max_evals)
        else:
            log_alphas = np.linspace(log_alpha_min, log_alpha_max, max_evals)
        if nb_hyperparam == 2:
            log_alphas = np.array(np.meshgrid(
                log_alphas, log_alphas)).T.reshape(-1, 2)

    elif samp == "random":
        rng = np.random.RandomState(random_state)
        log_alphas = rng.uniform(
            log_alpha_min, log_alpha_max, size=max_evals)
        if reverse:
            log_alphas = -np.sort(-log_alphas)
        else:
            log_alphas = np.sort(log_alphas)
        if nb_hyperparam == 2:
            log_alphas2 = rng.uniform(
                log_alpha_min, log_alpha_max, size=max_evals)
            if reverse:
                log_alphas2 = -np.sort(-log_alphas2)
            else:
                log_alphas2 = np.sort(log_alphas2)
            log_alphas = np.array(np.meshgrid(
                log_alphas, log_alphas2)).T.reshape(-1, 2)

    elif samp == "lhs":
        xlimits = np.array([[log_alpha_min, log_alpha_max]])
        sampling = LHS(xlimits=xlimits)
        num = max_evals
        log_alphas = sampling(num)
        log_alphas[log_alphas < log_alpha_min] = log_alpha_min
        log_alphas[log_alphas > log_alpha_max] = log_alpha_max
    min_g_func = np.inf
    log_alpha_opt = log_alphas[0]

    # if nb_hyperparam == 2:
    #     n_try = max_evals ** 2
    # else:
    #     n_try = log_alphas.shape[0]

    for i, log_alpha in enumerate(log_alphas):
        if verbose:
            print("Iteration %i / %i" % (i+1, len(log_alphas)))
        # try:
        #     log_alpha = log_alphas[i, :]
        # except Exception:
        #     log_alpha = log_alphas[i]
        if samp == "lhs":
            log_alpha = log_alpha[0]
        g_func = criterion.get_val(
            model, X, y, log_alpha, monitor, tol=tol)

        if g_func < min_g_func:
            min_g_func = g_func
            log_alpha_opt = log_alpha

        if monitor.times[-1] > t_max:
            break
    return log_alpha_opt, min_g_func
