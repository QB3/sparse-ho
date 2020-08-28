# This files contains the functions to perform zero order descent for HO
# hyperparameter setting
import numpy as np
try:
    from smt.sampling_methods import LHS
except Exception:
    print("could import smt.sampling_methods")


def grid_search(
        algo, log_alpha_min, log_alpha_max, monitor, max_evals=50,
        tol=1e-5, nb_hyperparam=1,
        beta_star=None, random_state=42, samp="grid", log_alphas=None,
        t_max=1000, reverse=True):

    if log_alphas is None and samp == "grid":
        if reverse:
            log_alphas = np.linspace(log_alpha_max, log_alpha_min, max_evals)
        else:
            log_alphas = np.linspace(log_alpha_min, log_alpha_max, max_evals)
        if nb_hyperparam == 2:
            log_alphas = np.array(np.meshgrid(log_alphas, log_alphas)).T.reshape(-1, 2)

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
            log_alphas = np.array(np.meshgrid(log_alphas, log_alphas2)).T.reshape(-1, 2)

    elif samp == "lhs":
        xlimits = np.array([[log_alpha_min, log_alpha_max]])
        sampling = LHS(xlimits=xlimits)
        num = max_evals
        log_alphas = sampling(num)
        log_alphas[log_alphas < log_alpha_min] = log_alpha_min
        log_alphas[log_alphas > log_alpha_max] = log_alpha_max
    min_g_func = np.inf
    log_alpha_opt = log_alphas[0]

    if nb_hyperparam == 2:
        n_try = max_evals ** 2
    else:
        n_try = log_alphas.shape[0]
    for i in range(n_try):
        try:
            log_alpha = log_alphas[i, :]
        except Exception:
            log_alpha = log_alphas[i]
        if samp == "lhs":
            log_alpha = log_alpha[0]
        # g_func = algo.criterion.get_val(
        #     log_alpha, tol=tol)
        # log_alpha, tol=algo.criterion.model.tol,
        # beta_star=beta_star, compute_jac=False)
        g_func, grad_lambda = algo.get_val_grad(
            log_alpha, tol=tol, beta_star=beta_star, compute_jac=False)

        if g_func < min_g_func:
            min_g_func = g_func
            log_alpha_opt = log_alpha

        monitor(g_func, algo.criterion.val_test, log_alpha, None,
                algo.criterion.rmse)
        if monitor.times[-1] > t_max:
            break
    return log_alpha_opt, min_g_func
