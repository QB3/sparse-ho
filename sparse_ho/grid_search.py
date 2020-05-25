# This files contains the functions to perform zero order descent for HO
# hyperparameter setting
import numpy as np
from smt.sampling_methods import LHS


def grid_search(
        algo, log_alpha_min, log_alpha_max, monitor, max_evals=50,
        tol=1e-5,
        beta_star=None, random_state=42, samp="grid", log_alphas=None):

    if log_alphas is None and samp == "grid":
        log_alphas = np.linspace(log_alpha_min, log_alpha_max, max_evals)

    elif samp == "random":
        rng = np.random.RandomState(random_state)
        log_alphas = rng.uniform(
            log_alpha_min, log_alpha_max, size=max_evals)
        log_alphas = -np.sort(-log_alphas)
    
    elif samp == "lhs":
        xlimits = np.array([[log_alpha_min,  log_alpha_max]])
        sampling = LHS(xlimits=xlimits)
        num = max_evals
        log_alphas = sampling(num)
    
    min_g_func = np.inf
    log_alpha_opt = log_alphas[0]

    for log_alpha in log_alphas:
        if samp == "lhs":
            log_alpha = log_alpha[0]
        g_func, grad_lambda = algo.get_val_grad(
            log_alpha, tol=tol, beta_star=beta_star, compute_jac=False)

        if g_func < min_g_func:
            min_g_func = g_func
            log_alpha_opt = log_alpha
        
        monitor(g_func, algo.criterion.val_test, log_alpha, None,
                algo.criterion.rmse)
    
    return log_alpha_opt, min_g_func
