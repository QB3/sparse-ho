# This files contains the functions to perform first order descent for HO
# hyperparameter setting


def grad_search(
        algo, criterion, model, optimizer, X, y, log_alpha0, monitor):
    """
    Parameters
    ----------
    algo: instance of BaseAlgo
        algorithm used to compute hypergradient.
    criterion:  instance of BaseCriterion
        criterion to optimize during hyperparameter optimization
        (outer optimization problem).
    model:  instance of BaseModel
        model on which hyperparameter has to be selected
        (inner optimization problem).
    X: array like of shape (n_samples, n_features)
        Design matrix.
    y: array like of shape (n_samples,)
        Target.
    log_alpha0: float
        initial value of the logarithm of the regularization coefficient alpha.
    monitor: instance of Monitor
        used to store the value of the cross-validation function.
    """

    def _get_val_grad(log_alpha, tol, monitor):
        return criterion.get_val_grad(
            model, X, y, log_alpha, algo.get_beta_jac_v, tol=tol,
            monitor=monitor)

    def _proj_hyperparam(log_alpha):
        return criterion.proj_hyperparam(model, X, y, log_alpha)

    return optimizer._grad_search(
        _get_val_grad, _proj_hyperparam, log_alpha0, monitor)
