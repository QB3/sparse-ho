import pytest
import numpy as np

from sparse_ho.criterion import (
    HeldOutMSE, HeldOutLogistic, FiniteDiffMonteCarloSure)
from sparse_ho.utils import Monitor
from sparse_ho import Forward

from sparse_ho.tests.common import (
    X, y, sigma_star, idx_train, idx_val,
    models, dict_list_log_alphas)


list_model_crit = [
    ('lasso',  HeldOutMSE(idx_train, idx_val)),
    ('enet', HeldOutMSE(idx_train, idx_val)),
    ('wLasso', HeldOutMSE(idx_train, idx_val)),
    ('lasso', FiniteDiffMonteCarloSure(sigma_star)),
    ('logreg', HeldOutLogistic(idx_train, idx_val))]


tol = 1e-15


@pytest.mark.parametrize('model_name,criterion', list_model_crit)
def test_cross_val_criterion(model_name, criterion):
    # verify dtype from criterion, and the good shape
    algo = Forward()
    monitor_get_val = Monitor()
    monitor_get_val_grad = Monitor()

    model = models[model_name]
    for log_alpha in dict_list_log_alphas[model_name]:
        criterion.get_val(
            model, X, y, log_alpha, tol=tol, monitor=monitor_get_val)
    for log_alpha in dict_list_log_alphas[model_name]:
        criterion.get_val_grad(
            model, X, y, log_alpha, algo.compute_beta_grad,
            tol=tol, monitor=monitor_get_val_grad)

    obj_val = np.array(monitor_get_val.objs)
    obj_val_grad = np.array(monitor_get_val_grad.objs)

    np.testing.assert_allclose(obj_val, obj_val_grad)


if __name__ == '__main__':
    for model_name, criterion in list_model_crit:
        test_cross_val_criterion(model_name, criterion)
