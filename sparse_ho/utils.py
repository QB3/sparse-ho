import time
import numpy as np
from numba import njit


@njit
def sparse_scalar_product(Xjs, idx_j, Xis, idx_i):
    product = 0
    if len(idx_j) != 0 and len(idx_i) != 0:
        cursor_j = 0
        cursor_i = 0
        for k in range(len(idx_j) + len(idx_i)):
            if idx_j[cursor_j] == idx_i[cursor_i]:
                product += Xjs[cursor_j] * Xis[cursor_i]
                cursor_i += 1
                cursor_j += 1

            elif idx_j[cursor_j] < idx_i[cursor_i]:
                cursor_j += 1
            else:
                cursor_i += 1
            if cursor_j >= (len(idx_j)) or cursor_i >= (len(idx_i)):
                break
        return product
    else:
        return 0.0


@njit
def ST(x, alpha):
    return np.sign(x) * np.maximum(np.abs(x) - alpha, 0.)


@njit
def prox_elasticnet(x, alpha_1, alpha_2):
    return (1 / (1 + (alpha_2))) * ST(x, alpha_1)


@njit
def proj_box_svm(x, C):
    return min(max(0, x), C)


@njit
def compute_grad_proj(theta, F, C):
    if theta == 0:
        return min(F, 0)
    elif theta == C:
        return max(F, 0)
    else:
        return F


@njit
def ind_box(x, C):
    return np.logical_and((x > 0), (x < C))


@njit
def sigma(z):
    return 1 / (1 + np.exp(-z))


@njit
def xlogx(x):
    if x < 1e-10:
        return 0.
    else:
        return x * np.log(x)


@njit
def negative_ent(x):
    """
    Negative entropy:
    x * log(x) + (1 - x) * log(1 - x)
    """
    if 0. <= x <= 1.:
        return xlogx(x) + xlogx(1. - x)
    else:
        return np.inf


@njit
def dual_logreg(y, theta, alpha):
    d_obj = 0
    n_samples = len(y)
    for i in range(y.shape[0]):
        d_obj -= negative_ent(alpha * n_samples * y[i] * theta[i])
    d_obj /= n_samples
    return d_obj


def mcp_pen(x, threshold, gamma=1.2):
    """ penalty value for mcp regularization
        Remind that gamma > 1
    """
    if isinstance(x, np.ndarray):
        z = (0.5 * threshold ** 2 * gamma) * np.ones(x.shape)
        j = np.abs(x) < gamma * threshold
        z[j] = threshold * np.abs(x[j]) - x[j] ** 2 / (2 * gamma)
    else:
        z = (0.5 * threshold ** 2 * gamma)
        if np.abs(x) < gamma * threshold:
            z = threshold * np.abs(x) - x ** 2 / (2 * gamma)
    return z


def smooth_hinge(x):
    val = np.zeros(len(x))
    val[x <= 0.0] = 0.5 - x[x <= 0.0]
    boole = np.logical_and(x > 0.0, x <= 1)
    val[boole] = 0.5 * (1 - x[boole]) ** 2

    return val


def derivative_smooth_hinge(x):
    deriv = np.zeros(len(x))
    deriv[x <= 0.0] = -1.0
    boole = np.logical_and(x > 0.0, x <= 1)
    deriv[boole] = -1.0 + x[boole]
    return deriv


def smooth_hinge_loss(X, y, beta):
    n_samples, n_features = X.shape
    val = 0
    grad = np.zeros(n_features)
    for i in range(n_samples):
        val += smooth_hinge((X[i, :].T @ beta) * y[i])
        grad += derivative_smooth_hinge(
            (X[i, :].T @ beta) * y[i]) * X[i, :] * y[i]
    val /= X.shape[0]
    grad /= X.shape[0]
    return val, grad


@njit
def mcp_prox(x, threshold, gamma=1.2):
    """MCP-proximal operator function, as a constraint gamma >1."""
    y = np.sign(x) * np.maximum(np.abs(x) - threshold, 0) / (1 - 1 / gamma)
    if np.abs(x) > gamma * threshold:
        y = x
    return y


@njit
def mcp_dalpha(x, alpha, gamma):
    if np.abs(x) >= alpha * gamma:
        return 0
    else:
        return - np.sign(x) / (1 - 1 / gamma)


@njit
def mcp_dgamma(x, alpha, gamma):
    if np.abs(x) >= alpha * gamma:
        return 0
    else:
        return - ST(x, alpha) / (gamma - 1) ** 2


@njit
def mcp_dx(x, alpha, gamma):
    if np.abs(x) >= alpha * gamma:
        return 1
    else:
        return np.abs(np.sign(x)) / (1 - 1 / gamma)


@njit
def init_dbeta0_new_p(jac0, mask, mask_old):
    mask_both = np.logical_and(mask_old, mask)
    size_mat = mask.sum()
    dbeta0_new = np.zeros((size_mat, size_mat))
    count = 0
    count_old = 0
    n_features = mask.shape[0]
    for j in range(n_features):
        if mask_both[j]:
            dbeta0_new[count, :] = init_dbeta0_new(
                jac0[count_old, :], mask, mask_old)
        if mask_old[j]:
            count_old += 1
        if mask[j]:
            count += 1
    return dbeta0_new


@njit
def init_dbeta0_new(dbeta0, mask, mask_old):
    mask_both = np.logical_and(mask_old, mask)
    size_mat = mask.sum()
    dbeta0_new = np.zeros(size_mat)
    count = 0
    count_old = 0
    n_features = mask.shape[0]
    for j in range(n_features):
        if mask_both[j]:
            dbeta0_new[count] = dbeta0[count_old]
        if mask_old[j]:
            count_old += 1
        if mask[j]:
            count += 1
    return dbeta0_new


def iou(supp1, supp2):
    return np.logical_and(
        supp1, supp2).sum() / np.logical_or(supp1, supp2).sum()


def iou_beta(beta1, beta2):
    supp1 = beta1 != 0
    supp2 = beta2 != 0
    return np.logical_and(
        supp1, supp2).sum() / np.logical_or(supp1, supp2).sum()


class Monitor():
    """
    Class used to store computed metrics at each iteration of the outer loop.
    """

    def __init__(self, callback=None):
        self.t0 = time.time()
        self.objs = []   # TODO rename, use self.value_outer?
        self.times = []
        self.log_alphas = []
        self.grads = []
        self.callback = callback
        self.acc_vals = []
        self.all_betas = []

    def __call__(
            self, obj, grad, mask=None, dense=None, log_alpha=None,
            acc_val=None, acc_test=None):
        self.objs.append(obj)
        try:
            self.log_alphas.append(log_alpha.copy())
        except Exception:
            self.log_alphas.append(log_alpha)
        self.times.append(time.time() - self.t0)
        self.grads.append(grad)
        if self.callback is not None:
            self.callback(obj, grad, mask, dense, log_alpha)
        if acc_val is not None:
            self.acc_vals.append(acc_val)
        if acc_test is not None:
            self.acc_vals.append(acc_test)


class WarmStart():
    """
    Class used to warm start all algorithms.
    """

    def __init__(self):
        self.beta_old = None
        self.beta_old2 = None
        self.dbeta_old = None
        self.dbeta_old2 = None
        self.mask_old = None
        self.mask_old2 = None
        self.sol_lin_sys = None
        self.sol_lin_sys2 = None

    def __call__(
            self, mask_old, beta_old, dbeta_old=None, mask_old2=None,
            beta_old2=None, dbeta_old2=None):
        """
        Here we save te masks of the active coefficients, the active
        coefficients of the regressions coefficients, and the active
        coefficient of the Jacobians.
        For the SURE criterion there are 2 optimization problem to solve
        """
        self.mask_old = mask_old
        self.beta_old = beta_old
        self.dbeta_old = dbeta_old
        self.mask_old2 = mask_old2
        self.beta_old2 = beta_old2
        self.dbeta_old2 = dbeta_old2
        return self.beta_old

    def set_sol_lin_sys(self, sol_lin_sys, sol_lin_sys2=None):
        """
        For the implicit differentiation the solution of the previous
        linear system can be used as a warm start for the next conjugate
        gradient.
        """
        self.sol_lin_sys = sol_lin_sys
        self.sol_lin_sys2 = sol_lin_sys2
