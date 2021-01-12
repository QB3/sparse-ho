import cvxpy as cp
import numpy as np
import torch
from cvxpylayers.torch import CvxpyLayer
# from sklearn.model_selection import train_test_split

torch.set_default_dtype(torch.double)
np.set_printoptions(precision=3, suppress=True)


# torch.manual_seed(0)
# np.random.seed(0)
# n = 2
# N = 60
# X, y = make_blobs(N, n, centers=np.array([[2, 2], [-2, -2]]), cluster_std=3)
# Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=.5)

# Xtrain, Xtest, ytrain, ytest = map(
#     torch.from_numpy, [Xtrain, Xtest, ytrain, ytest])
# Xtrain.requires_grad_(True)
# m = Xtrain.shape[0]

# a = cp.Variable((n, 1))
# b = cp.Variable((1, 1))
# X = cp.Parameter((m, n))
# Y = ytrain.numpy()[:, np.newaxis]

# log_likelihood = (1. / m) * cp.sum(
#     cp.multiply(Y, X @ a + b) - cp.logistic(X @ a + b)
# )
# regularization = - 0.1 * cp.norm(a, 1) - 0.1 * cp.sum_squares(a)
# prob = cp.Problem(cp.Maximize(log_likelihood + regularization))
# fit_logreg = CvxpyLayer(prob, [X], [a, b])

# torch.manual_seed(0)
# np.random.seed(0)
# n = 1
# N = 60
# X = np.random.randn(N, n)
# theta = np.random.randn(n)
# y = X @ theta + .5 * np.random.randn(N)
# Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=.5)


def enet_cvx_py(X, y, alpha, idx_train, idx_val):
    N, n = X.shape
    Xtrain, Xtest, ytrain, ytest = map(
        torch.from_numpy, [
            X[idx_train, :], X[idx_val], y[idx_train], y[idx_val]])
    Xtrain.requires_grad_(True)

    m = Xtrain.shape[0]

    # set up variables and parameters
    a = cp.Variable(n)
    b = cp.Variable()
    X = cp.Parameter((m, n))
    Y = cp.Parameter(m)
    lam = cp.Parameter(nonneg=True)
    alpha = cp.Parameter(nonneg=True)

    # set up objective
    loss = (1/m)*cp.sum(cp.square(X @ a + b - Y))
    reg = lam * cp.norm1(a) + alpha * cp.sum_squares(a)
    objective = loss + reg

    # set up constraints
    constraints = []

    prob = cp.Problem(cp.Minimize(objective), constraints)

    # convert into pytorch layer in one line
    fit_lr = CvxpyLayer(prob, [X, Y, lam, alpha], [a, b])
    # this object is now callable with pytorch tensors
    fit_lr(Xtrain, ytrain, torch.zeros(1), torch.zeros(1))


    # sweep over values of alpha, holding lambda=0, evaluating the gradient
    # along the way
    # alphas = [1., 1.]
    # alphas = np.logspace(-3, 2, 200)
    test_losses = []
    grads = []
    # for alpha_vals in alphas:
    # print(alpha_vals)
    alpha_vals = 1.
    alpha_lambda_tch = torch.tensor(
        [alpha_vals, alpha_vals], requires_grad=True)
    alpha_lambda_tch.grad = None
    a_tch, b_tch = fit_lr(
        Xtrain, ytrain, alpha_lambda_tch[0], alpha_lambda_tch[1])
    test_loss = (Xtest @ a_tch.flatten() + b_tch - ytest).pow(2).mean()
    test_loss.backward()
    test_losses.append(test_loss.item())
    grad_alpha_lmabda = alpha_lambda_tch.grad
    grads.append(grad_alpha_lmabda)
