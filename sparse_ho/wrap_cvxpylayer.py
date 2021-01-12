import cvxpy as cp
import numpy as np
import torch
from cvxpylayers.torch import CvxpyLayer

torch.set_default_dtype(torch.double)
np.set_printoptions(precision=3, suppress=True)


def enet_cvx_py(X, y, lambda_alpha, idx_train, idx_val):
    N, n = X.shape
    Xtrain, Xtest, ytrain, ytest = map(
        torch.from_numpy, [
            X[idx_train, :], X[idx_val], y[idx_train], y[idx_val]])
    Xtrain.requires_grad_(True)

    m = Xtrain.shape[0]

    # set up variables and parameters
    a = cp.Variable(n)
    # b = cp.Variable()
    X = cp.Parameter((m, n))
    Y = cp.Parameter(m)
    lam = cp.Parameter(nonneg=True)
    alpha = cp.Parameter(nonneg=True)

    # set up objective
    loss = (1/m)*cp.sum(cp.square(X @ a - Y))
    reg = lam * cp.norm1(a) + alpha * cp.sum_squares(a) / 2
    objective = loss + reg

    # set up constraints
    constraints = []

    prob = cp.Problem(cp.Minimize(objective), constraints)

    # convert into pytorch layer in one line
    fit_lr = CvxpyLayer(prob, [X, Y, lam, alpha], [a])
    # this object is now callable with pytorch tensors
    fit_lr(Xtrain, ytrain, torch.zeros(1), torch.zeros(1))

    # sweep over values of alpha, holding lambda=0, evaluating the gradient
    # along the way

    test_losses = []

    lambda_alpha_tch = torch.tensor(
        lambda_alpha, requires_grad=True)
    lambda_alpha_tch.grad = None
    a_tch = fit_lr(
        Xtrain, ytrain, lambda_alpha_tch[0], lambda_alpha_tch[1])

    test_loss = (Xtest @ a_tch[0] - ytest).pow(2).mean()
    test_loss.backward()
    test_losses.append(test_loss.item())
    grad_alpha_lambda = lambda_alpha_tch.grad
    # grads.append(grad_alpha_lambda)
    return np.array(grad_alpha_lambda)
