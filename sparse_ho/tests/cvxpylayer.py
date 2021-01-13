import cvxpy as cp
import numpy as np
import torch
from cvxpylayers.torch import CvxpyLayer

torch.set_default_dtype(torch.double)
# np.set_printoptions(precision=3, suppress=True)


def enet_cvxpy(X, y, lambda_alpha, idx_train, idx_val):
    Xtrain, Xtest, ytrain, ytest = map(
        torch.from_numpy, [
            X[idx_train, :], X[idx_val], y[idx_train], y[idx_val]])

    m, n = Xtrain.shape

    # set up variables and parameters
    beta = cp.Variable(n)
    X = cp.Parameter((m, n))
    Y = cp.Parameter(m)
    lambda_cp = cp.Parameter(nonneg=True)
    alpha_cp = cp.Parameter(nonneg=True)

    # set up objective
    loss = (1/(2 * m)) * cp.sum(cp.square(X @ beta - Y))
    reg = lambda_cp * cp.norm1(beta) + alpha_cp * cp.sum_squares(beta) / 2
    objective = loss + reg

    problem = cp.Problem(cp.Minimize(objective))
    layer = CvxpyLayer(problem, [X, Y, lambda_cp, alpha_cp], [beta])

    lambda_alpha_th = torch.tensor(lambda_alpha, requires_grad=True)
    beta_ = layer(Xtrain, ytrain, lambda_alpha_th[0], lambda_alpha_th[1])

    test_loss = (Xtest @ beta_[0] - ytest).pow(2).mean()
    test_loss.backward()

    grad_alpha_lambda = lambda_alpha_th.grad
    val = test_loss.detach().numpy()
    grad = np.array(grad_alpha_lambda)
    return val, grad


def weighted_lasso_cvxpy(X, y, lambdas, idx_train, idx_val):
    Xtrain, Xtest, ytrain, ytest = map(
        torch.from_numpy, [
            X[idx_train, :], X[idx_val], y[idx_train], y[idx_val]])

    m, n = Xtrain.shape

    # set up variables and parameters
    beta = cp.Variable(n)
    X = cp.Parameter((m, n))
    Y = cp.Parameter(m)
    lambda_cp = cp.Parameter(shape=n, nonneg=True)

    # set up objective
    loss = (1/(2 * m)) * cp.sum(cp.square(X @ beta - Y))
    reg = lambda_cp @ cp.abs(beta)
    objective = loss + reg

    problem = cp.Problem(cp.Minimize(objective))

    layer = CvxpyLayer(problem, [X, Y, lambda_cp], [beta])

    # solve the problem
    lambdas_th = torch.tensor(lambdas, requires_grad=True)
    beta_ = layer(Xtrain, ytrain, lambdas_th)

    test_loss = (Xtest @ beta_[0] - ytest).pow(2).mean()
    test_loss.backward()

    grad_alpha_lambda = lambdas_th.grad
    val = test_loss.detach().numpy()
    grad = np.array(grad_alpha_lambda)
    return val, grad


def logreg_cvxpy(X, y, alpha, idx_train, idx_val):
    Xtrain, Xtest, ytrain, ytest = map(
        torch.from_numpy, [
            X[idx_train, :], X[idx_val], y[idx_train], y[idx_val]])

    m, n = Xtrain.shape

    beta = cp.Variable(n)
    alpha_cp = cp.Parameter(nonneg=True)

    loss = cp.sum(cp.logistic(X @ beta) - cp.multiply(y, X @ beta)) / m
    reg = alpha_cp * cp.norm(beta, 1)
    objective = loss + reg
    problem = cp.Problem(cp.Minimize(objective))

    assert problem.is_dpp()
    layer = CvxpyLayer(problem, parameters=[alpha_cp], variables=[beta])

    # solve the problem
    alpha_th = torch.tensor(alpha, requires_grad=True)
    solution, = layer(alpha_th)

    # Evaluate test loss and backward
    loss = torch.nn.modules.loss.BCEWithLogitsLoss(reduction='mean')
    val = loss(Xtest @ solution, ytest)
    val.backward()

    val = val.detach().numpy()
    grad = alpha_th.grad
    return val, grad
