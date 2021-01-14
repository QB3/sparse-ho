import cvxpy as cp
import numpy as np
import torch
from cvxpylayers.torch import CvxpyLayer

torch.set_default_dtype(torch.double)


def lasso_cvxpy(X, y, lambd, idx_train, idx_val):
    val, grad = enet_cvxpy(X, y, [float(lambd), 0], idx_train, idx_val)
    return val, grad[0]


def enet_cvxpy(X, y, lambda_alpha, idx_train, idx_val):
    Xtrain, Xtest, ytrain, ytest = map(
        torch.from_numpy, [
            X[idx_train, :], X[idx_val], y[idx_train], y[idx_val]])

    n_samples_train, n_features = Xtrain.shape

    # set up variables and parameters
    beta_cp = cp.Variable(n_features)
    lambda_cp = cp.Parameter(nonneg=True)
    alpha_cp = cp.Parameter(nonneg=True)

    # set up objective
    loss = ((1 / (2 * n_samples_train)) *
            cp.sum(cp.square(Xtrain @ beta_cp - ytrain)))
    reg = (lambda_cp * cp.norm1(beta_cp) +
           alpha_cp * cp.sum_squares(beta_cp) / 2)
    objective = loss + reg

    # define problem
    problem = cp.Problem(cp.Minimize(objective))
    assert problem.is_dpp()

    # solve problem
    layer = CvxpyLayer(problem, [lambda_cp, alpha_cp], [beta_cp])
    lambda_alpha_th = torch.tensor(lambda_alpha, requires_grad=True)
    beta_, = layer(lambda_alpha_th[0], lambda_alpha_th[1])

    # get test loss and it's gradient
    test_loss = (Xtest @ beta_ - ytest).pow(2).mean()
    test_loss.backward()

    val = test_loss.detach().numpy()
    grad = np.array(lambda_alpha_th.grad)
    return val, grad


def weighted_lasso_cvxpy(X, y, lambdas, idx_train, idx_val):
    Xtrain, Xtest, ytrain, ytest = map(
        torch.from_numpy, [
            X[idx_train, :], X[idx_val], y[idx_train], y[idx_val]])

    n_samples_train, n_features = Xtrain.shape

    # set up variables and parameters
    beta_cp = cp.Variable(n_features)
    lambdas_cp = cp.Parameter(shape=n_features, nonneg=True)

    # set up objective
    loss = ((1 / (2 * n_samples_train)) *
            cp.sum(cp.square(Xtrain @ beta_cp - ytrain)))
    reg = lambdas_cp @ cp.abs(beta_cp)
    objective = loss + reg

    # define problem
    problem = cp.Problem(cp.Minimize(objective))
    assert problem.is_dpp()

    # solve problem
    layer = CvxpyLayer(problem, [lambdas_cp], [beta_cp])
    lambdas_th = torch.tensor(lambdas, requires_grad=True)
    beta_, = layer(lambdas_th)

    # get test loss and it's gradient
    test_loss = (Xtest @ beta_ - ytest).pow(2).mean()
    test_loss.backward()

    val = test_loss.detach().numpy()
    grad = np.array(lambdas_th.grad)
    return val, grad


def logreg_cvxpy(X, y, alpha, idx_train, idx_val):
    alpha = float(alpha)
    assert np.all(np.unique(y) == np.array([-1, 1]))
    Xtrain, Xtest, ytrain, ytest = map(
        torch.from_numpy, [
            X[idx_train, :], X[idx_val], y[idx_train], y[idx_val]])

    n_samples_train, n_features = Xtrain.shape

    # set up variables and parameters
    beta_cp = cp.Variable(n_features)
    alpha_cp = cp.Parameter(nonneg=True)

    # set up objective
    loss = cp.sum(
        cp.logistic(cp.multiply(-ytrain, Xtrain @ beta_cp))) / n_samples_train
    # loss = cp.sum(cp.logistic(Xtrain @ beta_cp) -
    #               cp.multiply(ytrain, Xtrain @ beta_cp)) / n_samples_train
    reg = alpha_cp * cp.norm(beta_cp, 1)
    objective = loss + reg

    # define problem
    problem = cp.Problem(cp.Minimize(objective))
    assert problem.is_dpp()

    # solve problem
    layer = CvxpyLayer(problem, parameters=[alpha_cp], variables=[beta_cp])
    alpha_th = torch.tensor(alpha, requires_grad=True)
    beta_, = layer(alpha_th)

    # get test loss and it's gradient
    # loss_th = torch.nn.modules.loss.BCEWithLogitsLoss(reduction='mean')
    # test_loss = loss_th(Xtest @ beta_, ytest)
    # ytest[ytest == 0] = -1
    test_loss = torch.mean(torch.log(1 + torch.exp(-ytest * (Xtest @ beta_))))
    test_loss.backward()

    val = test_loss.detach().numpy()
    grad = np.array(alpha_th.grad)
    return val, grad
