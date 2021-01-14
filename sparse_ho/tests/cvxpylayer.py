import cvxpy as cp
import numpy as np
import torch
from cvxpylayers.torch import CvxpyLayer
from sklearn.utils import check_random_state

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


def lasso_sure_cvxpy(X, y, alpha, sigma, random_state=42):
    # lambda_alpha = [alpha, alpha]
    n_samples, n_features = X.shape
    epsilon = 2 * sigma / n_samples ** 0.3
    rng = check_random_state(random_state)
    delta = rng.randn(n_samples)

    y2 = y + epsilon * delta
    Xth, yth, y2th, deltath = map(torch.from_numpy, [X, y, y2, delta])

    n_samples_train, n_features = Xth.shape

    # set up variables and parameters
    beta_cp = cp.Variable(n_features)
    lambda_cp = cp.Parameter(nonneg=True)

    # set up objective
    loss = ((1 / (2 * n_samples_train)) * cp.sum(
        cp.square(Xth @ beta_cp - yth)))
    reg = lambda_cp * cp.norm1(beta_cp)
    objective = loss + reg

    # define problem
    problem1 = cp.Problem(cp.Minimize(objective))
    assert problem1.is_dpp()

    # solve problem1
    layer = CvxpyLayer(problem1, [lambda_cp], [beta_cp])
    alpha_th1 = torch.tensor(alpha, requires_grad=True)
    beta1, = layer(alpha_th1)

    # get test loss and it's gradient
    test_loss1 = (Xth @ beta1 - yth).pow(2).sum()
    test_loss1 -= 2 * sigma ** 2 / epsilon * (Xth @ beta1) @ deltath
    test_loss1.backward()
    val1 = test_loss1.detach().numpy()
    grad1 = np.array(alpha_th1.grad)

    # set up variables and parameters
    beta_cp = cp.Variable(n_features)
    lambda_cp = cp.Parameter(nonneg=True)

    # set up objective
    loss = ((1 / (2 * n_samples_train)) * cp.sum(
        cp.square(Xth @ beta_cp - y2th)))
    reg = lambda_cp * cp.norm1(beta_cp)
    objective = loss + reg

    # define problem
    problem2 = cp.Problem(cp.Minimize(objective))
    assert problem2.is_dpp()

    # solve problem2
    layer = CvxpyLayer(problem2, [lambda_cp], [beta_cp])
    alpha_th2 = torch.tensor(alpha, requires_grad=True)
    beta2, = layer(alpha_th2)

    # get test loss and it's gradient
    test_loss1 = 2 * sigma ** 2 / epsilon * (Xth @ beta2) @ deltath
    test_loss1.backward()
    val2 = test_loss1.detach().numpy()
    grad2 = np.array(alpha_th2.grad)

    val = val1 + val2
    grad = grad1 + grad2
    return val, grad

# import pytest
# import itertools

# import numpy as np
# from scipy.optimize import check_grad
# from scipy.sparse import csc_matrix
# from sklearn import linear_model
# import celer

# from sparse_ho.datasets.synthetic import get_synt_data
# from sparse_ho.models import Lasso, ElasticNet, WeightedLasso, SparseLogreg

# from sparse_ho import Forward, ImplicitForward, Implicit

# from sparse_ho.algo.forward import get_beta_jac_iterdiff
# from sparse_ho.algo.implicit_forward import get_beta_jac_fast_iterdiff
# from sparse_ho.algo.implicit import get_beta_jac_t_v_implicit
# from sparse_ho.criterion import (
#     HeldOutMSE, FiniteDiffMonteCarloSure, HeldOutLogistic)
# from sparse_ho.tests.cvxpylayer import \
#     enet_cvxpy, weighted_lasso_cvxpy, logreg_cvxpy, lasso_cvxpy


# # Generate data
# n_samples, n_features = 10, 10
# X, y, _, _, sigma_star = get_synt_data(
#     dictionary_type="Toeplitz", n_samples=n_samples,
#     n_features=n_features, n_times=1, n_active=5, rho=0.1,
#     SNR=3, seed=0)
# y = np.sign(y)
# idx_train = np.arange(0, n_features//2)
# idx_val = np.arange(n_features//2, n_features)

# # Set alpha for the Lasso
# alpha_max = (np.abs(X[idx_train, :].T @ y[idx_train])).max() / n_samples
# p_alpha = 0.8
# alpha = p_alpha * alpha_max
# sigma = 1

# val, grad = lasso_sure_cvxpy(X, y, alpha, sigma)

# # lasso_cvxpy(X, y, alpha, idx_train, idx_val)
