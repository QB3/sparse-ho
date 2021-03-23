# License: BSD 3 clause

# from itertools import cycle

import os
import numpy as np

from sklearn.linear_model import Lasso, ElasticNet, LogisticRegression
from sklearn import datasets
from sklearn.svm import l1_min_c
# load diabetes dataset for regression model
X, y = datasets.load_diabetes(return_X_y=True)
# Standardize data (easier to set the l1_ratio parameter)
X /= X.std(axis=0)
n_samples = len(y)

dict_X = {}
dict_X["lasso"] = X
dict_X["enet"] = X

dict_y = {}
dict_y["lasso"] = y
dict_y["enet"] = y
# load iris for classification model

dict_X["logreg"] = X
dict_y["logreg"] = (y > 100) * 1.0

name_models = ["lasso", "enet", "logreg"]

dict_models = {}
dict_models["lasso"] = Lasso(fit_intercept=False, warm_start=False)
dict_models["logreg"] = LogisticRegression(
    penalty="l1", fit_intercept=False, warm_start=False, solver='liblinear',
    max_iter=10000, tol=1e-9)
dict_models["enet"] = ElasticNet(fit_intercept=False, warm_start=False)

# Compute alpha_max

dict_alpha_max = {}
dict_alpha_max["lasso"] = np.max(
        np.abs(dict_X["lasso"].T.dot(dict_y["lasso"]))) / n_samples
dict_alpha_max["logreg"] = 1 / l1_min_c(dict_X['logreg'], dict_y['logreg'])
dict_alpha_max["enet"] = dict_alpha_max["lasso"]
# Setting grid of values for alpha
n_alphas = 1000
p_alpha_min = 1e-5
p_alphas = np.geomspace(1, p_alpha_min, n_alphas)

os.makedirs('./results', exist_ok=True)

for name_model in name_models:
    alphas = dict_alpha_max[name_model] * p_alphas
    coefs = []

    print("Starting path computation for ", name_model)
    for alpha in alphas:
        if name_model == "lasso":
            dict_models[name_model].set_params(alpha=alpha)
        elif name_model == "enet":
            l1_ratio = 0.8
            alpha1 = 0.2 * alpha / l1_ratio
            dict_models[name_model].set_params(alpha=alpha+alpha1,
                                               l1_ratio=alpha/(alpha+alpha1))
        elif name_model == "logreg":
            dict_models[name_model].set_params(C=1/(alpha))

        dict_models[name_model].fit(dict_X[name_model], dict_y[name_model])
        coefs.append(dict_models[name_model].coef_)
    print("End path computation for ", name_model)

    coefs = np.array(coefs)
    np.save("results/coefs_%s" % name_model, coefs)
    np.save("results/alphas_%s" % name_model, alphas)
