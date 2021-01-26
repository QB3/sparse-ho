# from sparse_ho.models import Lasso
# from sparse_ho.utils import Monitor
# from sparse_ho.ho import grad_search


# class LassoAuto():
#     """Automatic hyperparameter selction for the Lasso
#     The optimization objective for Lasso is:
#     (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

#     Parameters
#     ----------
#     estimator: instance of ``sklearn.base.BaseEstimator``
#         An estimator that follows the scikit-learn API.
#     criterion: instance of Base
#         # XXX TODO
#     alpha0: float
#         initialization of alpha of the optimization
#     """

#     def __init__(self, estimator, criterion, alpha0=1):
#         self.estimator = estimator
#         self.criterion = criterion
#         self.alpha0 = alpha0
#         self.monitor = None

#     def fit(self, X, y, algo, optimizer):
#         model = Lasso(estimator=self.estimator)
#         monitor = Monitor()
#         grad_search(
#             algo, self.criterion, model, optimizer, X, y, self.alpha0, monitor)

#         self.monitor = monitor
