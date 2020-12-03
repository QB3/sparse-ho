from abc import ABC, abstractmethod


class BaseOptimizer(ABC):

    @abstractmethod
    def __init__(cls):
        pass

    @abstractmethod
    def _grad_search(
            self, _get_val_grad, proj_hyperparam, log_alpha0, monitor):
        return NotImplemented
