from abc import ABC, abstractmethod


class BaseCriterion(ABC):

    @abstractmethod
    def __init__(cls):
        pass

    @abstractmethod
    def get_val_outer(cls, *args, **kwargs):
        return NotImplemented

    @abstractmethod
    def get_val(cls, *args, **kwargs):
        return NotImplemented

    @abstractmethod
    def get_val_grad(cls, *args, **kwargs):
        return NotImplemented

    # @abstractmethod
    # def proj_hyperparam(cls, *args, **kwargs):
    #     return NotImplemented
