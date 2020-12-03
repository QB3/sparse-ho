from abc import ABC, abstractmethod


class BaseModel(ABC):

    @abstractmethod
    def __init__(cls):
        pass

    @abstractmethod
    def get_hessian(cls, *args, **kwargs):
        return NotImplemented
