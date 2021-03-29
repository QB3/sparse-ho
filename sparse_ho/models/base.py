from abc import ABC, abstractmethod


class BaseModel(ABC):

    @abstractmethod
    def __init__(cls):
        pass

    @abstractmethod
    def get_mv(cls, *args, **kwargs):
        return NotImplemented
