import abc
import numpy as np


class SamplingMethod(abc.ABC):
    @abc.abstractmethod
    def __init__(self, name, seed, **kwargs):
        self.name = name
        self.seed = seed

    @abc.abstractmethod
    def select_batch_(self):
        return

    def select_batch(self, **kwargs):
        return self.select_batch_(**kwargs)
