from abc import ABC, abstractmethod


class AbstractSampler(ABC):
    @abstractmethod
    def select_batch(self):
        raise NotImplementedError

    @abstractmethod
    def get_weights(self):
        raise NotImplementedError
