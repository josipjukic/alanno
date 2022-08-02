from .abstract_sampler import AbstractSampler
import numpy as np


class RandomSampler(AbstractSampler):
    name = "random"

    def select_batch(self, X_unlab, batch_size, **kwargs):
        if len(X_unlab) <= batch_size:
            return np.arange(len(X_unlab))

        unlab_inds = np.arange(X_unlab.shape[0])
        return np.random.choice(unlab_inds, size=batch_size, replace=False)

    def get_weights(self, X_unlab, **kwargs):
        return np.ones(len(X_unlab))
