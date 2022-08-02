import numpy as np

from al.sampling_methods.sampling_def import SamplingMethod


class UniformSampling(SamplingMethod):
    def __init__(self, X, y=None, seed=None):
        super(UniformSampling, self).__init__(X, y, seed)
        self.name = "uniform"

    def select_batch_(self, already_selected, N, **kwargs):
        """Returns batch of randomly sampled datapoints.

        Assumes that data has already been shuffled.

        Args:
          already_selected: index of datapoints already selected
          N: batch size

        Returns:
          indices of points selected to label
        """

        # This is uniform given the remaining pool but biased wrt the entire pool.
        sample = [i for i in range(self.X.shape[0]) if i not in already_selected]
        return sample[0:N]
