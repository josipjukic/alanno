"""
Entropy based AL method.
Samples in batches by choosing the samples that maximize the entropy over the
probability of predicted classes.
"""

import numpy as np
from sampling_methods.sampling_def import SamplingMethod


class EntropyAL(SamplingMethod):
    def __init__(self, X, y, seed):
        self.X = X
        self.y = y
        self.name = "entropy"

    def select_batch_(self, model, already_selected, N, multilabel, **kwargs):
        """Returns the batch of datapoints with highest entropy / uncertainty.
        Args:
            model: scikit learn model with decision_function implemented
            already_selected: index of datapoints already selected
            N: batch size

        Returns:
            indices of points selected to add using entropy active learner
        """
        probs = model.predict_proba(self.X)
        # if not multilabel:
        #     # normalize - (OVR) trained on one-hots doesn't normalize probs
        #     probs = probs / np.sum(probs, axis=1, keepdims=1)

        # clip for numerical stability
        probs = np.clip(probs, a_min=1e-6, a_max=None)
        entropies = np.sum(-probs * np.log(probs), axis=1)
        # sort descending - higher entropy is better
        active_indices = np.argsort(entropies)[::-1]
        active_indices = [i for i in active_indices if i not in already_selected]
        return active_indices[0:N]
