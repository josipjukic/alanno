import numpy as np
from .sampling_def import SamplingMethod


class MarginAL(SamplingMethod):
    def __init__(self, seed):
        super(MarginAL, self).__init__("margin", seed)

    def select_batch_(self, X, model, N, multilabel, **kwargs):
        """Returns batch of datapoints with smallest margin/highest uncertainty.

        For binary classification, can just take the absolute distance to decision
        boundary for each point.
        For multiclass classification, must consider the margin between distance for
        top two most likely classes.

        Args:
          model: scikit learn model with decision_function implemented
          already_selected: index of datapoints already selected
          N: batch size

        Returns:
          indices of points selected to add using margin active learner
        """

        min_margin = margin(model, X, multilabel=multilabel)
        rank_ind = np.argsort(min_margin)
        rank_ind = rank_ind.tolist()
        active_samples = rank_ind[0:N]
        return active_samples


def margin(model, X, multilabel):
    min_margin = None
    try:
        distances = model.decision_function(X)
        if multilabel:
            min_margin = np.sum(abs(distances), axis=1)
        elif len(distances.shape) == 1:
            min_margin = abs(distances)
        else:
            sort_distances = np.sort(distances, 1)[:, -2:]
            min_margin = sort_distances[:, 1] - sort_distances[:, 0]

    except:
        # the model doesn't have decision_function implemented, work with probabilities
        probs = model.predict_proba(X)
        if multilabel:
            min_margin = np.sum(abs(probs - 0.5), axis=1)
        else:
            sort_probs = np.sort(probs, 1)[:, -2:]
            min_margin = sort_probs[:, 1] - sort_probs[:, 0]

    return min_margin
