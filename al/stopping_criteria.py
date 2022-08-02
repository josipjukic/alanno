import abc
import math
import numpy as np


class StoppingCriterion(abc.ABC):
    @abc.abstractmethod
    def __init__(
        self, train_size=None, batch_size=None, seed_batch=None, X=None, y=None
    ):
        self.train_size = train_size
        self.batch_size = batch_size
        self.seed_batch = seed_batch
        self.X = X
        self.y = y
        self.annotated_total = 0
        self.current_batch = 0

    def next_state(self, annotated_count):
        self.annotated_total += annotated_count
        self.current_batch += 1

    @abc.abstractmethod
    def is_over(self, **kwargs):
        pass

    @abc.abstractmethod
    def reset(self):
        pass


class VarianceSurge(StoppingCriterion):
    def __init__(
        self,
        train_size,
        batch_size,
        seed_batch,
        X,
        y,
        threshold=5e-3,
        train_horizon=1.0,
    ):
        super(VarianceSurge, self).__init__(train_size, batch_size, seed_batch, X, y)
        self.n_batches = 1 + int(
            math.ceil((train_horizon * train_size - seed_batch) / batch_size)
        )
        self.vars = []
        self.labeled = []
        self.threshold = threshold

    def is_over(self, model, batch_selected_inds, selected_inds, **kwargs):
        if check_fitted(model) and len(batch_selected_inds) > 0:
            Xs, ys = self.X[batch_selected_inds], self.y[batch_selected_inds]
            self.labeled.append(len(selected_inds))
            probs = model.predict_proba(Xs)
            confidence = np.max(probs, axis=1)
            var = np.var(confidence)
            self.vars.append(var)
            return var > self.threshold

        return False

    def reset(self):
        self.current_batch = 0


class QueryCount(StoppingCriterion):
    def __init__(self, train_size, batch_size, seed_batch, train_horizon=1.0):
        super(QueryCount, self).__init__(train_size, batch_size, seed_batch)
        self.n_batches = 1 + int(
            math.ceil((train_horizon * train_size - seed_batch) / batch_size)
        )

    def is_over(self, **kwargs):
        return self.current_batch >= self.n_batches

    def reset(self):
        self.current_batch = 0


class UsersChoice(StoppingCriterion):
    def __init__(self):
        super(UsersChoice, self).__init__()

    def is_over(self):
        return False

    def reset(self):
        self.current_batch = 0


def check_fitted(clf):
    return hasattr(clf, "classes_")
