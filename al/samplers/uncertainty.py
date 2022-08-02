from .abstract_sampler import AbstractSampler
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class LeastConfidentSampler(AbstractSampler):
    name = "least_confident"

    def select_batch(self, X_unlab, batch_size, model, multilabel, **kwargs):
        if len(X_unlab) <= batch_size:
            return np.arange(len(X_unlab))

        probs = model.predict_proba(X_unlab)
        max_probs = self.get_weights(probs, multilabel)

        # Retrieve `batch_size` instances with lowest posterior probabilities.
        top_n = np.argpartition(max_probs, batch_size)[:batch_size]
        return top_n

    def get_weights(self, probs, multilabel, **kwargs):
        if multilabel:
            max_binary_probs = np.where(probs > 0.5, probs, 1 - probs)
            max_probs = np.prod(max_binary_probs, axis=1)
        else:
            max_probs = np.max(probs, axis=1)

        scaler = MinMaxScaler()
        max_probs = scaler.fit_transform(max_probs.reshape(-1, 1)).reshape(-1)
        return max_probs


class MarginSampler(AbstractSampler):
    name = "margin"

    def select_batch(self, X_unlab, batch_size, model, multilabel, **kwargs):
        if len(X_unlab) <= batch_size:
            return np.arange(len(X_unlab))

        if not multilabel and hasattr(model, "decision_function"):
            probs = model.decision_function(X_unlab)
        else:
            # the model doesn't have decision_function implemented, work with probabilities
            probs = model.predict_proba(X_unlab)
        min_margin = self.get_weights(probs, multilabel)

        # Retrieve `batch_size` instances with smallest margins.
        top_n = np.argpartition(min_margin, batch_size)[:batch_size]
        return top_n

    def get_weights(self, probs, multilabel, **kwargs):
        if multilabel:
            max_binary_probs = np.where(probs > 0.5, probs, 1 - probs)
            second_max_binary_probs = np.copy(max_binary_probs)
            # Take 1-p of the smallest probability in each row
            second_max_binary_probs[
                np.arange(second_max_binary_probs.shape[0]),
                np.argmin(second_max_binary_probs, axis=1),
            ] = (
                1
                - second_max_binary_probs[
                    np.arange(second_max_binary_probs.shape[0]),
                    np.argmin(second_max_binary_probs, axis=1),
                ]
            )
            max_probs = np.prod(max_binary_probs, axis=1)
            second_max_probs = np.prod(second_max_binary_probs, axis=1)
            min_margin = max_probs - second_max_probs
        else:
            if len(probs.shape) == 1:
                # If decision_function was used and task is binary
                min_margin = abs(probs)
            else:
                sort_probs = np.sort(probs, 1)[:, -2:]
                min_margin = sort_probs[:, 1] - sort_probs[:, 0]

        scaler = MinMaxScaler()
        min_margin = scaler.fit_transform(min_margin.reshape(-1, 1)).reshape(-1)
        return min_margin


class EntropySampler(AbstractSampler):
    name = "entropy"

    def select_batch(self, X_unlab, batch_size, model, multilabel, **kwargs):
        if len(X_unlab) <= batch_size:
            return np.arange(len(X_unlab))

        probs = model.predict_proba(X_unlab)

        entropies = self.get_weights(probs, multilabel)

        # Retrieve `batch_size` instances with lowest negative entropies.
        top_n = np.argpartition(entropies, batch_size)[:batch_size]
        return top_n

    def get_weights(self, probs, multilabel, **kwargs):
        # Clip for numerical stability.
        probs = np.clip(probs, a_min=1e-6, a_max=1 - 1e-6)
        if multilabel:
            binary_entropies = -probs * np.log(probs) - (1 - probs) * np.log(1 - probs)
            entropies = np.sum(binary_entropies, axis=1)
        else:
            entropies = np.sum(-probs * np.log(probs), axis=1)

        # We are selecting batch with argmin
        entropies = -entropies
        scaler = MinMaxScaler()
        entropies = scaler.fit_transform(entropies.reshape(-1, 1)).reshape(-1)
        return entropies


class MultilabelUncertaintySampler(AbstractSampler):
    name = "multilabel_uncertainty"

    def select_batch(self, X_unlab, batch_size, model, **kwargs):
        if len(X_unlab) <= batch_size:
            return np.arange(len(X_unlab))

        probs = model.predict_proba(X_unlab)
        min_margin = self.get_weights(probs)
        top_n = np.argpartition(min_margin, batch_size)[:batch_size]
        return top_n

    def get_weights(self, probs, **kwargs):
        min_margin = np.sum(abs(probs - 0.5), axis=1)
        scaler = MinMaxScaler()
        min_margin = scaler.fit_transform(min_margin.reshape(-1, 1)).reshape(-1)
        return min_margin
