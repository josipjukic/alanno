from .abstract_sampler import AbstractSampler
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from .density import *
from .uncertainty import *


class CombinedSampler(AbstractSampler):
    name = "combined"

    def select_batch(
        self,
        X_unlab,
        samplers,
        batch_size,
        powers=None,
        model=None,
        multilabel=None,
        **kwargs,
    ):
        if len(X_unlab) <= batch_size:
            return np.arange(len(X_unlab))

        if model:
            probs = model.predict_proba(X_unlab)
        else:
            probs = None
        final_weights = self.get_weights(
            samplers=samplers,
            powers=powers,
            probs=probs,
            X_unlab=X_unlab,
            multilabel=multilabel,
            **kwargs,
        )

        # Retrieve `batch_size` instances with lowest final weight.
        top_n = np.argpartition(final_weights, batch_size)[:batch_size]
        return top_n

    def get_weights(self, samplers, powers=None, **kwargs):
        if not powers:
            powers = [1 for _ in range(len(samplers))]

        all_weights = []
        try:
            for sampler, power in zip(samplers, powers):
                sampler_weights = sampler.get_weights(**kwargs)
                sampler_weights = np.clip(sampler_weights, a_min=1e-6, a_max=1 - 1e-6)
                sampler_weights = np.power(sampler_weights, power)
                all_weights.append(sampler_weights)
        except NameError as e:
            raise ValueError(f"{e} for {sampler.name} sampler")

        all_weights = np.asarray(all_weights)
        final_weights = np.prod(all_weights, axis=0)

        scaler = MinMaxScaler()
        final_weights = scaler.fit_transform(final_weights.reshape(-1, 1)).reshape(-1)
        return final_weights


class CombinedEntropyDensitySampler(CombinedSampler):
    name = "combined_entropy_density"

    def select_batch(
        self, X_unlab, batch_size, model, multilabel, powers=None, **kwargs
    ):
        if len(X_unlab) <= batch_size:
            return np.arange(len(X_unlab))

        if not powers:
            powers = [1, 1]
        return super().select_batch(
            X_unlab=X_unlab,
            samplers=[EntropySampler(), DensitySampler()],
            batch_size=batch_size,
            powers=powers,
            model=model,
            multilabel=multilabel,
        )

    def get_weights(self, probs, X_unlab, multilabel, powers=None, **kwargs):
        if not powers:
            powers = [1, 1]
        return super().get_weights(
            samplers=[EntropySampler(), DensitySampler()],
            powers=powers,
            probs=probs,
            X_unlab=X_unlab,
            multilabel=multilabel,
        )
