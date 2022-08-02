from django.test import TestCase

import numpy as np
import difflib
from al.samplers.uncertainty import *
from al.samplers.density import *
from al.samplers.combined_sampler import *


def list_similarity(a, b, ref):
    """
    Test if list 'a' is more similar to a reference list (ref) than list 'b' is.
    """
    sim_a = difflib.SequenceMatcher(None, a, ref)
    sim_b = difflib.SequenceMatcher(None, b, ref)
    return sim_a.ratio() > sim_b.ratio()


class MockModel:
    def __init__(self, multilabel):
        self.multilabel = multilabel

    def predict_proba(self, X):
        if self.multilabel:
            return np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.5, 0.5, 0.5],
                    [0.5, 0.6, 0.6],
                    [0.3, 0.3, 0.2],
                    [0.1, 0.8, 0.9],
                    [0.2, 0.3, 0.3],
                ]
            )
        else:
            return np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.33, 0.33, 0.33],
                    [0.4, 0.4, 0.2],
                    [0.8, 0.1, 0.1],
                    [0.6, 0.3, 0.1],
                    [0.8, 0.1, 0.1],
                ]
            )


class Samplers(TestCase):
    def setUp(self):
        self.X = np.array(
            [[1, 2, 3], [0, 0, 0], [3, 2, 1], [1, 0, 0], [2, 3, 5], [2, 3, 5]]
        )

    def test_multiclass_leastConfident(self):
        sampler = LeastConfidentSampler()
        model = MockModel(False)
        probs = model.predict_proba(self.X)
        weights = sampler.get_weights(probs, False)
        self.assertEqual(6, len(weights))
        self.assertEqual(round(weights[3], 6), round(weights[-1], 6))
        self.assertEqual(0, np.argmax(weights))
        self.assertEqual(1, np.argmin(weights))

        batch = sampler.select_batch(self.X, 2, model, False)
        self.assertEqual(2, len(batch))
        self.assertIn(1, batch)
        self.assertIn(2, batch)

        batch = sampler.select_batch(self.X, len(self.X), model, False)
        self.assertEqual(len(self.X), len(batch))
        self.assertEqual(list(batch), list(range(len(self.X))))
        batch = sampler.select_batch(self.X, len(self.X) + 1, model, False)
        self.assertEqual(len(self.X), len(batch))
        self.assertEqual(list(batch), list(range(len(self.X))))

    def test_multilabel_leastConfident(self):
        sampler = LeastConfidentSampler()
        model = MockModel(True)
        probs = model.predict_proba(self.X)
        weights = sampler.get_weights(probs, True)
        self.assertEqual(6, len(weights))
        self.assertEqual(round(weights[3], 6), round(weights[-1], 6))
        self.assertEqual(0, np.argmax(weights))
        self.assertEqual(1, np.argmin(weights))

        batch = sampler.select_batch(self.X, 2, model, True)
        self.assertEqual(2, len(batch))
        self.assertIn(1, batch)
        self.assertIn(2, batch)

        batch = sampler.select_batch(self.X, len(self.X), model, True)
        self.assertEqual(len(self.X), len(batch))
        self.assertEqual(list(batch), list(range(len(self.X))))
        batch = sampler.select_batch(self.X, len(self.X) + 1, model, True)
        self.assertEqual(len(self.X), len(batch))
        self.assertEqual(list(batch), list(range(len(self.X))))

    def test_multiclass_margin(self):
        sampler = MarginSampler()
        model = MockModel(False)
        probs = model.predict_proba(self.X)
        weights = sampler.get_weights(probs, False)
        self.assertEqual(6, len(weights))
        self.assertEqual(round(weights[3], 6), round(weights[-1], 6))
        self.assertEqual(0, np.argmax(weights))
        self.assertEqual(1, np.argmin(weights))

        batch = sampler.select_batch(self.X, 2, model, False)
        self.assertEqual(2, len(batch))
        self.assertIn(1, batch)
        self.assertIn(2, batch)

        batch = sampler.select_batch(self.X, len(self.X), model, False)
        self.assertEqual(len(self.X), len(batch))
        self.assertEqual(list(batch), list(range(len(self.X))))
        batch = sampler.select_batch(self.X, len(self.X) + 1, model, False)
        self.assertEqual(len(self.X), len(batch))
        self.assertEqual(list(batch), list(range(len(self.X))))

    def test_multilabel_margin(self):
        sampler = MarginSampler()
        model = MockModel(True)
        probs = model.predict_proba(self.X)
        weights = sampler.get_weights(probs, True)
        self.assertEqual(6, len(weights))
        self.assertEqual(round(weights[3], 6), round(weights[-1], 6))
        self.assertEqual(0, np.argmax(weights))
        self.assertEqual(1, np.argmin(weights))

        batch = sampler.select_batch(self.X, 2, model, True)
        self.assertEqual(2, len(batch))
        self.assertIn(1, batch)
        self.assertIn(2, batch)

        batch = sampler.select_batch(self.X, len(self.X), model, True)
        self.assertEqual(len(self.X), len(batch))
        self.assertEqual(list(batch), list(range(len(self.X))))
        batch = sampler.select_batch(self.X, len(self.X) + 1, model, True)
        self.assertEqual(len(self.X), len(batch))
        self.assertEqual(list(batch), list(range(len(self.X))))

    def test_multiclass_entropy(self):
        sampler = EntropySampler()
        model = MockModel(False)
        probs = model.predict_proba(self.X)
        weights = sampler.get_weights(probs, False)
        self.assertEqual(6, len(weights))
        self.assertEqual(round(weights[3], 6), round(weights[-1], 6))
        self.assertEqual(0, np.argmax(weights))
        self.assertEqual(1, np.argmin(weights))

        batch = sampler.select_batch(self.X, 2, model, False)
        self.assertEqual(2, len(batch))
        self.assertIn(1, batch)
        self.assertIn(2, batch)

        batch = sampler.select_batch(self.X, len(self.X), model, False)
        self.assertEqual(len(self.X), len(batch))
        self.assertEqual(list(batch), list(range(len(self.X))))
        batch = sampler.select_batch(self.X, len(self.X) + 1, model, False)
        self.assertEqual(len(self.X), len(batch))
        self.assertEqual(list(batch), list(range(len(self.X))))

    def test_multilabel_entropy(self):
        sampler = EntropySampler()
        model = MockModel(True)
        probs = model.predict_proba(self.X)
        weights = sampler.get_weights(probs, True)
        self.assertEqual(6, len(weights))
        self.assertEqual(round(weights[3], 6), round(weights[-1], 6))
        self.assertEqual(0, np.argmax(weights))
        self.assertEqual(1, np.argmin(weights))

        batch = sampler.select_batch(self.X, 2, model, True)
        self.assertEqual(2, len(batch))
        self.assertIn(1, batch)
        self.assertIn(2, batch)

        batch = sampler.select_batch(self.X, len(self.X), model, True)
        self.assertEqual(len(self.X), len(batch))
        self.assertEqual(list(batch), list(range(len(self.X))))
        batch = sampler.select_batch(self.X, len(self.X) + 1, model, True)
        self.assertEqual(len(self.X), len(batch))
        self.assertEqual(list(batch), list(range(len(self.X))))

    def test_multilabel_multilabelUncertainty(self):
        sampler = MultilabelUncertaintySampler()
        model = MockModel(True)
        probs = model.predict_proba(self.X)
        weights = sampler.get_weights(probs)
        self.assertEqual(6, len(weights))
        self.assertEqual(round(weights[3], 6), round(weights[-1], 6))
        self.assertEqual(0, np.argmax(weights))
        self.assertEqual(1, np.argmin(weights))

        batch = sampler.select_batch(self.X, 2, model)
        self.assertEqual(2, len(batch))
        self.assertIn(1, batch)
        self.assertIn(2, batch)

        batch = sampler.select_batch(self.X, len(self.X), model)
        self.assertEqual(len(self.X), len(batch))
        self.assertEqual(list(batch), list(range(len(self.X))))
        batch = sampler.select_batch(self.X, len(self.X) + 1, model)
        self.assertEqual(len(self.X), len(batch))
        self.assertEqual(list(batch), list(range(len(self.X))))

    def test_density(self):
        sampler = DensitySampler()
        weights = sampler.get_weights(self.X, "cos")
        self.assertEqual(6, len(weights))
        self.assertEqual(round(weights[-2], 6), round(weights[-1], 6))
        weights = sampler.get_weights(self.X, "euclid")
        self.assertEqual(6, len(weights))
        self.assertEqual(round(weights[-2], 6), round(weights[-1], 6))

        batch = sampler.select_batch(self.X, 2, "cos")
        self.assertEqual(2, len(batch))
        batch = sampler.select_batch(self.X, 2, "euclid")
        self.assertEqual(2, len(batch))

        batch = sampler.select_batch(self.X, len(self.X))
        self.assertEqual(len(self.X), len(batch))
        batch = sampler.select_batch(self.X, len(self.X) + 1)
        self.assertEqual(len(self.X), len(batch))

    def test_combined(self):
        sampler_combined = CombinedSampler()
        sampler_uncertainty = LeastConfidentSampler()
        sampler_density = DensitySampler()

        model = MockModel(False)
        probs = model.predict_proba(self.X)
        weights_uncertainty = sampler_uncertainty.get_weights(probs, False)
        weights_density = sampler_density.get_weights(self.X)

        weights_combined = sampler_combined.get_weights(
            [sampler_uncertainty, sampler_density],
            [10, 0.1],
            probs=probs,
            X_unlab=self.X,
            multilabel=False,
        )
        self.assertEqual(6, len(weights_combined))
        self.assertTrue(
            list_similarity(
                np.argsort(weights_uncertainty),
                np.argsort(weights_density),
                np.argsort(weights_combined),
            )
        )

        weights_combined = sampler_combined.get_weights(
            [sampler_uncertainty, sampler_density],
            [0.1, 10],
            probs=probs,
            X_unlab=self.X,
            multilabel=False,
        )
        self.assertEqual(6, len(weights_combined))
        self.assertTrue(
            list_similarity(
                np.argsort(weights_density),
                np.argsort(weights_uncertainty),
                np.argsort(weights_combined),
            )
        )

        batch_uncertainty = sampler_uncertainty.select_batch(self.X, 2, model, False)
        batch_density = sampler_density.select_batch(self.X, 2)

        batch_combined = sampler_combined.select_batch(
            self.X,
            [sampler_uncertainty, sampler_density],
            2,
            powers=[10, 0.1],
            model=model,
            multilabel=False,
        )
        self.assertEqual(2, len(batch_combined))
        self.assertEqual(list(batch_uncertainty), list(batch_combined))

        batch_combined = sampler_combined.select_batch(
            self.X,
            [sampler_uncertainty, sampler_density],
            2,
            powers=[0.1, 10],
            model=model,
            multilabel=False,
        )
        self.assertEqual(2, len(batch_combined))
        self.assertEqual(list(batch_density), list(batch_combined))

        batch = sampler_combined.select_batch(
            self.X,
            [sampler_uncertainty, sampler_density],
            len(self.X),
            model=model,
            multilabel=False,
        )
        self.assertEqual(len(self.X), len(batch))
        self.assertEqual(list(batch), list(range(len(self.X))))
        batch = sampler_combined.select_batch(
            self.X,
            [sampler_uncertainty, sampler_density],
            len(self.X) + 1,
            model=model,
            multilabel=False,
        )
        self.assertEqual(len(self.X), len(batch))
        self.assertEqual(list(batch), list(range(len(self.X))))

    def test_combined_entropy_density(self):
        sampler_combined_entropy_density = CombinedEntropyDensitySampler()
        sampler_uncertainty = LeastConfidentSampler()
        sampler_density = DensitySampler()

        model = MockModel(False)
        probs = model.predict_proba(self.X)
        weights_uncertainty = sampler_uncertainty.get_weights(probs, False)
        weights_density = sampler_density.get_weights(self.X)

        weights_combined = sampler_combined_entropy_density.get_weights(
            powers=[10, 0.1],
            probs=probs,
            X_unlab=self.X,
            multilabel=False,
        )
        self.assertEqual(6, len(weights_combined))
        self.assertTrue(
            list_similarity(
                np.argsort(weights_uncertainty),
                np.argsort(weights_density),
                np.argsort(weights_combined),
            )
        )

        weights_combined = sampler_combined_entropy_density.get_weights(
            powers=[0.1, 10],
            probs=probs,
            X_unlab=self.X,
            multilabel=False,
        )
        self.assertEqual(6, len(weights_combined))
        self.assertTrue(
            list_similarity(
                np.argsort(weights_density),
                np.argsort(weights_uncertainty),
                np.argsort(weights_combined),
            )
        )

        batch_uncertainty = sampler_uncertainty.select_batch(self.X, 2, model, False)
        batch_density = sampler_density.select_batch(self.X, 2)

        batch_combined = sampler_combined_entropy_density.select_batch(
            self.X,
            2,
            powers=[10, 0.1],
            model=model,
            multilabel=False,
        )
        self.assertEqual(2, len(batch_combined))
        self.assertEqual(list(batch_uncertainty), list(batch_combined))

        batch_combined = sampler_combined_entropy_density.select_batch(
            self.X,
            2,
            powers=[0.1, 10],
            model=model,
            multilabel=False,
        )
        self.assertEqual(2, len(batch_combined))
        self.assertEqual(list(batch_density), list(batch_combined))

        batch = sampler_combined_entropy_density.select_batch(
            self.X, len(self.X), model=model, multilabel=False
        )
        self.assertEqual(len(self.X), len(batch))
        self.assertEqual(list(batch), list(range(len(self.X))))
        batch = sampler_combined_entropy_density.select_batch(
            self.X, len(self.X) + 1, model=model, multilabel=False
        )
        self.assertEqual(len(self.X), len(batch))
        self.assertEqual(list(batch), list(range(len(self.X))))
