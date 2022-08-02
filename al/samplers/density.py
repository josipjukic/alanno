from .abstract_sampler import AbstractSampler
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances

distance_types = {"cos": cosine_distances, "euclid": euclidean_distances}


class DensitySampler(AbstractSampler):
    name = "density"

    def select_batch(self, X_unlab, batch_size, distance="cos", **kwargs):
        if len(X_unlab) <= batch_size:
            return np.arange(len(X_unlab))

        avg_distance = self.get_weights(X_unlab, distance)

        # Retrieve `batch_size` instances with lowest average distance.
        top_n = np.argpartition(avg_distance, batch_size)[:batch_size]
        return top_n

    def get_weights(self, X_unlab, distance="cos", **kwargs):
        if isinstance(distance, str):
            try:
                distance = distance_types[distance]
            except KeyError:
                raise ValueError("Distance %s not supported" % (distance))

        # Working with distance instead of similarity, because argmin is used to create batch
        distance_matrix = distance(X_unlab)
        avg_distance = np.sum(distance_matrix, axis=1) / (len(X_unlab) - 1)

        scaler = MinMaxScaler()
        avg_distance = scaler.fit_transform(avg_distance.reshape(-1, 1)).reshape(-1)
        return avg_distance
