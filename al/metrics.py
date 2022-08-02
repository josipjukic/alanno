from functools import partial
import random

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import numpy as np

metric_types = {
    "f1_macro": partial(f1_score, average="macro"),
    "f1_micro": partial(f1_score, average="micro"),
    "f1_samples": partial(f1_score, average="samples"),
    "accuracy": accuracy_score,
    "precision_macro": partial(precision_score, average="macro"),
    "precision_micro": partial(precision_score, average="micro"),
    "precision_samples": partial(precision_score, average="samples"),
    "recall_macro": partial(recall_score, average="macro"),
    "recall_micro": partial(recall_score, average="micro"),
    "recall_samples": partial(recall_score, average="samples"),
}


def calculate_metrics(model, X_train, y_train, X_test, y_test, measure="f1_micro"):
    """
    Calculates train/test performances on the currently annotated datapoints.
    """
    if isinstance(measure, str):
        try:
            measure = metric_types[measure]
        except KeyError:
            raise ValueError("Measure %s not supported" % (measure))

    metric = Metric(measure)
    m_train = metric.eval_metric(y_train, model.predict(X_train))

    m_test = None
    conf = None
    if X_test.shape[0] > 0:
        m_test = metric.eval_metric(y_test, model.predict(X_test))
        alpha = 0.01
        bootstrap, lower, upper = bootstrap_confidence_interval(
            model, X_test, y_test, metric, alpha=alpha
        )
        conf = {}
        conf["lower"] = lower
        conf["upper"] = upper
        conf["bootstrap"] = bootstrap
        conf["alpha"] = alpha

    result = dict(
        train=m_train,
        test=m_test,
        labeled_train=X_train.shape[0],
        labeled_test=X_test.shape[0],
        conf=conf,
    )
    return result


class Metric:
    def __init__(self, measure=metric_types["f1_micro"]):
        self.measure = measure

    def eval_metric(self, y_true, y_pred):
        return self.measure(y_true, y_pred)


def bootstrap_confidence_interval(model, X, y, stat_metric, N=100, alpha=0.01):
    statistics = []
    k_size = int(y.shape[0])
    indices = range(k_size)
    for _ in range(N):
        random_indices = random.choices(population=indices, k=k_size)
        X_sample = X[random_indices]
        y_sample = y[random_indices]
        if X_sample.shape[0] == 1:
            X_sample = X_sample.reshape(1, -1)
        y_pred = model.predict(X_sample)
        measure = stat_metric.eval_metric(y_sample, y_pred)
        statistics.append(measure)

    mean_score = np.mean(statistics)
    std = np.std(statistics)
    lower = mean_score - std
    upper = mean_score + std

    return mean_score, lower, upper


def exact_match(p, ps):
    return p in ps


def lenient_match(p, ps):
    for pi in ps:
        if p in pi or pi in p:
            return True

    return False


def f1_score_over_phrases(gold_p, pred_p, match_fn):
    # Remove duplicates.
    print("GOLD", gold_p)
    print("PRED", pred_p)
    gold_p = set(gold_p)
    pred_p = set(pred_p)

    total_p = gold_p | pred_p

    gold = list()
    pred = list()

    for p in total_p:
        if match_fn(p, gold_p):
            gold.append(1)
        else:
            gold.append(0)

        if match_fn(p, pred_p):
            pred.append(1)
        else:
            pred.append(0)

    return f1_score(gold, pred, average="binary")


# class MetricsModel:
#     def __init__(self, total, per_class):
#         self.total = total
#         self.per_class = per_class
#
#
# class MultilabelMetrics:
#     def __init__(self, binarizer=None):
#         if binarizer:
#             self.binarizer = binarizer
#             self.n_classes = len(binarizer.classes_)
#
#         self.single_metrics = dict(
#             [
#                 ("f1_macro", f1_macro),
#                 ("f1_micro", f1_micro),
#             ]
#         )
#
#         self.category_metrics = dict(
#             [
#                 ("f1_each_class", f1_each_class),
#                 ("confusion_matrix", multilabel_confusion_matrix),
#             ]
#         )
#
#     def eval_metrics(self, ys_true, hs):
#         return MetricsModel(
#             dict(
#                 [
#                     (name, metric(ys_true, hs))
#                     for name, metric in self.single_metrics.items()
#                 ]
#             ),
#             dict(
#                 [
#                     (name, metric(ys_true, hs))
#                     for name, metric in self.category_metrics.items()
#                 ]
#             ),
#         )
#
#     def eval_single_metric(self, ys_true, hs):
#         return self.single_metrics["f1_macro"](ys_true, hs)
#
#     def eval_per_category(self, ys_true, hs):
#         if not self.binarizer:
#             raise Exception("Cannot call this method without a binarizer")
#         iteration = enumerate(self.binarizer.classes_)
#         for i, category in iteration:
#             h_category = hs[:, i]
#             y_category = ys_true[:, i]
#             evaluated_metrics = [
#                 (name, metric(y_category, h_category))
#                 for name, metric in self.category_metrics.items()
#             ]
#             yield category, evaluated_metrics
