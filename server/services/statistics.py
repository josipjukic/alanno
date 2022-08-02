from server.models import Project, ProjectStatistics, UserLog
from collections import Counter
from itertools import chain
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
from torch import nn
from pyro.nn import PyroSample
import torch
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import SVI, Trace_ELBO
from pyro.infer import Predictive
import numpy as np


def calculate_project_stats(project, progress_recorder):
    """Calculates and saves all statistics for the given project.

    This method is in charge of calculating five groups of statistical measures:
        - General annotation progress
        - Label frequencies
        - Per-user annotation progress
        - Inter-Annotator Agreement statistics
        - AL train/test metrics (if applicable)

    All measures are calculated for each annotation round of the project,
    and once again for the whole project's lifecycle.

    Once the calculation is finished, the statistics are saved into the database
    and ready for future retrieval.

    Parameters
    ----------
    project                 : Project
                              The project for which to calculate statistics
    progress_recorder       : ProgressRecorder
                              Celery progress recorder tracking the task's progress
    """
    labels = [label.text for label in project.labels.all()]
    users = project.annotators.order_by("username").all()

    rounds = list(project.rounds.all()) + [None]
    progress_size = len(rounds)

    progress_recorder.set_progress(
        1,
        progress_size + 1,
        description=f"Initializing project statistics...",
    )

    # Duration statistics
    # ==================================================
    dist_docs = project.get_distributed_documents(None)
    user_doc_timings = calculate_timings(users, project, dist_docs)
    # ==================================================

    # AL train/test metrics.
    # ==================================================
    perf = project.get_model_performance()
    if perf:
        train, test = perf["train"], perf["test"]
    else:
        train, test = None, None
    # ==================================================

    stats = {
        "train": train,
        "test": test,
        "mean_pred": None,
        "lb_pred": None,
        "ub_pred": None,
        "x_pred": None,
    }
    if train is not None and len(test["count"]) > 2:
        # Step is an average batch size rounded to the nearest integer
        step = sum(
            [test["count"][0]]
            + [
                test["count"][i + 1] - test["count"][i]
                for i in range(len(train["count"]) - 1)
            ]
        ) / len(test["count"])
        step = int(np.around(step))

        mean_pred, lb, ub, x_pred = predict_score(test["metric"], test["count"], step)

        # Aline prediction's confidence interval with bootstrap's
        lb[0] = test["lb"][-1]
        ub[0] = test["ub"][-1]

        stats["mean_pred"] = mean_pred
        stats["lb_pred"] = lb
        stats["ub_pred"] = ub
        stats["x_pred"] = x_pred

    # Get all available rounds
    stats["rounds"] = sorted([r.number for r in rounds if r is not None])

    # Calculate stats per round and for all
    for i, round in enumerate(rounds):
        docs = project.get_distributed_documents(round)
        total_progress = get_progress(project, docs)  # Distribution counts.
        label_counts = get_label_counts(labels, docs)  # Label counts.
        gl_label_counts = None
        if round is None:
            docs_gl = project.documents.filter(gl_annotated=True)
            if docs_gl:
                gl_label_counts = get_label_counts(
                    labels, docs_gl, True
                )  # Guided learning label counts.
        user_lists, user_dicts = get_user_stats(
            users, project, docs
        )  # Document statistics per user.
        user_times = get_user_times(
            users, docs, user_doc_timings
        )  # Timing statistics per user
        iaa = get_iaa_stats(project, users, docs)  # IAA statistics

        round_number = "All" if round is None else round.number
        stats[round_number] = {
            "label": {
                "labels": labels,
                "data": label_counts,
                "gl_data": gl_label_counts,
            },
            "annotators": {**user_lists, "all": user_dicts},
            "progress": total_progress,
            "iaa": iaa,
            "times": user_times,
        }
        progress_recorder.set_progress(
            i + 2,
            progress_size + 1,
            description=f"Calculating project statistics... ({i+1}/{progress_size})",
        )
    stats["finished"] = True
    data = ProjectStatistics(stats=stats, project=project)
    data.save()


def get_progress(project, distributed_docs):
    """Calculates annotation progress on the given subset of distributed docs.

    The distributed_docs parameter includes all documents which were distributed
    in a certain round. This method calculates the number of distributed and
    queued documents, as well as the total number of documents in the project's
    dataset.

    Parameters
    ----------
    project             : Project
                          The project for which to calculate progress.
    distributed_docs    : list[Document]
                          Documents that were distributed in a certain round

    Returns
    -------
                        : dict
                        Dictionary with all progress data.
    """
    total_count = project.documents.filter(is_warm_start=False).count()
    distributed_count = len(distributed_docs)
    queued_count = total_count - distributed_count
    distributed_percentage = (
        int(distributed_count / total_count * 100) if total_count > 0 else 0
    )
    return {
        "distributed": distributed_count,
        "queued": queued_count,
        "percentage": distributed_percentage,
        "total": total_count,
    }


def get_label_counts(labels, distributed_docs, gl=False):
    """Calculates label frequency data.

    Calculates the number of appearances of each label in all annotations
    of all the documents distributed in a certain round.

    Parameters
    ----------
    labels              : list[str]
                          All labels registered for a Project
    distributed_docs    : list[Document]
                          All documents distributed in a certain round.

    Returns
    -------
                        : list[int]
                          List of counts for each label, in order as defined in the Project

    """
    annotations = lambda doc: doc.get_gl_annotations() if gl else doc.get_annotations()
    labels_per_doc = [
        [a.label.text for a in annotations(doc)] for doc in distributed_docs
    ]
    label_count_dict = Counter(chain(*labels_per_doc))
    return [
        0 if name not in label_count_dict.keys() else label_count_dict[name]
        for name in labels
    ]


def get_user_stats(users, project, docs):
    """Calculates annotation progress independently for each user.

    Calculates the total number of documents assigned to each user,
    as well as how many of them are completed and how many are still
    in progress (active).

    Parameters
    ----------
    users           : list[User]
                      List of User objects, representing annotators on the given Project
    project         : Project
                      The project for which to calculate user statistics
    docs            : list[Document]
                      List of documents distributed in a given round

    Returns
    -------
                    : list[dict], dict[list]
                      User statistics in two formats.
    """
    doc_id_set = set([d.id for d in docs])

    completed_counts = [
        len(
            [
                doc
                for doc in user.completed_docs.filter(project=project)
                if doc.id in doc_id_set
            ]
        )
        for user in users
    ]
    active_counts = [
        len(
            [
                doc
                for doc in user.selected_docs.filter(project=project).exclude(
                    completed_by=user
                )
                if doc.id in doc_id_set
            ]
        )
        for user in users
    ]
    total_counts = [i + j for i, j in zip(active_counts, completed_counts)]
    percentages = [
        int(i / j * 100) if j > 0 else 0 for i, j in zip(completed_counts, total_counts)
    ]

    user_lists = {
        "username": [user.username for user in users],
        "active": active_counts,
        "completed": completed_counts,
        "total": total_counts,
        "percentage": percentages,
    }
    user_dicts = [dict(zip(user_lists, t)) for t in zip(*user_lists.values())]
    return user_lists, user_dicts


def get_user_times(users, docs, timings):
    result = {u.username: {"total": 0, "average": 0} for u in users}
    for user in users:
        doc_times = timings[user.id]
        count = 0
        for doc in docs:
            time = doc_times[doc.id]
            if time is None:
                continue
            count += 1
            result[user.username]["total"] += time
        if count > 0:
            result[user.username]["average"] = result[user.username]["total"] / count
    return result


def get_iaa_stats(project, users, docs):
    """Calculates Inter-Annotator Agreement statistics.

    Most of the logic is delegated to the following methods:
        - Project :: get_iaa_data(list[User], list[Document])
        - Project :: calculate_iaa(dict, list[str])

    Parameters
    ----------
    project         : Project
                      The Project for which to calculate IAA statistics
    users           : list[User]
                      List of annotators included in IAA stats calculation
    docs            : list[Document]
                      List of documents distributed in a certain round

    Returns
    -------
                    : dict
                      IAA statistics
    """
    iaa_data = project.get_iaa_data(users, docs)
    if len(users) <= 1 or not iaa_data:
        return None

    max_annotations = (
        iaa_data["max_annotations"] if "max_annotations" in iaa_data else None
    )
    usernames = [user.username for user in users]
    iaa_metrics = project.calculate_iaa(iaa_data, usernames)
    if project.is_type_of(Project.KEX):
        lenient_iaa, exact_iaa = iaa_metrics
        l_pairs_html, l_avg_html, series = iaa_html(lenient_iaa, usernames)
        e_pairs_html, e_avg_html, series = iaa_html(exact_iaa, usernames)
        return {
            "lenient_pairs": l_pairs_html,
            "lenient_avg": l_avg_html,
            "exact_pairs": e_pairs_html,
            "exact_avg": e_avg_html,
        }
    else:
        series, avgs, iaa_joint = iaa_html(iaa_metrics, usernames)
        return {
            "avg": avgs,
            "pairwise": series,
            "max_annotations": max_annotations,
            "joint": iaa_joint["score"],
            "joint_used_docs": iaa_joint["used_docs"],
        }


def iaa_html(iaa_metrics, usernames):
    series = []
    avgs = []
    iaa_pairwise, iaa_joint = iaa_metrics
    for i, user1 in enumerate(usernames):
        values = []
        total = 0
        cnt = 0
        for j, user2 in enumerate(usernames):
            pair = min(i, j), max(i, j)
            val = None
            if i == j:
                val = 1
            elif pair in iaa_pairwise:
                val = iaa_pairwise[pair]
                total += val
                cnt += 1
            item = {"x": user2, "y": val}
            values.append(item)
        series.append({"name": user1, "data": values})
        if cnt > 0:
            avgs.append(round(total / cnt, 2))
        else:
            avgs.append(None)
    return series, [{"data": avgs}], iaa_joint


def calculate_timings(users, project, docs):
    result = {u.id: {} for u in users}
    req_types = [UserLog.DOC_OPEN, UserLog.DOC_CLOSE]
    for user in users:
        for doc in docs:
            doc_logs = UserLog.objects.filter(
                user=user, project=project, document=doc, type__in=req_types
            ).order_by("timestamp")
            result[user.id][doc.id] = time_for_document(doc_logs)
    return result


def time_for_document(logs):
    if len(logs) == 0:
        return None

    total = 0.0
    i = 0
    while True:  # Repeat until logs exhausted
        while i < len(logs) and logs[i].type != UserLog.DOC_OPEN:
            # Find first DOC_OPEN log
            i += 1
        if i >= len(logs):
            # No DOC_OPEN log
            break

        open_log = logs[i]
        i += 1

        while i < len(logs) and logs[i].type != UserLog.DOC_CLOSE:
            # Find first DOC_CLOSE log
            open_log = logs[i]  # Update open_log to last before DOC_CLOSE
            i += 1
        if i >= len(logs):
            # No matching DOC_CLOSE log
            break

        close_log = logs[i]
        i += 1

        timedelta = close_log.timestamp - open_log.timestamp
        total += timedelta.total_seconds()

    return total if total > 0 else None


def find_first_log_after_ts(close_logs, ts):
    post_ts = [l for l in close_logs if l.timestamp > ts]
    if len(post_ts) > 0:
        return post_ts[0]
    else:
        return None


def predict_score(scores, indices, step):
    """
    Predicts scores for next five batches.
    """
    X_test = [indices[-1]] + [indices[-1] + (i + 1) * step for i in range(5)]
    scores = torch.tensor(scores).float()
    indices = torch.log(torch.tensor(indices)).reshape(-1, 1).float()
    X_test = torch.log(torch.tensor(X_test)).reshape(-1, 1).float()

    model = BayesianRegression()
    guide = AutoDiagonalNormal(model)
    adam = pyro.optim.Adam({"lr": 0.02, "weight_decay": 0.001})
    svi = SVI(model, guide, adam, loss=Trace_ELBO())

    # Train loop
    pyro.clear_param_store()
    for j in range(3000):
        svi.step(indices, scores)

    # Predict new scores
    predictive = Predictive(model, guide=guide, num_samples=800, return_sites=["obs"])
    samples = predictive(X_test)
    mean = torch.mean(samples["obs"], 0)
    std = torch.std(samples["obs"], 0)
    lb = mean - std
    ub = mean + std

    # Move prediction to the last test sample and clip to range [0,1]
    addon = scores[-1] - mean[0]
    mean = torch.clip(mean + addon, 0, 1).tolist()
    lb = torch.clip(lb + addon, 0, 1).tolist()
    ub = torch.clip(ub + addon, 0, 1).tolist()

    return mean, lb, ub, torch.exp(X_test).reshape(-1).int().tolist()


class BayesianRegression(PyroModule):
    def __init__(self, in_features=1, out_features=1):
        super().__init__()
        self.linear = PyroModule[nn.Linear](in_features, out_features)
        self.linear.weight = PyroSample(
            dist.Normal(0.05, 0.05).expand([out_features, in_features]).to_event(2)
        )
        self.linear.bias = PyroSample(
            dist.Normal(0.5, 0.5).expand([out_features]).to_event(1)
        )

    def forward(self, x, y=None):
        sigma = pyro.sample("sigma", dist.Uniform(0.0, 5.0))
        mean = self.linear(x).squeeze(-1)
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mean, sigma), obs=y)
        return mean
