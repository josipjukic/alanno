import random
from collections import namedtuple

import torch
from torch.quasirandom import SobolEngine


def round_robin_iter(iterable, n, weights=None):
    if not weights:
        weights = {anno.username: 1 for anno in iterable}
    distributed_docs = {anno.username: 1 for anno in iterable}
    anno_dict = {anno.username: anno for anno in iterable}
    DistributionData = namedtuple("DistributionData", ["username", "value"])

    while True:
        data = [
            DistributionData(
                anno.username, distributed_docs[anno.username] / weights[anno.username]
            )
            for anno in iterable
        ]
        random.shuffle(data)
        data.sort(key=lambda x: x.value)
        for anno_data in data[:n]:
            distributed_docs[anno_data.username] += 1
        yield tuple([anno_dict[anno_data.username] for anno_data in data[:n]])


def round_robin(annotators, docs, anno_per_dp, **kwargs):
    for doc, anno_tup in zip(docs, round_robin_iter(annotators, anno_per_dp, **kwargs)):
        for annotator in anno_tup:
            annotator.selected_docs.add(doc)


def quasi_monte_carlo(annotators, docs, anno_per_dp, **kwargs):
    num_docs = docs.count()
    N = len(annotators)
    K = anno_per_dp
    N_keep = N
    N_keep_prev = N
    anno_indices = list(range(N))

    dpa = num_docs // N * K
    mod = num_docs % N
    docs_per_anno = torch.tensor(dpa).repeat(N)
    for i in range(mod):
        docs_per_anno[i] += 1

    docs_counter = torch.zeros(N)

    scramble = False
    sobol = SobolEngine(dimension=N_keep, scramble=scramble)

    total_runs = 1
    tups = []
    while True:
        # Draw vectors generated from the Sobol sequence
        sobol_vector = sobol.draw().squeeze()
        # Create binary vector by rounding
        sobol_binary = sobol_vector.round()
        # Check if number of 1 in the vector is the same as number of annotators per document
        if sobol_binary.sum() == K:
            # Make annotator tuples at indices where vector has a 1
            indices, *_ = torch.where(sobol_binary == 1)
            indices = indices.tolist()
            mapped_indices = [anno_indices[i] for i in indices]
            tups.append(tuple(mapped_indices))

            # Check whether a sufficient number of documents have been distributed to the annotator
            for i in reversed(indices):
                j = anno_indices[i]
                docs_counter[j] += 1
                if docs_counter[j] >= docs_per_anno[j]:
                    del anno_indices[i]
                    N_keep -= 1

            # If number of available annotators has changed, start
            # drawing vectors with the dimension of newly available annotators
            if N_keep != N_keep_prev:
                # If there are less or equal available annotators than required per document,
                # distribute all of the documents to those annotators
                if N_keep <= K:
                    leftover = num_docs - len(tups)
                    for i in range(leftover):
                        _, torch_tup = torch.topk(docs_counter, largest=False, k=K)
                        docs_counter[torch_tup] += 1
                        tup = tuple(torch_tup.tolist())
                        tups.append(tup)
                    break

                sobol = SobolEngine(dimension=N_keep, scramble=scramble)
                N_keep_prev = N_keep

            if len(tups) >= num_docs:
                break
        total_runs += 1

    for doc, anno_tup in zip(docs, tups):
        for anno_idx in anno_tup:
            annotator = annotators[anno_idx]
            annotator.selected_docs.add(doc)
