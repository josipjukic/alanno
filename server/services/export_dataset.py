from server.models import ExportData
import csv
import json
import numpy as np
from io import StringIO


# MESSAGES
MSG_FORMATTING_DOCS = "Formatting documents..."
MSG_AGGREGATING_LABELS = "Aggregating labels..."


def export_csv(project, docs, user, aggregation, unlabeled, progress_recorder):
    # Iterating over docs + 1 for label aggregation
    progress_size = len(docs) + 1

    text = StringIO()
    writer = csv.writer(text)
    csv_header = project.get_csv_header(aggregation)
    if csv_header:
        writer.writerow(csv_header)
    doc_csv_representation = representation_csv(
        docs, aggregation, unlabeled, progress_recorder, progress_size
    )
    if aggregation:
        doc_csv_representation = label_aggregation_csv(
            project, docs, doc_csv_representation, progress_recorder, progress_size
        )
    else:
        doc_csv_representation = unpack_entries_csv(
            doc_csv_representation, progress_recorder, progress_size
        )
    writer.writerows(doc_csv_representation)
    data = ExportData(
        text=text.getvalue(),
        user=user,
        project=project,
        format="csv",
        is_aggregated=aggregation,
        is_unlabeled=unlabeled,
    )
    data.save()


def representation_csv(docs, aggregation, unlabeled, progress_recorder, progress_size):
    doc_csv_representation = []
    for i, d in enumerate(docs):
        rep = d.to_csv(aggregation=aggregation, unlabeled=unlabeled)
        if not aggregation:
            rep = sorted(rep)
        doc_csv_representation.append(rep)
        progress_recorder.set_progress(
            i + 1, progress_size, description=MSG_FORMATTING_DOCS
        )
    return doc_csv_representation


def label_aggregation_csv(
    project, docs, doc_csv_representation, progress_recorder, progress_size
):
    """
    Inserts column of aggregated labels.
    """
    progress_recorder.set_progress(
        progress_size, progress_size, description=MSG_AGGREGATING_LABELS
    )
    aggregated_labels_raw = project.get_aggregated_labels(docs)
    aggregated_labels = [
        "" if lab is None else format_label(lab) for lab in aggregated_labels_raw
    ]
    # Index before which aggregated labels are inserted, depends on csv_header
    insert_index = -6 if project.multilabel else -5
    for i, rep in enumerate(doc_csv_representation):
        rep.insert(insert_index, aggregated_labels[i])
    return doc_csv_representation


def format_label(lab):
    if isinstance(lab, str):
        return lab
    else:
        return ";".join(lab)


def unpack_entries_csv(doc_csv_representation, progress_recorder, progress_size):
    """
    If data is not aggregated doc_csv_representation
    is a list, of list of (doc_id, doc_text, label, annotator), for each document,
    but needs to be converted to list of (doc_id, doc_text, label, annotator)
    in order to make the csv file.
    """
    temp = []
    progress_recorder.set_progress(
        progress_size, progress_size, description=MSG_FORMATTING_DOCS
    )
    for doc in doc_csv_representation:
        temp.extend(doc)
    return temp


def export_json(project, docs, user, aggregation, unlabeled, progress_recorder):
    # Iterating over docs + 1 for label aggregation
    progress_size = len(docs) + 1

    json_list = representation_json(
        docs, aggregation, unlabeled, progress_recorder, progress_size
    )
    if aggregation:
        json_list = label_aggregation_json(
            project, docs, json_list, progress_recorder, progress_size
        )
    data = ExportData(
        text=json.dumps(json_list, ensure_ascii=False),
        user=user,
        project=project,
        format="json",
        is_aggregated=aggregation,
        is_unlabeled=unlabeled,
    )
    data.save()


def label_aggregation_json(project, docs, json_list, progress_recorder, progress_size):
    """
    Fill aggregated_label of json with aggregated labels.
    """
    progress_recorder.set_progress(
        progress_size, progress_size, description="Aggregating labels..."
    )
    aggregated_labels = np.array(project.get_aggregated_labels(docs))
    aggregated_labels[aggregated_labels == None] = ""
    for i in range(len(json_list)):
        label = (
            aggregated_labels[i]
            if isinstance(aggregated_labels[i], str)
            else list(aggregated_labels[i])
        )
        json_list[i]["aggregated_label"] = label
    return json_list


def representation_json(docs, aggregation, unlabeled, progress_recorder, progress_size):
    json_list = []
    for i, d in enumerate(docs):
        json_list.extend(d.to_json(aggregation=aggregation, unlabeled=unlabeled))
        progress_recorder.set_progress(
            i + 1, progress_size, description="Formatting documents..."
        )
    return json_list
