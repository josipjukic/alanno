import csv
import re

from server.models import Project, Label
from text.preprocessing import Indexer

# CSV HEADERS
H_ID, H_TEXT, H_LABEL, H_ANNOTATOR = "document_id", "text", "label", "annotator"

# AGGREGATION KEYS
K_TXT, K_ANNOTS, K_HASID, K_HASLABEL = "text", "annotations", "has_id", "has_label"

# MESSAGES
MSG_READING_FILE = "Reading the file... (1/2)"
MSG_PROCESSING_DATA = "Processing data... (2/2)"
MSG_PROCESSING_JSON = "Processing data..."


def import_csv(project, lines, progress_recorder):
    """Imports a CSV dataset to a project.

    Converts lines from a CSV dataset into the appropriate Document subclass,
    saves them under the given Project, and links Annotations to them.

    Currently supported CSV headers are:
          HEADER        |   REQUIRED    |   DESCRIPTION
        - text          |   yes         |   document text
        - document_id   |   no          |   external ID of the document, used to group annotations
        - label         |   no          |   document label, discarded if doesn't exist in project
        - annotator     |   no          |   annotator username, discarded if doesn't exist in project
                                                                             or if label not present
    All unrecognized headers are ignored.

    Parameters
    ----------
    project             : Project
                          The project to which to add the Documents.
    lines               : list[str]
                          List of lines in the CSV file, including the header at index 0.
    progress_recorder   : ProgressRecorder
                          Progress recorder tracking the celery task's progress.

    Returns
    -------
    docs, label_stats   : list[Document], (int,int)
                          List of all created documents and a tuple of the number
                          of labeled Documents and the number of created annotations.
    """
    reader = csv.reader(lines)
    header = [text.strip().lower() for text in next(reader)]
    entries = list(reader)

    indices = [-1 if key not in header else header.index(key) for key in [H_ID, H_TEXT, H_LABEL, H_ANNOTATOR]]
    indices = [
        -1 if key not in header else header.index(key)
        for key in [H_ID, H_TEXT, H_LABEL, H_ANNOTATOR]
    ]
    if indices[1] == -1:
        raise ValueError("The 'text' header must be present.")
    if not project.is_type_of(Project.DOCUMENT_CLASSIFICATION):
        indices[2] = indices[3] = -1

    data = aggregate_csv(entries, indices, progress_recorder)
    docs, label_stats = build_documents(data, project, indices[0] >= 0, progress_recorder)
    if not project.is_type_of(Project.DOCUMENT_CLASSIFICATION):
        label_stats = None
    return docs, label_stats


def aggregate_csv(entries, indices, progress_recorder):
    """Reads the CSV lines and prepares the data for saving.

    Reorganizes the dataset into a format better suited for
    saving. The data is organized into a dictionary with the
    following format:
        {
            DOCUMENT_ID: {
                "text": text,
                "annotations": {
                    ANNOTATOR: list[LABEL]
                }
            },
        }

    If the 'document_id' header is not present, the index of the
    CSV entry is temporarily used as the DOCUMENT_ID key.

    Parameters
    ----------
    entries             : list[list[str]]
                          List of lists of comma-separated values for each CSV row
    indices             : list[int]
                          List of indices of properties, in order: idx_document_id, idx_text, idx_label, idx_annotator
    progress_recorder   : ProgressRecorder
                          Progress recorder tracking the celery task's progress.

    Returns
    -------
    data                : dict[str: dict[str: str || dict[str: list[str]]]]
                          Dictionary with reorganized CSV data, grouped by document id.
    """
    total_size = len(entries)
    idx_id, idx_txt, idx_lbl, idx_ant = indices

    data = {}
    for i, entry in enumerate(entries):
        if idx_id >= 0 and entry[idx_id] == "":
            raise ValueError(
                "If the 'document_id' header exists, all entries must have a defined ID."
            )

        doc_key = str(i) if idx_id < 0 else entry[idx_id]
        if doc_key not in data:
            if entry[idx_txt].strip() == "":
                raise ValueError(
                    "All new document rows must have a non-blank 'text' field."
                )
            data[doc_key] = {K_TXT: entry[idx_txt].strip(), K_ANNOTS: {}}

        if idx_lbl >= 0:
            label = entry[idx_lbl].strip()
            if label != "":
                annotator = None
                if idx_ant >= 0 and entry[idx_ant].strip() != "":
                    annotator = entry[idx_ant].strip()
                if annotator not in data[doc_key][K_ANNOTS]:
                    data[doc_key][K_ANNOTS][annotator] = set()
                data[doc_key][K_ANNOTS][annotator].add(label)

        progress_recorder.set_progress(i + 1, total_size, description=MSG_READING_FILE)
    return data


def build_documents(data, project, has_doc_id, progress_recorder):
    """Builds and saves Documents based on the given data object.

    Initializes Documents from the given data object, attaching
    them to the given Project, and attaching all valid annotations
    to the Document.

    For a specification of the data object, see method aggregate_csv.

    Parameters
    ----------
    data                : dict[str: dict[str: str || dict[str: list[str]]]]
                          Aggregated CSV data
    project             : Project
                          Project to which to add the documents.
    has_doc_id          : bool
                          Flag determining whether the data dictionary keys are to be used as external document IDs
    progress_recorder   : ProgressRecorder
                          Progress recorder tracking the celery task's progress.

    Returns
    -------
    docs, label_stats   : list[Document], (int,int)
                          List of all created documents and a tuple of the number
                          of labeled Documents and the number of created annotations.
    """
    total_size = len(data)
    labeled_docs, num_labels = 0, 0

    doc_class = project.get_document_class()
    anno_class = project.get_annotation_class()

    docs = []
    for i, (doc_key, info) in enumerate(data.items()):
        try:
            cleaned = clean_text(info[K_TXT])
        except ValueError:
            continue
        doc = doc_class(
            text=cleaned,
            project=project,
            document_id=(doc_key if has_doc_id else None),
            raw_html=info[K_TXT],
        )
        doc.save()
        docs.append(doc)

        is_labeled = 0
        for annotator, labels in info[K_ANNOTS].items():
            if None in info[K_ANNOTS] and annotator is not None:
                continue
            if len(labels) > 1 and not project.multilabel:
                continue
            annotator_obj = None
            if annotator is not None:
                annotator_obj = project.annotators.filter(username=annotator).first()
                if not annotator_obj:
                    continue
            for label in labels:
                label_obj = (
                    Label.objects.filter(project__id__exact=project.id)
                    .filter(text=label)
                    .first()
                )
                if not label_obj:
                    continue
                annotation = anno_class(document=doc, label=label_obj)
                annotation.user = annotator_obj
                if annotator_obj is not None:
                    doc.completed_by.add(annotator_obj)
                doc.is_warm_start = True
                doc.save()
                annotation.save()
                num_labels += 1
                is_labeled = 1

        labeled_docs += is_labeled
        progress_recorder.set_progress(
            i + 1, total_size, description=MSG_PROCESSING_DATA
        )

    index_documents(docs, project.language)

    return docs, (labeled_docs, num_labels)


def index_documents(docs, lang):
    indexer = Indexer(lang)
    doc_list = [doc.text for doc in docs]
    indexing_list = [" ".join(index) for index in indexer.index(doc_list)]
    for i in range(len(docs)):
        docs[i].indexing = indexing_list[i]
        docs[i].save()


def clean_text(text):
    pattern = re.compile(r"<script>")
    result = pattern.search(text)
    if result is not None:
        raise ValueError("Script tag not allowed in document!")

    pattern = re.compile(r"<.*?>")
    result = pattern.search(text)
    raw_text = text
    if result is not None:
        raw_text = pattern.sub("", text)
    return raw_text
