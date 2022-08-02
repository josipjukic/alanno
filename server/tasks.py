from app.celery import app
from celery_progress.backend import ProgressRecorder
from functools import partial

from django.contrib.auth.models import User
from django.core.cache import cache
from django.core.mail import send_mass_mail
from django.db.models import Max
from django.shortcuts import get_object_or_404

from collections import Counter
from itertools import chain
from random import shuffle, sample
from smtplib import SMTPException
from datetime import datetime

from annotation_base.doc_distribution import round_robin, quasi_monte_carlo
from .models import Project, Document, ProjectStatistics, Round, UserLog

from server.services.import_dataset import import_csv
from server.services.statistics import calculate_project_stats

from server.services.export_dataset import export_csv, export_json

@app.task(ignore_result=True)
def delete_documents_task(doc_ids):
    Document.objects_with_deleted.filter(pk__in=doc_ids).delete()

@app.task(ignore_result=True)
def delete_project_task(project_id):
    Project.objects_with_deleted.get(pk=project_id).delete()

@app.task(ignore_result=True)
def edit_annotators_api_task(project_id, annotator_ids):
    """
    Update project annotators.

    Parameters
    ----------
    project_id: int
        ID of the project
    annotator_ids: list[int]
        IDs of annotators working on the project
    """
    project = get_object_or_404(Project, pk=project_id)
    new_annotators = User.objects.filter(id__in=annotator_ids)
    project.annotators.set(new_annotators)
    project.save(update_fields=["updated_at"])


@app.task(ignore_result=True)
def guidelines_api_task(project_id, guidelines):
    """
    Update project guidelines.

    Parameters
    ----------
    project_id: int
        ID of the project
    guidelines: str
        new project guidelines
    """
    project = get_object_or_404(Project, pk=project_id)
    project.guidelines = guidelines
    project.save(update_fields=["guidelines", "updated_at"])


@app.task(ignore_result=True)
def change_al_method_api_task(project_id, name):
    project = get_object_or_404(Project, pk=project_id)
    project.al_method = name
    project.save(update_fields=["al_method", "updated_at"])


@app.task(ignore_result=True)
def change_model_name_api_task(project_id, name):
    project = get_object_or_404(Project, pk=project_id)
    project.model_name = name
    project.save(update_fields=["model_name", "updated_at"])


@app.task(ignore_result=True)
def change_token_type_api_task(project_id, name):
    project = get_object_or_404(Project, pk=project_id)
    project.token_type = name
    project.save(update_fields=["token_type", "updated_at"])


@app.task(ignore_result=True)
def change_vectorizer_name_api_task(project_id, name):
    project = get_object_or_404(Project, pk=project_id)
    project.vectorizer_name = name
    project.save(update_fields=["vectorizer_name", "updated_at"])


@app.task(ignore_result=True)
def change_adjustable_vocab_api_task(project_id, value):
    project = get_object_or_404(Project, pk=project_id)
    project.adjustable_vocab = value
    project.save(update_fields=["adjustable_vocab", "updated_at"])


@app.task(ignore_result=True)
def change_vocab_max_size_api_task(project_id, value):
    project = get_object_or_404(Project, pk=project_id)
    project.vocab_max_size = value
    project.save(update_fields=["vocab_max_size", "updated_at"])


@app.task(ignore_result=True)
def change_vocab_min_freq_api_task(project_id, value):
    project = get_object_or_404(Project, pk=project_id)
    project.vocab_min_freq = value
    project.save(update_fields=["vocab_min_freq", "updated_at"])


@app.task(ignore_result=True)
def change_min_ngram_api_task(project_id, value):
    project = get_object_or_404(Project, pk=project_id)
    project.min_ngram = value
    project.save(update_fields=["min_ngram", "updated_at"])


@app.task(ignore_result=True)
def change_max_ngram_api_task(project_id, value):
    project = get_object_or_404(Project, pk=project_id)
    project.max_ngram = value
    project.save(update_fields=["max_ngram", "updated_at"])


@app.task
def completion_api_task(doc_id, completed, user_id, project_id):
    """
    Update users completed status for a given document.

    Parameters
    ----------
    doc_id: int
        ID of the document
    completed: bool
        whether the annotator has completed the document
    user_id: int
        ID of the annotator
    """
    ts = datetime.utcnow()
    document = Document.objects.get(pk=doc_id)
    user = User.objects.get(pk=user_id)
    project = Project.objects.get(pk=project_id)
    if completed:
        document.completed_by.add(user)
        log = UserLog(
            type=UserLog.DOC_LOCK,
            timestamp=ts,
            user=user,
            project=project,
            document=document,
            label=None,
            metadata=None,
        )
    else:
        document.completed_by.remove(user)
        log = UserLog(
            type=UserLog.DOC_UNLOCK,
            timestamp=ts,
            user=user,
            project=project,
            document=document,
            label=None,
            metadata=None,
        )
    document.save()
    log.save()


@app.task(bind=True)
def data_upload_task(self, project_id, file):
    """
    Extract documents from a CSV file and save them in the project.

    Parameters
    ----------
    project_id: int
        ID of the project
    file: list[str]
        file containing documents for import
    Returns
    -------
    str: information about successful import
    """
    project = get_object_or_404(Project, pk=project_id)
    progress_recorder = ProgressRecorder(self)
    docs, label_stats = import_csv(project, file, progress_recorder)

    msg = "Data upload finished.\n"
    msg += f"Uploaded {len(docs)} document{'s' if len(docs) != 1 else ''}"
    if not label_stats:
        msg += "."
    else:
        msg += f" of which {label_stats[0]} {'was' if label_stats[0] == 1 else 'were'}"
        msg += f" labeled with {label_stats[1]} label{'s' if len(docs) != 1 else ''}."

    # Randomly separate the data for the test set
    if project.al_mode:
        test_prop = 0.2
        docs_not_warm_start_ids = [doc.id for doc in docs if not doc.is_warm_start]
        count = len(docs_not_warm_start_ids)
        test_count = int(test_prop * count)
        test_ids = sample(docs_not_warm_start_ids, k=test_count)
        project.documents.filter(pk__in=test_ids).update(is_test=True)
        msg += f" Test pool size: {test_count}."

    project.save()

    return msg


@app.task(bind=True)
def get_export_csv_task(self, user_id, doc_ids, project_id, aggregation, unlabeled):
    """
    Create an instance of ExportData in CSV format that saves export data of a project.

    Parameters
    ----------
    user_id: int
        ID of the user to whom ExportData will be bound
    doc_ids: list[int]
        list of documents to be exported
    project_id: int
        ID of the project
    aggregation: bool
        exporting aggregated data
    unlabeled: bool
        exporting unlabeled data
    """
    user = get_object_or_404(User, id=user_id)
    project = get_object_or_404(Project, pk=project_id)
    docs = [Document.objects.get(pk=doc_id) for doc_id in doc_ids]
    progress_recorder = ProgressRecorder(self)

    export_csv(project, docs, user, aggregation, unlabeled, progress_recorder)


@app.task(bind=True)
def get_export_json_task(self, user_id, doc_ids, project_id, aggregation, unlabeled):
    """
    Create an instance of ExportData in JSON format that saves export data of a project.

    Parameters
    ----------
    user_id: int
        ID of the user to whom ExportData will be bound
    doc_ids: list[int]
        list of documents to be exported
    project_id: int
        ID of the project
    aggregation: bool
        exporting aggregated data
    unlabeled: bool
        exporting unlabeled data
    """
    user = get_object_or_404(User, id=user_id)
    project = get_object_or_404(Project, pk=project_id)
    docs = [Document.objects.get(pk=doc_id) for doc_id in doc_ids]
    progress_recorder = ProgressRecorder(self)

    export_json(project, docs, user, aggregation, unlabeled, progress_recorder)


@app.task(bind=True)
def calculate_project_stats_task(self, project_id):
    """
    Create an instance of ProjectStatistics that saves statistics data of a project.

    Parameters
    ----------
    project_id: int
        ID of the project
    """
    p = get_object_or_404(Project, pk=project_id)
    progress_recorder = ProgressRecorder(self)
    calculate_project_stats(p, progress_recorder)


@app.task(bind=True)
def create_batch_task(self, data, project_id, annotator_ids, anno_per_dp):
    """
    Create new batch from the form.

    Parameters
    ----------
    data: dict
        form for creating a batch
    project_id: int
        ID of the project
    annotator_ids: list[int]
        IDs of selected annotators
    anno_per_dp: int
        number of annotators per document

    Returns
    -------
    str: status of the task
    """
    progress_recorder = ProgressRecorder(self)

    # Document selection
    progress_recorder.set_progress(1, 3, description="Document selection...")
    project = get_object_or_404(Project, pk=project_id)
    selected_ids = project.select_batch(**data)

    if not selected_ids:
        return "error"

    # Document distribution
    progress_recorder.set_progress(2, 3, description="Document distribution...")
    shuffle(annotator_ids)

    annotators = []
    for anno_id in annotator_ids:
        annotator = User.objects.get(pk=int(anno_id))
        project.reset_completed_selection(annotator)
        annotators.append(annotator)

    docs = project.documents.filter(id__in=selected_ids)

    DISTRIBUTION_METHODS = {
        "round robin": round_robin,
        "quasi-monte carlo": quasi_monte_carlo,
        "weighted distribution": partial(round_robin, weights=data["weights"]),
    }
    distr_method = DISTRIBUTION_METHODS[data["method"].lower()]
    distr_method(annotators, docs, anno_per_dp)

    for annotator in annotators:
        annotator.save()

    # Create round information and save
    progress_recorder.set_progress(3, 3, description="Saving round information...")
    max_round = project.rounds.aggregate(Max("number"))["number__max"]
    if max_round is None:
        max_round = 0  # Index of the first round is zero
    else:
        max_round += 1
    round = Round(number=max_round, project=project)
    round.save()
    docs.update(round=round)

    project.save(update_fields=["updated_at"])

    return "success"


@app.task(bind=True, max_retries=5)
def send_batch_generation_email_task(self, mode, project_id, annotator_ids, user_id):
    """
    Sending email to project managers if batch is completed by all annotators.
    The method is chained after create_batch_task.
    If email fails to send task will be retried 5 times with exponential delay.

    Parameters
    ----------
    mode: str
        'success' - batch generation successful
        'error' - data is depleted
    project_id: int
        ID of the project
    annotator_ids: list[int]
        list of IDs of selected annotators
    user_id: int
        ID of the user who generated the batch
    """
    data = []
    project = get_object_or_404(Project, pk=project_id)
    if mode == "error":
        user = get_object_or_404(User, pk=user_id)
        msg = f'Dear {user.username},\n\nYour batch generation request for project "{project.name}" failed because data has been depleted.\n\nAlanno'
        data.append(("Failed batch generation", msg, None, [user.email]))
    else:
        for anno_id in annotator_ids:
            user = get_object_or_404(User, pk=anno_id)
            msg = f'Dear {user.username},\n\nNew batch has been generated for annotation of "{project.name}" project.\n\nAlanno'
            data.append(("New data for annotation", msg, None, [user.email]))
    data = tuple(data)
    try:
        send_mass_mail(data, fail_silently=False)
    except SMTPException as exc:
        self.retry(exc=exc, countdown=2 ** self.request.retries)


@app.task
def check_batch_completion_task(_, project_id):
    """
    Check if batch is completed by all annotators.
    The method is chained after completion_api_task.

    Parameters
    ----------
    _: None
        dummy parameter used to make the chaining possible
    project_id: int
        ID of the project

    Returns
    -------
    bool: are all annotators done
    """
    project = get_object_or_404(Project, pk=project_id)
    remaining_count = 0
    for anno in project.annotators.all():
        total_batch = project.documents.filter(selectors__id__exact=anno.id)
        batch_remaining = total_batch.exclude(completed_by__id__exact=anno.id).count()
        remaining_count += batch_remaining
    return remaining_count == 0


@app.task(bind=True, max_retries=5)
def send_batch_done_email_task(self, done, project_id):
    """
    Sending email to project managers if batch is completed by all annotators.
    The method is chained after check_batch_completion_task.
    If email fails to send task will be retried 5 times with exponential delay.

    Parameters
    ----------
    done: bool
        are all annotators finished with the batch
    project_id: int
        ID of the project
    """
    if done:
        project = get_object_or_404(Project, pk=project_id)
        data = []
        for manager in project.managers.all():
            msg = f'Dear {manager.username},\n\nAll annotators have finished their batch in "{project.name}" project.\n\nAlanno'
            data.append(("Batch done", msg, None, [manager.email]))
        data = tuple(data)
        try:
            send_mass_mail(data, fail_silently=False)
        except SMTPException as exc:
            self.retry(exc=exc, countdown=2 ** self.request.retries)
