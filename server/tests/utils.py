from celery_progress.backend import ProgressRecorder

from django.contrib.auth.models import User
from server.models import ClassificationProject, ClassificationDocument, Label, DocumentAnnotation


class MockProgressRecorder(ProgressRecorder):

    def __init__(self, mock_task=None):
        super().__init__(mock_task)

    def set_progress(self, current, total, description=""):
        print(f"\r{description} {current}/{total}          ", end='')


def init_clx_project(multilabel=False, user_range=None, label_range=None, document_range=None):
    project = ClassificationProject(name="Single-label project", description="Test", multilabel=multilabel)
    project.save()

    users = []
    if user_range is not None:
        first, last = user_range
        for i in range(first, last):
            username = f"user{i}"
            if User.objects.filter(username=username).count() >= 1:
                users.append(User.objects.filter(username=username).first())
            else:
                users.append(User.objects.create_user(f"user{i}", f"user{i}@test.com"))
                users[-1].save()
            project.annotators.add(users[-1])

    labels = []
    if label_range is not None:
        first, last = label_range
        for i in range(first, last):
            labels.append(Label(text=f"label{i}", project=project))
            labels[-1].save()

    docs = []
    if document_range is not None:
        first, last = document_range
        for i in range(first, last):
            docs.append(ClassificationDocument(text=f"document{i}", project=project))
            docs[-1].save()

    return project, users, labels, docs


def add_clx_annotations(users, docs, labels, annotations):
    for uid, did, lid in annotations:
        annotation = DocumentAnnotation(document=docs[did], label=labels[lid])
        annotation.user = users[uid]
        annotation.save()
        docs[did].completed_by.add(users[uid])
        docs[did].save()
