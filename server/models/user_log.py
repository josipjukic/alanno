from django.db import models
from django.contrib.auth.models import User

from .project import Project
from .document import Document
from .label import Label


class UserLog(models.Model):
    ANNO_START = "anno_start"
    DOC_OPEN = "doc_open"
    DOC_LOCK = "doc_lock"
    DOC_UNLOCK = "doc_unlock"
    DOC_CLOSE = "doc_close"
    LBL_SELECT = "lbl_select"
    LBL_REMOVE = "lbl_remove"
    ANNO_END = "anno_end"

    type = models.CharField(max_length=64)
    timestamp = models.DateTimeField()
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    project = models.ForeignKey(
        Project, on_delete=models.SET_NULL, blank=True, null=True
    )
    document = models.ForeignKey(
        Document, on_delete=models.SET_NULL, blank=True, null=True
    )
    label = models.ForeignKey(
        Label, on_delete=models.SET_NULL, blank=True, null=True)
    metadata = models.CharField(max_length=1024, blank=True, null=True)
