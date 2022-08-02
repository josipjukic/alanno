from django.db import models

from .annotation import Annotation
from .document import Document
from .label import Label


class DocumentAnnotation(Annotation):
    document = models.ForeignKey(
        Document, related_name="doc_annotations", on_delete=models.CASCADE
    )
    label = models.ForeignKey(Label, on_delete=models.CASCADE)

    class Meta:
        unique_together = ("document", "user", "label", "gl_annotation")
