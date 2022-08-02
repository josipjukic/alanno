from .annotation import Annotation
from .document import Document
from .label import Label
from django.db import models

from django.core.exceptions import ValidationError


class SequenceAnnotation(Annotation):
    document = models.ForeignKey(
        Document, related_name="seq_annotations", on_delete=models.CASCADE
    )
    label = models.ForeignKey(Label, on_delete=models.CASCADE)
    start_offset = models.IntegerField()
    end_offset = models.IntegerField()

    def clean(self):
        if self.start_offset >= self.end_offset:
            raise ValidationError("start_offset is after end_offset")

    class Meta:
        unique_together = (
            "document",
            "user",
            "label",
            "start_offset",
            "end_offset",
            "gl_annotation",
        )
