from django.db import models
from django.contrib.auth.models import User


class Annotation(models.Model):
    user = models.ForeignKey(
        User, on_delete=models.CASCADE, blank=True, null=True)
    gl_annotation = models.BooleanField(
        default=False, null=True
    )  # Was created via guided learning

    class Meta:
        abstract = True
