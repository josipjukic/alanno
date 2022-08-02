from django.db import models
from .project import Project


class Round(models.Model):
    number = models.IntegerField(null=True, blank=True)
    project = models.ForeignKey(
        Project, on_delete=models.SET_NULL, related_name="rounds", null=True, blank=True
    )
    created_at = models.DateTimeField(auto_now_add=True)
