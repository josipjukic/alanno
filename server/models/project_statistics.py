from django.db import models
from .project import Project
from picklefield.fields import PickledObjectField


class ProjectStatistics(models.Model):
    stats = PickledObjectField(null=True, blank=True, editable=True)
    project = models.OneToOneField(
        Project, on_delete=models.CASCADE, related_name="stats"
    )
    created_at = models.DateTimeField(auto_now_add=True)
