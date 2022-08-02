import string
from django.db import models


from .project import Project


class Label(models.Model):
    # TODO - add prefix keys to shortcuts
    PREFIX_KEYS = (("ctrl", "ctrl"), ("shift", "shift"),
                   ("ctrl shift", "ctrl shift"))
    KEY_CHOICES = (
        (u, c) for u, c in zip(string.ascii_lowercase, string.ascii_lowercase)
    )
    COLOR_CHOICES = ()

    text = models.CharField(max_length=100)
    shortcut = models.CharField(
        null=True, blank=True, max_length=10, choices=KEY_CHOICES
    )
    project = models.ForeignKey(
        Project, related_name="labels", on_delete=models.CASCADE
    )

    background_color = models.CharField(max_length=7, default="#ffffff")
    alt_color = models.CharField(
        max_length=7, default="#ffffff", null=True, blank=True)
    text_color = models.CharField(max_length=7, default="#ffffff")

    is_leaf = models.BooleanField(null=True, blank=True)
    parent = models.ForeignKey(
        "Label",
        related_name="children",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
    )

    def __str__(self):
        return self.text

    class Meta:
        unique_together = (
            ("project", "text"),
            # TODO: add unique check for shortcut
        )
