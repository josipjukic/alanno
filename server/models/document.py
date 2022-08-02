from django.db import models
from server.managers import DeleteManager
from polymorphic.models import PolymorphicModel
from django.contrib.postgres.fields import ArrayField
from picklefield.fields import PickledObjectField
from abc import abstractmethod
from django.contrib.auth.models import User

from .project import Project
from .round import Round


class Document(PolymorphicModel):
    objects = DeleteManager()  # Filter by is_deleted by default
    objects_with_deleted = DeleteManager(deleted=True)
    is_deleted = models.BooleanField(default=False, null=True)

    document_id = models.CharField(max_length=100, blank=True, null=True)
    text = models.TextField()
    raw_html = models.TextField(blank=True, null=True)
    indexing = models.TextField(blank=True, null=True)
    html_mapping = ArrayField(
        models.IntegerField(null=True, blank=True), null=True, blank=True
    )
    project = models.ForeignKey(
        Project,
        related_name="documents",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
    )
    selectors = models.ManyToManyField(User, related_name="selected_docs")

    is_selected = models.BooleanField(default=False, null=True)
    is_test = models.BooleanField(default=False, null=True)
    is_al = models.BooleanField(default=False, null=True)
    is_warm_start = models.BooleanField(default=False, null=True)
    gl_annotated = models.BooleanField(
        default=False, null=True
    )  # Document was annotated in guided learning mode
    completed_by = models.ManyToManyField(User, related_name="completed_docs")

    round = models.ForeignKey(
        Round, on_delete=models.CASCADE, related_name="documents", null=True, blank=True
    )
    updated_at = models.DateTimeField(auto_now=True)

    # [(raw text:str, lemma:str, start offset:int, end_offset:int)]
    lemmas = ArrayField(
        PickledObjectField(null=True, editable=True), null=True, blank=True
    )

    def clean_text(self):
        return " ".join(self.text.split())

    def to_csv(self, *args, **kwargs):
        return self.make_dataset(*args, **kwargs)

    @abstractmethod
    def get_annotations(self, is_null=False):
        raise NotImplementedError

    @abstractmethod
    def get_all_annotations(self):
        raise NotImplementedError

    @abstractmethod
    def make_dataset(self, *args, **kwargs):
        raise NotImplementedError

    def to_json(self, *args, **kwargs):
        return self.make_dataset_json(*args, **kwargs)

    @abstractmethod
    def make_dataset_json(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_gold_label(self):
        raise NotImplementedError

    @abstractmethod
    def get_gl_annotations(self):
        raise NotImplementedError

    def has_gold_label(self):
        return bool(self.get_gold_label())

    def __str__(self):
        return self.text[:50]
