from django.db import models
from polymorphic.managers import PolymorphicManager


class DeleteManager(PolymorphicManager):
    def __init__(self, *args, **kwargs):
        self.with_deleted = kwargs.pop("deleted", False)
        super(DeleteManager, self).__init__(*args, **kwargs)

    def get_queryset(self):
        q = super().get_queryset()

        if not self.with_deleted:
            q = q.exclude(is_deleted=True)

        return q
