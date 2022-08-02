import os

from django.conf import settings
from django.contrib.auth.models import User, Group, Permission
from django.core.management.base import BaseCommand


manager_permissions = [
    # PROJECT
    "add_project",
    "change_project",
    "delete_project",
    "view_project",
    "add_classificationproject",
    "change_classificationproject",
    "delete_classificationproject",
    "view_classificationproject",
    "add_contextproject",
    "change_contextproject",
    "delete_contextproject",
    "view_contextproject",
    "add_seqlabelingproject",
    "change_seqlabelingproject",
    "delete_seqlabelingproject",
    "view_seqlabelingproject",
    "add_seq2seqproject",
    "change_seq2seqproject",
    "delete_seq2seqproject",
    "view_seq2seqproject",
    "add_kexproject",
    "change_kexproject",
    "delete_kexproject",
    "view_kexproject",
    # DOCUMENT
    "add_document",
    "change_document",
    "delete_document",
    "view_document",
    "add_classificationdocument",
    "change_classificationdocument",
    "delete_classificationdocument",
    "view_classificationdocument",
    "add_sequencedocument",
    "change_sequencedocument",
    "delete_sequencedocument",
    "view_sequencedocument",
    "add_seq2seqdocument",
    "change_seq2seqdocument",
    "delete_seq2seqdocument",
    "view_seq2seqdocument",
    "add_kexdocument",
    "change_kexdocument",
    "delete_kexdocument",
    "view_kexdocument",
    "can_upload_html",
    # LABEL
    "add_label",
    "change_label",
    "delete_label",
    "view_label",
    # ANNOTATION
    "add_documentannotation",
    "change_documentannotation",
    "delete_documentannotation",
    "view_documentannotation",
    "add_sequenceannotation",
    "change_sequenceannotation",
    "delete_sequenceannotation",
    "view_sequenceannotation",
    "add_seq2seqannotation",
    "change_seq2seqannotation",
    "delete_seq2seqannotation",
    "view_seq2seqannotation",
    "add_kexannotation",
    "change_kexannotation",
    "delete_kexannotation",
    "view_kexannotation",
]

annotator_permissions = [
    # ANNOTATION
    "add_documentannotation",
    "change_documentannotation",
    "delete_documentannotation",
    "view_documentannotation",
    "add_sequenceannotation",
    "change_sequenceannotation",
    "delete_sequenceannotation",
    "view_sequenceannotation",
    "add_seq2seqannotation",
    "change_seq2seqannotation",
    "delete_seq2seqannotation",
    "view_seq2seqannotation",
    "add_kexannotation",
    "change_kexannotation",
    "delete_kexannotation",
    "view_kexannotation",
]


class Command(BaseCommand):
    def handle(self, *args, **options):

        groups = ["manager", "annotator"]
        permissions = {
            "manager": manager_permissions,
            "annotator": annotator_permissions,
        }

        for group_name in groups:
            group, created = Group.objects.get_or_create(name=group_name)
            if created:
                perm_codenames = permissions[group_name]
                perms = Permission.objects.filter(codename__in=perm_codenames).all()
                group.permissions.add(*perms)
                print(
                    f"Created group {group_name} with permissions : {perm_codenames}."
                )
