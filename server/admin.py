from django.contrib import admin

from .models import UserLog, UserOptions
from .models import ExportData, ProjectStatistics
from .models import (
    Project,
    Document,
    Annotation,
    Label,
    Round,
    ClassificationProject,
    ClassificationDocument,
    DocumentAnnotation,
    SeqLabelingProject,
    SequenceDocument,
    SequenceAnnotation,
)


@admin.register(Project, ClassificationProject, SeqLabelingProject)
class ProjectAdmin(admin.ModelAdmin):
    save_as = True
    filter_horizontal = ("annotators", "managers", "members")
    list_filter = (
        "al_mode",
        "project_type",
        "annotators",
    )
    search_fields = ("name",)


@admin.register(Document, ClassificationDocument, SequenceDocument)
class DocumentAdmin(admin.ModelAdmin):
    filter_horizontal = (
        "selectors",
        "completed_by",
    )
    list_filter = (
        "project",
        "is_selected",
        "is_al",
        "is_test",
        "is_warm_start",
        "gl_annotated",
        "round__number",
    )
    search_fields = (
        "selectors__username",
        "text",
    )


@admin.register(DocumentAnnotation, SequenceAnnotation)
class AnnotationAdmin(admin.ModelAdmin):
    list_filter = ("document__project",)
    search_fields = ("user__username",)


admin.site.register(Label)
admin.site.register(ExportData)
admin.site.register(ProjectStatistics)
admin.site.register(Round)
admin.site.register(UserLog)
admin.site.register(UserOptions)
