from django.shortcuts import get_object_or_404
from server.models import Project


def common_context(request):
    kwargs = request.resolver_match.kwargs
    key = "project_id"
    if key in kwargs:
        project = get_object_or_404(Project, pk=kwargs["project_id"])
        return {
            "project_id": project.id,
            "project_name": project.name,
            "project_type": project.project_type,
            "project_description": project.description,
            "al_mode": project.al_mode,
            "project_multilabel": project.multilabel,
            "hierarchy": project.hierarchy,
            "managers": project.managers.all(),
            "annotators": sorted(
                project.annotators.all(), key=lambda a: a.username.lower()
            ),
            "main_annotator": project.main_annotator
        }
    else:
        return {}
