from django.contrib.auth.mixins import UserPassesTestMixin
from django.shortcuts import get_object_or_404
from rest_framework.permissions import BasePermission, SAFE_METHODS, IsAdminUser

from .models import Project


class IsAnnotationOwner(BasePermission):
    def has_permission(self, request, view):
        user = request.user
        project_id = view.kwargs.get("project_id")
        annotation_id = view.kwargs.get("annotation_id")
        project = get_object_or_404(Project, pk=project_id)
        Annotation = project.get_annotation_class()
        annotation = Annotation.objects.get(id=annotation_id)

        return annotation.user == user


class ProjectManagerMixin(UserPassesTestMixin):
    def test_func(self):
        project_id = self.kwargs["project_id"]
        project = get_object_or_404(Project, pk=project_id)
        return (
            self.request.user.is_superuser
            or self.request.user in project.managers.all()
        )


class ProjectAnnotatorMixin(UserPassesTestMixin):
    def test_func(self):
        project_id = self.kwargs["project_id"]
        project = get_object_or_404(Project, pk=project_id)
        return (
            self.request.user.is_superuser
            or self.request.user in project.annotators.all()
        )


class ProjectUserMixin(UserPassesTestMixin):
    def test_func(self):
        project_id = self.kwargs["project_id"]
        project = get_object_or_404(Project, pk=project_id)
        return (
            self.request.user.is_superuser
            or self.request.user in project.get_all_users()
        )
