from io import TextIOWrapper

from django.http import HttpResponse, HttpResponseRedirect
from django.views.generic import TemplateView, CreateView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.shortcuts import get_object_or_404
from django.shortcuts import render
from django.urls import reverse
from django.views import View

from django.core.exceptions import ObjectDoesNotExist
from django.core.cache import cache
from django.db.models import Max
from django.http import Http404
from django.contrib import messages

from .permissions import ProjectManagerMixin, ProjectUserMixin
from .forms import ProjectForm
from .models import (
    Project,
    ProjectFactory,
    Label,
    ExportData,
    ProjectStatistics,
)

from server.tasks import (
    data_upload_task,
    get_export_csv_task,
    get_export_json_task,
    calculate_project_stats_task,
    create_batch_task,
)

from datetime import datetime, timedelta


def get_user(obj):
    return obj.request.user._wrapped


def get_locked_project(pk):
    """
    Used to retrieve project and ensure that it is locked in the database.
    Method in which it is called should have `transaction.atomic` decorator.
    """
    try:
        return Project.objects.select_for_update().get(pk=pk)
    except:
        raise Http404()


class Test(TemplateView):
    template_name = "test.html"


class HomePage(TemplateView):
    template_name = "general/home.html"


class SettingsView(TemplateView):
    template_name = "admin/settings.html"


class ProjectView(ProjectUserMixin, LoginRequiredMixin, TemplateView):
    def get_template_names(self):
        project = get_object_or_404(Project, pk=self.kwargs["project_id"])
        return [project.get_template_name()]


class GLAnnotationView(ProjectUserMixin, LoginRequiredMixin, TemplateView):
    def get_template_names(self):
        project = get_object_or_404(Project, pk=self.kwargs["project_id"])
        if project.project_type == "Classification":
            return ["gl_anno/gl_classification.html"]
        else:
            return ["gl_anno/gl_seq_lab.html"]


class ProjectsView(LoginRequiredMixin, CreateView):
    form_class = ProjectForm
    template_name = "general/projects.html"

    def get_form_kwargs(self):
        kwargs = super(ProjectsView, self).get_form_kwargs()
        kwargs["user"] = get_user(self)
        return kwargs

    def form_valid(self, form):
        proj = form.save(commit=False)
        user = get_user(self)

        AL_method = form.cleaned_data["al_method"]
        model_name = form.cleaned_data["model_name"]
        language = form.cleaned_data["language"]

        # NEW PROJECT CREATION
        new_proj = ProjectFactory.create_subclass(proj)
        new_proj.save()
        # new_proj.init_task(language, AL_method, model_name)
        new_proj.members.add(user)
        new_proj.managers.add(user)

        if new_proj.project_type == Project.KEX:
            Label.objects.create(
                text="keyphrase",
                shortcut="k",
                background_color="#3372dd",  # blue
                alt_color="#cf2929",  # red
                project=new_proj,
            )

        new_proj.save()
        return HttpResponseRedirect(reverse("settings", args=[new_proj.id]))


class DatasetView(ProjectManagerMixin, LoginRequiredMixin, TemplateView):
    template_name = "admin/dataset.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        return context


class LabelView(ProjectManagerMixin, LoginRequiredMixin, TemplateView):
    template_name = "admin/label.html"


class InstructionsView(LoginRequiredMixin, TemplateView):
    template_name = "admin/instructions.html"


class StatsView(ProjectManagerMixin, LoginRequiredMixin, TemplateView):
    template_name = "admin/stats.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        project = get_object_or_404(Project, pk=context["project_id"])
        count = ProjectStatistics.objects.filter(project=project).count()
        context["data_ready"] = count > 0

        cache_key = f"{project.id}_stats"
        task_id = cache.get(cache_key)
        if (
            task_id
            and calculate_project_stats_task.AsyncResult(task_id).status != "SUCCESS"
            and calculate_project_stats_task.AsyncResult(task_id).status != "FAILURE"
        ):
            context["task_id"] = task_id
        return context


class GuidelinesView(ProjectManagerMixin, LoginRequiredMixin, TemplateView):
    template_name = "admin/guidelines.html"


class ControlView(ProjectManagerMixin, LoginRequiredMixin, TemplateView):
    template_name = "admin/control.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        project = get_object_or_404(Project, pk=context["project_id"])

        cache_key = f"{project.id}_generate_status"
        task_id = cache.get(cache_key)
        if task_id:
            task = create_batch_task.AsyncResult(task_id)
            if task.status != "SUCCESS" and task.status != "FAILURE":
                context["task_id"] = task_id
            elif task.status == "SUCCESS":
                result = task.get()
                if result == "success":
                    context["success"] = "Batch generation was successful."
                else:
                    context["error"] = "Data has been depleted."
                cache.delete(cache_key)
            elif task.status == "FAILURE":
                context["error"] = "Batch generation failed."
                cache.delete(cache_key)
        return context


class DataUpload(ProjectManagerMixin, LoginRequiredMixin, TemplateView):
    template_name = "admin/import.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        project = get_object_or_404(Project, pk=context["project_id"])
        user = self.request.user

        cache_key = f"{user.id}_{project.id}_import"
        task_id = cache.get(cache_key)
        if task_id:
            context["task_id"] = task_id
        return context


class DataUploadFile(ProjectManagerMixin, LoginRequiredMixin, View):
    def post(self, request, *args, **kwargs):
        project_id = kwargs.get("project_id")
        form_data = TextIOWrapper(
            request.FILES["file"].file, encoding="utf-8"
        ).readlines()
        task = data_upload_task.delay(project_id, form_data)

        cache_key = f"{self.request.user.id}_{project_id}_import"
        cache.set(cache_key, task.task_id, 60 * 60)

        return HttpResponseRedirect(reverse("import", args=[project_id]))


class DataDownload(ProjectManagerMixin, LoginRequiredMixin, TemplateView):
    template_name = "admin/export.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        project = get_object_or_404(Project, pk=context["project_id"])
        user = self.request.user
        try:
            context["data"] = ExportData.objects.get(user=user, project=project)
            context["data_ready"] = True
        except ObjectDoesNotExist:
            context["data_ready"] = False

        cache_key = f"{user.id}_{project.id}_export"
        task_id = cache.get(cache_key)
        if (
            task_id
            and calculate_project_stats_task.AsyncResult(task_id).status != "SUCCESS"
            and calculate_project_stats_task.AsyncResult(task_id).status != "FAILURE"
        ):
            context["task_id"] = task_id
        return context


class DataDownloadFile(ProjectManagerMixin, LoginRequiredMixin, View):
    def get(self, request, *args, **kwargs):
        project_id = self.kwargs["project_id"]
        project = get_object_or_404(Project, pk=project_id)
        user = request.user
        req = request.GET
        export_format = req.get("format")
        aggregation = req.get("aggregation") == "aggregation" and project.is_type_of(Project.DOCUMENT_CLASSIFICATION)
        unlabeled = req.get("unlabeled") == "unlabeled"
        action = req.get("action")

        try:
            data = ExportData.objects.get(user=user, project=project)
        except ObjectDoesNotExist:
            data = None
        data_is_done = False

        if data:
            max_doc_update = project.documents.aggregate(Max("updated_at"))[
                "updated_at__max"
            ]
            now = datetime.now()
            if action == "download":
                filename = "_".join(project.name.lower().split())
                response = HttpResponse(content_type=f"text/{data.format}")
                response[
                    "Content-Disposition"
                ] = f'attachment; filename="{filename}.{data.format}"'
                response.write(data.text)
                return response
            elif (
                (
                    not max_doc_update
                    or max_doc_update < data.created_at
                    or data.created_at + timedelta(minutes=15) > now
                )
                and export_format == data.format
                and aggregation == data.is_aggregated
                and unlabeled == data.is_unlabeled
            ):
                data_is_done = True
            else:
                data.delete()

        cache_key = f"{user.id}_{project.id}_export"
        task_id = cache.get(cache_key)
        if action == "create" and not data_is_done:
            task_options = {"csv": get_export_csv_task, "json": get_export_json_task}
            if (
                not task_id
                or task_options[export_format].AsyncResult(task_id).status == "SUCCESS"
                or task_options[export_format].AsyncResult(task_id).status == "FAILURE"
            ):
                if unlabeled:
                    docs = project.documents.all()
                else:
                    docs = project.filter_docs(labeled=True).distinct()
                doc_ids = [doc.id for doc in docs]

                task = task_options[export_format].delay(
                    user.id, doc_ids, project_id, aggregation, unlabeled
                )
                task_id = task.task_id
                cache.set(cache_key, task_id, 60 * 60)
        else:
            task_id = None

        context = {
            "task_id": task_id,
            "project_id": project_id,
            "data_ready": data_is_done,
            "data": data,
        }
        return render(
            request,
            "admin/export.html",
            context=context,
        )
