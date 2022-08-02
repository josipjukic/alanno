from datetime import datetime, timedelta
from text.information_retrieval import bm25_search

from django_filters.rest_framework import DjangoFilterBackend
from rest_framework.permissions import IsAuthenticated
from rest_framework import viewsets, generics, filters
from django.core.exceptions import ObjectDoesNotExist
from django.shortcuts import get_object_or_404
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.views import APIView

from django.core.cache import cache

from django.contrib.auth.models import User
from django.contrib import messages

from django.db.models import Count, F, Case, When, Max

from .models import Project, Label, Document, ProjectStatistics, UserLog
from .serializers import LabelSerializer

from .permissions import ProjectUserMixin, ProjectManagerMixin, ProjectAnnotatorMixin
from .permissions import IsAnnotationOwner

from rest_framework import status
from server.serializers import ProjectPolymorphicSerializer, UserSerializer

from celery import chain
from .tasks import (
    edit_annotators_api_task,
    guidelines_api_task,
    completion_api_task,
    calculate_project_stats_task,
    create_batch_task,
    send_batch_generation_email_task,
    check_batch_completion_task,
    send_batch_done_email_task,
    change_al_method_api_task,
    change_model_name_api_task,
    change_token_type_api_task,
    change_vectorizer_name_api_task,
    change_adjustable_vocab_api_task,
    change_vocab_max_size_api_task,
    change_vocab_min_freq_api_task,
    change_min_ngram_api_task,
    change_max_ngram_api_task,
    delete_documents_task,
    delete_project_task,
)


class ProjectTypesAPI(APIView):
    permission_classes = (IsAuthenticated,)

    def get(self, request):
        return Response(dict(Project.PROJECT_CHOICES))


class ProjectJoinAPI(APIView):
    permission_classes = (IsAuthenticated,)

    def post(self, request):
        access_code = request.data["access_code"]
        p = Project.objects.filter(access_code=access_code).first()
        if p:
            if request.user in p.annotators.all():
                messages = [f"You are already a part of the project '{p.name}'!"]
                message_type = "notice"
            else:
                messages = [f"You have successfully joined the project '{p.name}'!"]
                message_type = "success"
                p.members.add(request.user)
                p.annotators.add(request.user)
        else:
            messages = ["Incorrect access code!"]
            message_type = "error"

        response_data = {"messages": messages, "message_type": message_type}
        return Response(response_data, status=status.HTTP_200_OK)


class ProjectViewSet(viewsets.ModelViewSet):
    queryset = Project.objects.all()
    serializer_class = ProjectPolymorphicSerializer
    pagination_class = None
    permission_classes = (IsAuthenticated,)

    def get_queryset(self):
        if self.request.user.is_superuser:
            r = Project.objects.all()
        else:
            q1 = self.request.user.manager_in_projects.all()
            q2 = self.request.user.annotator_in_projects.all()
            r = (q1 | q2).distinct()

        return r.order_by("-updated_at")

    @action(methods=["get"], detail=True)
    def progress(self, request, pk=None):
        project = self.get_object()
        return Response(project.get_progress(self.request.user))


class LabelList(ProjectUserMixin, generics.ListCreateAPIView):
    queryset = Label.objects.all()
    serializer_class = LabelSerializer
    pagination_class = None
    permission_classes = (IsAuthenticated,)

    def get_queryset(self):
        queryset = self.queryset.filter(project=self.kwargs["project_id"])
        return queryset

    def create(self, request, *args, **kwargs):
        project = get_object_or_404(Project, pk=self.kwargs["project_id"])
        data = request.data

        if project.hierarchy and data["parent"]:
            parent = project.labels.filter(text=data["parent"]).first()
            data["parent"] = parent.id

        serializer = self.get_serializer(data=data)
        serializer.is_valid(raise_exception=True)
        try:
            self.perform_create(serializer)
        except RuntimeError as err:
            return Response({"error": str(err)}, status=status.HTTP_400_BAD_REQUEST)
        headers = self.get_success_headers(serializer.data)
        return Response(
            serializer.data, status=status.HTTP_201_CREATED, headers=headers
        )

    def perform_create(self, serializer):
        project = get_object_or_404(Project, pk=self.kwargs["project_id"])
        serializer.save(project=project)


class LabelDetail(ProjectUserMixin, generics.RetrieveUpdateDestroyAPIView):
    queryset = Label.objects.all()
    serializer_class = LabelSerializer
    permission_classes = (IsAuthenticated,)

    def get_queryset(self):
        queryset = self.queryset.filter(project=self.kwargs["project_id"])
        return queryset

    def get_object(self):
        queryset = self.filter_queryset(self.get_queryset())
        obj = get_object_or_404(queryset, pk=self.kwargs["label_id"])
        self.check_object_permissions(self.request, obj)
        return obj


class DistributionStatsAPI(ProjectManagerMixin, APIView):
    pagination_class = None
    permission_classes = (IsAuthenticated,)

    def get(self, request, *args, **kwargs):
        p = get_object_or_404(Project, pk=self.kwargs["project_id"])
        total_count = p.documents.filter(is_warm_start=False).count()
        docs = p.get_distributed_documents()
        distributed_count = len(docs)
        data = {"distributed": distributed_count, "total": total_count}
        return Response(data)


class ProjectStatsAPI(ProjectManagerMixin, APIView):
    pagination_class = None
    permission_classes = (IsAuthenticated,)

    def get(self, request, *args, **kwargs):
        forceUpdate = int(request.query_params.get("forceUpdate"))
        project_id = self.kwargs["project_id"]
        project = get_object_or_404(Project, pk=project_id)

        try:
            data = ProjectStatistics.objects.get(project=project)
        except ObjectDoesNotExist:
            data = None

        if forceUpdate == 1:
            if data:
                data.delete()
            data = None

        if data:
            most_recent_doc_update = project.documents.aggregate(Max("updated_at"))[
                "updated_at__max"
            ]
            now = datetime.now()
            if most_recent_doc_update is None:
                most_recent_doc_update = data.created_at
            if (
                most_recent_doc_update <= data.created_at
                or data.created_at + timedelta(minutes=15) > now
            ):
                return Response(
                    {
                        "stats": data.stats,
                        "lastUpdated": data.created_at,
                        "finished": True,
                    }
                )
            else:
                data.delete()

        cache_key = f"{project.id}_stats"
        task_id = cache.get(cache_key)
        task_running = True
        if (
            not task_id
            or calculate_project_stats_task.AsyncResult(task_id).status == "SUCCESS"
            or calculate_project_stats_task.AsyncResult(task_id).status == "FAILURE"
        ):
            task_running = False
            task = calculate_project_stats_task.delay(project_id)
            task_id = task.task_id
            cache.set(cache_key, task_id, 60 * 60)
        print(task_running)
        return Response({"finished": False, "task_running": task_running})


class AnnotatorList(ProjectManagerMixin, generics.ListCreateAPIView):
    serializer_class = UserSerializer
    permission_classes = (IsAuthenticated,)

    def get_queryset(self):
        project = get_object_or_404(Project, pk=self.kwargs["project_id"])
        queryset = project.annotators.all()
        return queryset


class AllMembersAPI(ProjectManagerMixin, generics.ListCreateAPIView):
    permission_classes = (IsAuthenticated,)

    def get(self, request, project_id):
        project = get_object_or_404(Project, pk=project_id)
        annotators = [dict(id=a.id, username=a.username) for a in project.members.all()]
        response_data = [dict(group="Other", annotators=annotators)]
        return Response(response_data, status=status.HTTP_200_OK)


class DocumentList(ProjectUserMixin, generics.ListCreateAPIView):
    queryset = Document.objects.all()
    filter_backends = (
        DjangoFilterBackend,
        filters.SearchFilter,
        filters.OrderingFilter,
    )
    search_fields = ("text",)
    permission_classes = (IsAuthenticated,)

    def get_serializer_class(self):
        project = get_object_or_404(Project, pk=self.kwargs["project_id"])
        self.serializer_class = project.get_document_serializer()

        return self.serializer_class

    def get_queryset(self):
        user_id = self.request.user.id
        search = self.request.query_params.get("search")
        ordering = self.request.query_params.get("ordering")

        project = get_object_or_404(Project, pk=self.kwargs["project_id"])
        queryset = self.queryset.filter(
            project=project.id, selectors__id__exact=user_id
        )

        if search == "active":
            queryset = queryset.exclude(completed_by__id__exact=user_id)
        elif search == "completed":
            queryset = queryset.filter(completed_by__id__exact=user_id)

        if ordering == "round":
            queryset = queryset.order_by("round__number", "id")

        return queryset


class AnnotationList(ProjectUserMixin, generics.ListCreateAPIView):
    pagination_class = None
    permission_classes = (IsAuthenticated,)

    def get_serializer_class(self):
        project = get_object_or_404(Project, pk=self.kwargs["project_id"])
        self.serializer_class = project.get_annotation_serializer()

        return self.serializer_class

    def get_queryset(self):
        project = get_object_or_404(Project, pk=self.kwargs["project_id"])
        document = project.documents.get(id=self.kwargs["doc_id"])
        self.queryset = document.get_annotations()
        self.queryset = self.queryset.filter(user=self.request.user)

        return self.queryset

    def perform_create(self, serializer):
        ts = datetime.utcnow()
        doc = get_object_or_404(Document, pk=self.kwargs["doc_id"])
        project = get_object_or_404(Project, pk=self.kwargs["project_id"])
        log = self.prepare_log(ts, doc, project)

        serializer.save(document=doc, user=self.request.user)
        log.save()

    def prepare_log(self, ts, doc, project):
        if "label" not in self.request.data.keys():
            return Response(status=status.HTTP_400_BAD_REQUEST)
        lid = self.request.data["label"]
        label = get_object_or_404(Label, pk=lid)
        return UserLog(
            type=UserLog.LBL_SELECT,
            timestamp=ts,
            user=self.request.user,
            project=project,
            document=doc,
            label=label,
            metadata=None,
        )


class AnnotationDetail(ProjectAnnotatorMixin, generics.RetrieveUpdateDestroyAPIView):
    permission_classes = (IsAuthenticated, IsAnnotationOwner)

    def get_serializer_class(self):
        project = get_object_or_404(Project, pk=self.kwargs["project_id"])
        self.serializer_class = project.get_annotation_serializer()
        return self.serializer_class

    def get_queryset(self):
        document = get_object_or_404(Document, pk=self.kwargs["doc_id"])
        self.queryset = document.get_annotations()
        return self.queryset

    def get_object(self):
        queryset = self.filter_queryset(self.get_queryset())
        obj = get_object_or_404(queryset, pk=self.kwargs["annotation_id"])
        self.check_object_permissions(self.request, obj)
        return obj

    def destroy(self, request, *args, **kwargs):
        ts = datetime.utcnow()
        instance = self.get_object()
        log = self.prepare_log(
            ts, instance, self.kwargs["doc_id"], self.kwargs["project_id"]
        )
        self.perform_destroy(instance)
        if log:
            log.save()
        return Response(status=status.HTTP_204_NO_CONTENT)

    def prepare_log(self, ts, anno, did, pid):
        if not hasattr(anno, "label"):
            return None
        label = anno.label
        doc = get_object_or_404(Document, pk=did)
        project = get_object_or_404(Project, pk=pid)
        return UserLog(
            type=UserLog.LBL_REMOVE,
            timestamp=ts,
            user=self.request.user,
            project=project,
            document=doc,
            label=label,
            metadata=None,
        )


class GuidelinesAPI(ProjectManagerMixin, APIView):
    permission_classes = (IsAuthenticated,)

    def post(self, request, project_id):
        guidelines_api_task.delay(project_id, request.data["guidelines"])
        return Response(status=status.HTTP_200_OK)


class PhraseDestroy(ProjectUserMixin, APIView):
    permission_classes = (IsAuthenticated,)

    def post(self, request, project_id, doc_id, format=None):
        phrase = self.request.data["phrase"]
        doc = Document.objects.get(pk=doc_id)
        annotations = doc.get_annotations().filter(user=self.request.user)
        annotation = annotations.filter(phrase=phrase)[0]
        annotations.filter(lemma=annotation.lemma).delete()

        return Response(status=status.HTTP_204_NO_CONTENT)


class CompletionAPI(ProjectUserMixin, APIView):
    permission_classes = (IsAuthenticated,)

    def post(self, request, project_id, doc_id):
        completed = request.data["completed"]
        user_id = request.user.id
        chain(
            completion_api_task.s(doc_id, completed, user_id, project_id),
            check_batch_completion_task.s(project_id),
            send_batch_done_email_task.s(project_id),
        )()
        return Response(status=status.HTTP_200_OK)


class BatchAPI(ProjectManagerMixin, APIView):
    permission_classes = (IsAuthenticated,)

    def post(self, request, project_id):
        annotator_ids = request.data["annotators"]
        anno_per_dp = request.data["anno_per_dp"]
        user_id = request.user.id
        if not annotator_ids:
            messages = ["Please select annotators."]
            message_type = "notice"
            data = {"messages": messages, "message_type": message_type}
            return Response(data, status=status.HTTP_200_OK)

        task = chain(
            create_batch_task.s(
                request.data, project_id, annotator_ids, anno_per_dp
            ).set(queue="al_loop"),
            send_batch_generation_email_task.s(project_id, annotator_ids, user_id),
        ).delay()

        cache_key = f"{project_id}_generate_status"
        cache.set(cache_key, task.parent.id, 60 * 60)

        messages = []
        message_type = "success"
        data = {"messages": messages, "message_type": message_type}
        return Response(data, status=status.HTTP_200_OK)


class AddAnnoAPI(ProjectManagerMixin, APIView):
    permission_classes = (IsAuthenticated,)
    pagination_class = None

    def post(self, request, project_id):
        annotator_ids = request.data["annotators"]
        project = get_object_or_404(Project, pk=project_id)

        if not annotator_ids:
            messages.error(request, "At least one annotator has to be chosen.")
            return Response(status=status.HTTP_400_BAD_REQUEST)

        add_list = User.objects.filter(id__in=annotator_ids)
        project.annotators.add(*add_list)

        project.save()

        messages.info(request, "Selected annotators added successfully!")
        return Response(status=status.HTTP_200_OK)


class EditAnnotatorsAPI(ProjectManagerMixin, APIView):
    permission_classes = (IsAuthenticated,)
    pagination_class = None

    def post(self, request, project_id):
        annotator_ids = request.data
        edit_annotators_api_task.delay(project_id, annotator_ids)
        return Response(status=status.HTTP_200_OK)


class RemoveAnnoAPI(ProjectManagerMixin, APIView):
    permission_classes = (IsAuthenticated,)
    pagination_class = None

    def post(self, request, project_id):
        annotator_ids = request.data["annotators"]
        project = get_object_or_404(Project, pk=project_id)

        if not annotator_ids:
            messages.error(request, "At least one annotator has to be chosen.")
            return Response(status=status.HTTP_400_BAD_REQUEST)

        remove_list = User.objects.filter(id__in=annotator_ids)
        project.annotators.remove(*remove_list)

        project.save()

        messages.info(request, "Selected annotators removed successfully!")
        return Response(status=status.HTTP_204_NO_CONTENT)


class BatchInfoAPI(ProjectManagerMixin, APIView):
    permission_classes = (IsAuthenticated,)

    def post(self, request, project_id):
        annotators = request.data
        project = get_object_or_404(Project, pk=project_id)
        info = {}

        anno_info = {}
        for annotator in annotators:
            total = project.documents.filter(selectors__id__exact=annotator["id"])
            active = total.exclude(completed_by__id__exact=annotator["id"])
            anno_info[annotator["id"]] = {
                "total": total.count(),
                "active": active.count(),
            }
        info["info"] = anno_info

        round = project.rounds.order_by("-number").first()
        info["has_round"] = False
        if round:
            info["has_round"] = True
            info["round_number"] = round.number + 1
            info["round_date"] = round.created_at
            info["round_documents"] = round.documents.count()

        return Response(info, status=status.HTTP_200_OK)


class AnnoInfoAPI(ProjectManagerMixin, APIView):
    permission_classes = (IsAuthenticated,)

    def post(self, request, project_id):
        project = get_object_or_404(Project, pk=project_id)

        project_annos = project.annotators.all()
        all_annos = User.objects.all()
        other_annos = all_annos.difference(project_annos)

        info = [{"id": anno.id, "username": anno.username} for anno in other_annos]
        return Response(info, status=status.HTTP_200_OK)


class DeleteProjectAPI(ProjectManagerMixin, APIView):
    permission_classes = (IsAuthenticated,)

    def delete(self, request, project_id):
        """
        Delete project.
        """
        project = get_object_or_404(Project, pk=project_id)
        print(f"Deleting project: {project.name}")

        project.is_deleted = True
        project.name = None
        project.save(update_fields=["is_deleted", "name"])
        delete_project_task.delay(project_id)
        return Response(project_id, status=status.HTTP_200_OK)


class DeleteDocumentsAPI(ProjectManagerMixin, APIView):
    permission_classes = (IsAuthenticated,)

    def delete(self, request, project_id):
        """
        Delete all documents.
        """
        project = get_object_or_404(Project, pk=project_id)
        print(f"Deleting all documents for project: {project.name}")

        doc_ids = list(
            Document.objects.filter(project=project).values_list("id", flat=True)
        )
        Document.objects.filter(project=project).update(is_deleted=True, project=None)
        delete_documents_task.delay(doc_ids)
        project.save(update_fields=["updated_at"])
        return Response(project_id, status=status.HTTP_200_OK)

    def post(self, request, project_id):
        """
        Delete single document.
        """
        doc_id = self.request.data["doc_id"]
        print(f"Deleting doc {doc_id}")

        project = get_object_or_404(Project, pk=project_id)
        doc = Document.objects.get(id=doc_id)
        doc.delete()
        project.save()
        return Response(doc_id, status=status.HTTP_200_OK)


class FetchDocumentsAPI(ProjectManagerMixin, generics.ListCreateAPIView):
    queryset = Document.objects.all()
    filter_backends = (
        DjangoFilterBackend,
        filters.SearchFilter,
    )
    search_fields = ("text",)
    permission_classes = (IsAuthenticated,)

    def get_serializer_class(self):
        project = get_object_or_404(Project, pk=self.kwargs["project_id"])
        self.serializer_class = project.get_document_serializer()

        return self.serializer_class

    def get_queryset(self):
        filter = self.request.query_params.get("filter")

        project = get_object_or_404(Project, pk=self.kwargs["project_id"])
        queryset = self.queryset.filter(project=project.id)

        if filter == "in progress":
            queryset = queryset.annotate(
                num_completed=Count("completed_by", distinct=True),
                num_selectors=Count("selectors", distinct=True),
            )
            queryset = queryset.filter(
                num_selectors__gt=0, num_completed__lt=F("num_selectors")
            )
        elif filter == "finished":
            queryset = queryset.annotate(
                num_completed=Count("completed_by", distinct=True),
                num_selectors=Count("selectors", distinct=True),
            )
            queryset = queryset.filter(
                num_selectors__gt=0, num_completed=F("num_selectors")
            )

        return queryset


class ChangeMainAnnotatorAPI(ProjectManagerMixin, APIView):
    permission_classes = (IsAuthenticated,)

    def post(self, request, project_id):
        project = get_object_or_404(Project, pk=project_id)
        new_user = request.data["user"]
        if not new_user:
            project.main_annotator = None
        else:
            user = project.members.filter(username=new_user).first()
            if not user:
                return Response(
                    f"User with username {new_user} not found among project members.",
                    status=status.HTTP_404_NOT_FOUND,
                )
            project.main_annotator = user
        project.save()
        return Response(status=status.HTTP_200_OK)


class ChangeALMethodAPI(ProjectManagerMixin, APIView):
    permission_classes = (IsAuthenticated,)

    def post(self, request, project_id):
        change_al_method_api_task.delay(project_id, request.data["name"])
        return Response(status=status.HTTP_200_OK)


class ChangeModelNameAPI(ProjectManagerMixin, APIView):
    permission_classes = (IsAuthenticated,)

    def post(self, request, project_id):
        change_model_name_api_task.delay(project_id, request.data["name"])
        return Response(status=status.HTTP_200_OK)


class ChangeTokenTypeAPI(ProjectManagerMixin, APIView):
    permission_classes = (IsAuthenticated,)

    def post(self, request, project_id):
        change_token_type_api_task.delay(project_id, request.data["name"])
        return Response(status=status.HTTP_200_OK)


class ChangeVectorizerNameAPI(ProjectManagerMixin, APIView):
    permission_classes = (IsAuthenticated,)

    def post(self, request, project_id):
        change_vectorizer_name_api_task.delay(project_id, request.data["name"])
        return Response(status=status.HTTP_200_OK)


class ChangeAdjustableVocabAPI(ProjectManagerMixin, APIView):
    permission_classes = (IsAuthenticated,)

    def post(self, request, project_id):
        print(request.data["value"])
        change_adjustable_vocab_api_task.delay(project_id, request.data["value"])
        return Response(status=status.HTTP_200_OK)


class ChangeVocabMaxSizeAPI(ProjectManagerMixin, APIView):
    permission_classes = (IsAuthenticated,)

    def post(self, request, project_id):
        change_vocab_max_size_api_task.delay(project_id, request.data["value"])
        return Response(status=status.HTTP_200_OK)


class ChangeVocabMinFreqAPI(ProjectManagerMixin, APIView):
    permission_classes = (IsAuthenticated,)

    def post(self, request, project_id):
        change_vocab_min_freq_api_task.delay(project_id, request.data["value"])
        return Response(status=status.HTTP_200_OK)


class ChangeMinNgramAPI(ProjectManagerMixin, APIView):
    permission_classes = (IsAuthenticated,)

    def post(self, request, project_id):
        change_min_ngram_api_task.delay(project_id, request.data["value"])
        return Response(status=status.HTTP_200_OK)


class ChangeMaxNgramAPI(ProjectManagerMixin, APIView):
    permission_classes = (IsAuthenticated,)

    def post(self, request, project_id):
        change_max_ngram_api_task.delay(project_id, request.data["value"])
        return Response(status=status.HTTP_200_OK)


class UserLogAPI(ProjectAnnotatorMixin, APIView):
    permission_classes = (IsAuthenticated,)

    def post(self, request, project_id, log_type):
        ts = datetime.utcnow()
        if log_type not in [
            UserLog.ANNO_START,
            UserLog.ANNO_END,
            UserLog.DOC_OPEN,
            UserLog.DOC_LOCK,
            UserLog.DOC_UNLOCK,
            UserLog.DOC_CLOSE,
            UserLog.LBL_SELECT,
            UserLog.LBL_REMOVE,
        ]:
            return Response(status=status.HTTP_400_BAD_REQUEST)

        user = request.user
        meta = request.data["metadata"] if "metadata" in request.data.keys() else None
        try:
            p = Project.objects.get(pk=project_id)
        except Project.DoesNotExist:
            return Response(status=status.HTTP_404_NOT_FOUND)

        d = None
        if log_type in [
            UserLog.DOC_LOCK,
            UserLog.DOC_OPEN,
            UserLog.DOC_UNLOCK,
            UserLog.DOC_CLOSE,
            UserLog.LBL_REMOVE,
            UserLog.LBL_SELECT,
        ]:
            if "document_id" not in request.data.keys():
                return Response(status=status.HTTP_400_BAD_REQUEST)
            did = int(request.data["document_id"])
            try:
                d = Document.objects.get(pk=did)
            except Document.DoesNotExist:
                return Response(status=status.HTTP_404_NOT_FOUND)

        l = None
        if log_type in [UserLog.LBL_REMOVE, UserLog.LBL_SELECT]:
            if "label_id" not in request.data.keys():
                return Response(status=status.HTTP_400_BAD_REQUEST)
            lid = int(request.data["label_id"])
            try:
                l = Label.objects.get(pk=lid)
            except Label.DoesNotExist:
                return Response(status=status.HTTP_404_NOT_FOUND)

        log = UserLog(
            type=log_type,
            timestamp=ts,
            user=user,
            project=p,
            document=d,
            label=l,
            metadata=meta,
        )
        log.save()
        return Response(status=status.HTTP_200_OK)


class GuidedDocumentsAPI(ProjectUserMixin, APIView):
    permission_classes = (IsAuthenticated,)
    pagination = 5
    max_num_docs = 100

    def get(self, request, project_id):
        """
        Get documents.
        """
        project = get_object_or_404(Project, pk=project_id)
        if not project.main_annotator:
            raise PermissionError("Guided Learning is not enabled for this project.")
        if project.main_annotator.id != self.request.user.id:
            raise PermissionError(
                "You are not authorized to manage Guided Learning annotations"
            )

        page_nbr = 1
        query = self.request.query_params.get("query")
        document_indexes = list(
            project.documents.exclude(is_warm_start=True).values_list("id", "indexing")
        )
        doc_id_index = list(zip(*document_indexes))
        doc_ids, doc_indexes = doc_id_index[0], doc_id_index[1]

        # Retrieve indices of max_num_docs number of documents that best fit the query
        sort_inds = bm25_search(query, doc_indexes, doc_ids, project.language)[
            : self.max_num_docs
        ]

        # Get documents by ids while preserving order of sort_inds
        preserved_order = Case(
            *[When(pk=pk, then=pos) for pos, pk in enumerate(sort_inds)]
        )
        documents = project.documents.filter(pk__in=sort_inds).order_by(preserved_order)

        # empty project
        if not documents:
            return Response(
                {
                    "current": [],
                    "maxPage": 1,
                    "page": 1,
                    "pagination": self.pagination,
                },
                status=status.HTTP_200_OK,
            )

        max_page = len(documents) // self.pagination
        if len(documents) % self.pagination != 0:
            max_page += 1

        # page can't be < 1 or > max_page
        page_nbr = min(max(page_nbr, 1), max_page)

        current = [
            {
                "text": doc.text,
                "id": doc.id,
                "raw_html": doc.raw_html,
                "is_selected": doc.is_selected,
                "is_gl_annotated": doc.gl_annotated,
                "annotations": self.format_annotations(doc, project),
            }
            for doc in documents
        ]

        return Response(
            {
                "current": current,
                "maxPage": max_page,
                "page": page_nbr,
                "pagination": self.pagination,
            },
            status=status.HTTP_200_OK,
        )

    def format_annotations(self, doc, project):
        anno_dict = {}
        for anno in doc.get_annotations():
            if anno.user == project.main_annotator and not anno.gl_annotation:
                # Ignore regular annotations of main annotator
                continue
            if anno.user.username not in anno_dict:
                anno_dict[anno.user.username] = []
            if project.is_type_of(Project.DOCUMENT_CLASSIFICATION):
                anno_dict[anno.user.username].append(
                    {
                        "id": anno.id,
                        "label": anno.label.id,
                        "annotator": anno.user.username,
                        "label_text": anno.label.text,
                        "gl_annotation": anno.gl_annotation,
                    }
                )
            elif project.is_type_of(Project.SEQUENCE_LABELING):
                anno_dict[anno.user.username].append(
                    {
                        "id": anno.id,
                        "label": anno.label.id,
                        "start_offset": anno.start_offset,
                        "end_offset": anno.end_offset,
                        "annotator": anno.user.username,
                        "label_text": anno.label.text,
                        "gl_annotation": anno.gl_annotation,
                    }
                )
        if project.main_annotator.username not in anno_dict.keys():
            anno_dict[project.main_annotator.username] = []
        return anno_dict


class GuidedLearningAnnotationList(ProjectUserMixin, generics.ListCreateAPIView):
    pagination_class = None
    permission_classes = (IsAuthenticated,)

    def get_serializer_class(self):
        project = get_object_or_404(Project, pk=self.kwargs["project_id"])
        self.serializer_class = project.get_annotation_serializer()

        return self.serializer_class

    def get_queryset(self):
        project = get_object_or_404(Project, pk=self.kwargs["project_id"])
        document = project.documents.get(id=self.kwargs["doc_id"])
        self.queryset = document.get_annotations()
        self.queryset = self.queryset.filter(user=self.request.user)

        return self.queryset

    def perform_create(self, serializer):
        project = get_object_or_404(Project, pk=self.kwargs["project_id"])
        if (
            not project.main_annotator
            or project.main_annotator.id != self.request.user.id
        ):
            raise PermissionError(
                "You are not authorized to create Guided Learning annotations"
            )
        doc = get_object_or_404(Document, pk=self.kwargs["doc_id"])
        doc.gl_annotated = True
        doc.save()

        serializer.save(document=doc, user=self.request.user, gl_annotation=True)


class GuidedLearningAnnotationDetail(
    ProjectAnnotatorMixin, generics.RetrieveUpdateDestroyAPIView
):
    permission_classes = (IsAuthenticated, IsAnnotationOwner)

    def get_serializer_class(self):
        project = get_object_or_404(Project, pk=self.kwargs["project_id"])
        self.serializer_class = project.get_annotation_serializer()
        return self.serializer_class

    def get_queryset(self):
        document = get_object_or_404(Document, pk=self.kwargs["doc_id"])
        self.queryset = document.get_annotations()
        return self.queryset

    def get_object(self):
        queryset = self.filter_queryset(self.get_queryset())
        obj = get_object_or_404(queryset, pk=self.kwargs["annotation_id"])
        self.check_object_permissions(self.request, obj)
        return obj

    def destroy(self, request, *args, **kwargs):
        project = get_object_or_404(Project, pk=self.kwargs["project_id"])
        if not project.main_annotator or project.main_annotator.id != request.user.id:
            raise PermissionError(
                "You are not authorized to delete Guided Learning annotations"
            )

        doc = get_object_or_404(Document, pk=self.kwargs["doc_id"])
        instance = self.get_object()
        self.perform_destroy(instance)
        if not doc.get_all_annotations().filter(gl_annotation=True):
            doc.gl_annotated = False
            doc.save()
        return Response({"gl_annotated": doc.gl_annotated}, status=status.HTTP_200_OK)


class WhoAmI(APIView):
    permission_classes = (IsAuthenticated,)

    def get(self, request):
        data = {
            "username": request.user.username,
            "use_color": request.user.options.use_color,
        }
        return Response(data, status=status.HTTP_200_OK)


class ColorOptions(ProjectUserMixin, APIView):
    permission_classes = (IsAuthenticated,)

    def post(self, request, project_id):
        user = request.user
        user.options.use_color = request.data["use_color"]
        user.options.save()
        return Response(status=status.HTTP_200_OK)
