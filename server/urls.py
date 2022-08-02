from django.urls import path, reverse
from rest_framework import routers
from django.urls import re_path, include

from .views import (
    HomePage,
    ProjectView,
    GLAnnotationView,
    DatasetView,
    DataUpload,
    LabelView,
    StatsView,
    ControlView,
    ProjectsView,
    DataDownload,
    DataDownloadFile,
    SettingsView,
    Test,
    DataUploadFile,
    InstructionsView,
)

from .api import (
    AllMembersAPI,
    ProjectViewSet,
    LabelList,
    ProjectStatsAPI,
    ProjectTypesAPI,
    ProjectJoinAPI,
    LabelDetail,
    AnnotationList,
    AnnotationDetail,
    DocumentList,
    PhraseDestroy,
    CompletionAPI,
    BatchAPI,
    BatchInfoAPI,
    RemoveAnnoAPI,
    AddAnnoAPI,
    AnnoInfoAPI,
    DeleteDocumentsAPI,
    DeleteProjectAPI,
    FetchDocumentsAPI,
    AnnotatorList,
    GuidelinesAPI,
    EditAnnotatorsAPI,
    DistributionStatsAPI,
    ChangeMainAnnotatorAPI,
    ChangeALMethodAPI,
    ChangeModelNameAPI,
    ChangeTokenTypeAPI,
    ChangeVectorizerNameAPI,
    ChangeAdjustableVocabAPI,
    ChangeVocabMaxSizeAPI,
    ChangeVocabMinFreqAPI,
    ChangeMinNgramAPI,
    ChangeMaxNgramAPI,
    UserLogAPI,
    GuidedDocumentsAPI,
    GuidedLearningAnnotationList,
    GuidedLearningAnnotationDetail,
    WhoAmI,
    ColorOptions,
)

router = routers.DefaultRouter()
router.register(r"projects", ProjectViewSet)


urlpatterns = [
    path("", HomePage.as_view(), name="home"),
    path(
        "api/projects/<int:project_id>/log/<str:log_type>",
        UserLogAPI.as_view(),
        name="user-log-api",
    ),
    path(
        "api/projects/<int:project_id>/stats/",
        ProjectStatsAPI.as_view(),
        name="stats-api",
    ),
    path(
        "api/projects/<int:project_id>/distribution-stats/",
        DistributionStatsAPI.as_view(),
        name="distribution-stats",
    ),
    path("api/projects/<int:project_id>/labels/", LabelList.as_view(), name="labels"),
    path(
        "api/projects/<int:project_id>/labels/<int:label_id>",
        LabelDetail.as_view(),
        name="label",
    ),
    path("api/projects/<int:project_id>/docs/", DocumentList.as_view(), name="docs"),
    path(
        "api/projects/<int:project_id>/generate-batch",
        BatchAPI.as_view(),
        name="generate-batch",
    ),
    path(
        "api/projects/<int:project_id>/remove-annotators",
        RemoveAnnoAPI.as_view(),
        name="remove-annotators",
    ),
    path(
        "api/projects/<int:project_id>/add-annotators",
        AddAnnoAPI.as_view(),
        name="add-annotators",
    ),
    path(
        "api/projects/<int:project_id>/edit-annotators",
        EditAnnotatorsAPI.as_view(),
        name="edit-annotators",
    ),
    path(
        "api/projects/<int:project_id>/annotators",
        AnnotatorList.as_view(),
        name="annotators",
    ),
    path(
        "api/projects/<int:project_id>/all-members",
        AllMembersAPI.as_view(),
        name="all-members",
    ),
    path(
        "api/projects/<int:project_id>/batch-info",
        BatchInfoAPI.as_view(),
        name="batch-info",
    ),
    path(
        "api/projects/<int:project_id>/anno-info",
        AnnoInfoAPI.as_view(),
        name="anno-info",
    ),
    path(
        "api/projects/<int:project_id>/docs/<int:doc_id>/annotations/",
        AnnotationList.as_view(),
        name="annotations",
    ),
    path(
        "api/projects/<int:project_id>/docs/<int:doc_id>/annotations-gl/",
        GuidedLearningAnnotationList.as_view(),
        name="annotations-gl",
    ),
    path(
        "api/projects/<int:project_id>/docs/<int:doc_id>/annotations/<int:annotation_id>",
        AnnotationDetail.as_view(),
        name="ann",
    ),
    path(
        "api/projects/<int:project_id>/docs/<int:doc_id>/annotations-gl/<int:annotation_id>",
        GuidedLearningAnnotationDetail.as_view(),
        name="ann-gl",
    ),
    path(
        "api/projects/<int:project_id>/docs/<int:doc_id>/phrases",
        PhraseDestroy.as_view(),
        name="phrase-destroy",
    ),
    path(
        "api/projects/<int:project_id>/docs/<int:doc_id>/completion",
        CompletionAPI.as_view(),
        name="completion",
    ),
    path(
        "api/projects/<int:project_id>/guidelines",
        GuidelinesAPI.as_view(),
        name="guidelines",
    ),
    path(
        "api/projects/<int:project_id>/main-annotator",
        ChangeMainAnnotatorAPI.as_view(),
        name="main-annotator",
    ),
    path(
        "api/projects/<int:project_id>/al-method",
        ChangeALMethodAPI.as_view(),
        name="al-method",
    ),
    path(
        "api/projects/<int:project_id>/model-name",
        ChangeModelNameAPI.as_view(),
        name="model-name",
    ),
    path(
        "api/projects/<int:project_id>/token-type",
        ChangeTokenTypeAPI.as_view(),
        name="token-type",
    ),
    path(
        "api/projects/<int:project_id>/vectorizer-name",
        ChangeVectorizerNameAPI.as_view(),
        name="vectorizer-name",
    ),
    path(
        "api/projects/<int:project_id>/adjustable-vocab",
        ChangeAdjustableVocabAPI.as_view(),
        name="adjustable-vocab",
    ),
    path(
        "api/projects/<int:project_id>/vocab-max-size",
        ChangeVocabMaxSizeAPI.as_view(),
        name="vocab-max-size",
    ),
    path(
        "api/projects/<int:project_id>/vocab-min-freq",
        ChangeVocabMinFreqAPI.as_view(),
        name="vocab-min-freq",
    ),
    path(
        "api/projects/<int:project_id>/min-ngram",
        ChangeMinNgramAPI.as_view(),
        name="min-ngram",
    ),
    path(
        "api/projects/<int:project_id>/max-ngram",
        ChangeMaxNgramAPI.as_view(),
        name="max-ngram",
    ),
    path(
        "api/projects/<int:project_id>/get-documents/",
        FetchDocumentsAPI.as_view(),
        name="get-documents",
    ),
    path(
        "api/projects/<int:project_id>/retrieve-documents",
        GuidedDocumentsAPI.as_view(),
        name="retrieve-documents",
    ),
    path(
        "api/projects/<int:project_id>/delete-documents",
        DeleteDocumentsAPI.as_view(),
        name="delete-documents",
    ),
    path(
        "api/projects/<int:project_id>/delete-project",
        DeleteProjectAPI.as_view(),
        name="delete-project",
    ),
    path("api/projects/types", ProjectTypesAPI.as_view(), name="types-api"),
    path("api/projects/join-project", ProjectJoinAPI.as_view(), name="join-project"),
    path("projects/", ProjectsView.as_view(), name="projects"),
    path("projects/<int:project_id>/export/", DataDownload.as_view(), name="export"),
    path(
        "projects/<int:project_id>/download_file",
        DataDownloadFile.as_view(),
        name="download_file",
    ),
    path(
        "projects/<int:project_id>/annotation/",
        ProjectView.as_view(),
        name="annotation",
    ),
    path(
        "projects/<int:project_id>/gl_annotation/",
        GLAnnotationView.as_view(),
        name="gl_annotation",
    ),
    path("projects/<int:project_id>/data/", DatasetView.as_view(), name="dataset"),
    path("projects/<int:project_id>/import/", DataUpload.as_view(), name="import"),
    path(
        "projects/<int:project_id>/upload_file",
        DataUploadFile.as_view(),
        name="upload_file",
    ),
    path(
        "projects/<int:project_id>/settings/", SettingsView.as_view(), name="settings"
    ),
    path(
        "projects/<int:project_id>/labels/",
        LabelView.as_view(),
        name="label-management",
    ),
    path("instructions", InstructionsView.as_view(), name="instructions"),
    path("projects/<int:project_id>/stats/", StatsView.as_view(), name="stats"),
    path("projects/<int:project_id>/control/", ControlView.as_view(), name="control"),
    path("test/", Test.as_view(), name="test"),
    path("api/whoami", WhoAmI.as_view(), name="whoami"),
    path(
        "api/projects/<int:project_id>/color-options",
        ColorOptions.as_view(),
        name="color-options",
    ),
]
