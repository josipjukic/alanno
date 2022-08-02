from abc import abstractmethod
import re


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


from rest_polymorphic.serializers import PolymorphicSerializer
from rest_framework import serializers

from django.contrib.auth.models import User


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = (
            "id",
            "username",
        )


class LabelSerializer(serializers.ModelSerializer):
    children = serializers.SerializerMethodField()

    def get_children(self, instance):
        request = self.context["request"]
        if request:
            qset = instance.children.all()
            children = []
            for lab in qset:
                children.append({"id": lab.id, "text": lab.text})
            return children

    class Meta:
        model = Label
        fields = (
            "id",
            "text",
            "shortcut",
            "parent",
            "children",
            "background_color",
            "text_color",
            "is_leaf",
        )


class ProjectSerializer(serializers.ModelSerializer):
    annotators = UserSerializer(many=True)
    managers = UserSerializer(many=True)
    main_annotator = UserSerializer(many=False)

    class Meta:
        model = Project
        fields = (
            "id",
            "name",
            "description",
            "guidelines",
            "managers",
            "annotators",
            "main_annotator",
            "project_type",
            "image",
            "updated_at",
            "language",
            "multilabel",
            "access_code",
            "al_method",
            "model_name",
            "vectorizer_name",
            "adjustable_vocab",
            "vocab_max_size",
            "vocab_min_freq",
            "token_type",
            "min_ngram",
            "max_ngram",
            "al_mode",
        )


class ProjectFilteredPrimaryKeyRelatedField(serializers.PrimaryKeyRelatedField):
    def get_queryset(self):
        view = self.context.get("view", None)
        request = self.context.get("request", None)
        queryset = super(ProjectFilteredPrimaryKeyRelatedField, self).get_queryset()
        if not request or not queryset or not view:
            return None
        return queryset.filter(project=view.kwargs["project_id"])


class DocumentAnnotationSerializer(serializers.ModelSerializer):
    label = ProjectFilteredPrimaryKeyRelatedField(queryset=Label.objects.all())
    annotator = serializers.SerializerMethodField()
    label_text = serializers.SerializerMethodField()

    def get_annotator(self, instance):
        return instance.user.username

    def get_label_text(self, instance):
        return instance.label.text

    class Meta:
        model = DocumentAnnotation
        fields = ("id", "label", "annotator", "label_text", "gl_annotation")

    def create(self, validated_data):
        annotation = DocumentAnnotation.objects.create(**validated_data)
        # annotation.update_task()
        return annotation


class ClassificationProjectSerializer(ProjectSerializer):
    class Meta:
        model = ClassificationProject
        fields = ProjectSerializer.Meta.fields


class SeqLabelingProjectSerializer(ProjectSerializer):
    class Meta:
        model = SeqLabelingProject
        fields = ProjectSerializer.Meta.fields


class ProjectPolymorphicSerializer(PolymorphicSerializer):
    model_serializer_mapping = {
        Project: ProjectSerializer,
        ClassificationProject: ClassificationProjectSerializer,
        SeqLabelingProject: SeqLabelingProjectSerializer,
    }


class DocumentSerializer(serializers.ModelSerializer):
    annotations = serializers.SerializerMethodField()
    completed_by_user = serializers.SerializerMethodField()
    round_number = serializers.SerializerMethodField()
    selectors = serializers.SerializerMethodField()
    completed = serializers.SerializerMethodField()
    all_annotations = serializers.SerializerMethodField()

    @abstractmethod
    def get_annotation_serializer(self):
        raise NotImplementedError

    def get_annotations(self, instance):
        request = self.context["request"]
        if request:
            annotations = instance.get_annotations().filter(
                user=request.user, gl_annotation=False
            )
            serializer = self.get_annotation_serializer()(annotations, many=True)
            data = serializer.data if serializer.data else []
            return data

    def get_completed_by_user(self, instance):
        request = self.context["request"]
        if request:
            completed = instance.completed_by.filter(pk=request.user.id).exists()
            return completed

    def get_round_number(self, instance):
        request = self.context["request"]
        if request:
            if instance.round:
                round_number = instance.round.number
                return round_number
            else:
                return None

    def get_selectors(self, instance):
        request = self.context["request"]
        if request:
            selectors = instance.selectors.values_list("username", flat=True)
            return sorted(selectors)

    def get_completed(self, instance):
        request = self.context["request"]
        if request:
            completed = instance.completed_by.values_list("username", flat=True)
            return sorted(completed)

    def get_all_annotations(self, instance):
        request = self.context["request"]
        if request:
            annotations = instance.get_annotations().filter(gl_annotation=False)
            serializer = self.get_annotation_serializer()(annotations, many=True)
            data = serializer.data if serializer.data else []
            return sorted(data, key=lambda x: (x["annotator"], x["label_text"]))

    class Meta:
        model = Document
        fields = (
            "id",
            "text",
            "is_selected",
            "is_test",
            "is_al",
            "round_number",
            "raw_html",
            "html_mapping",
            "annotations",
            "completed_by_user",
            "selectors",
            "completed",
            "all_annotations",
        )


class ClassificationDocumentSerializer(DocumentSerializer):
    def get_annotation_serializer(self):
        return DocumentAnnotationSerializer

    class Meta:
        model = ClassificationDocument
        fields = DocumentSerializer.Meta.fields


class SequenceDocumentSerializer(DocumentSerializer):
    def get_annotation_serializer(self):
        return SequenceAnnotationSerializer

    class Meta:
        model = SequenceDocument
        fields = DocumentSerializer.Meta.fields


class DocumentPolymorphicSerializer(PolymorphicSerializer):
    model_serializer_mapping = {
        Document: DocumentSerializer,
        ClassificationDocument: ClassificationDocumentSerializer,
        SequenceDocument: SequenceDocumentSerializer,
    }


class SequenceAnnotationSerializer(serializers.ModelSerializer):
    label = ProjectFilteredPrimaryKeyRelatedField(queryset=Label.objects.all())
    annotator = serializers.SerializerMethodField()
    label_text = serializers.SerializerMethodField()

    def get_annotator(self, instance):
        return instance.user.username

    def get_label_text(self, instance):
        return instance.label.text

    class Meta:
        model = SequenceAnnotation
        fields = (
            "id",
            "label",
            "start_offset",
            "end_offset",
            "annotator",
            "label_text",
            "gl_annotation",
        )

    def create(self, validated_data):
        annotation = SequenceAnnotation.objects.create(**validated_data)
        # annotation.update_task()
        return annotation


class RoundSerializer(serializers.ModelSerializer):
    class Meta:
        model = Round
        fields = ("id", "number")
