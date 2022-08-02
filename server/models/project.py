from abc import abstractmethod
from itertools import combinations
from collections import Counter

from random import sample
import random
import pandas as pd
import numpy as np

from django.contrib.staticfiles.storage import staticfiles_storage
from django.contrib.auth.models import User
from django.db import models
from django.core.validators import MinValueValidator
from django.db.models import Count

from statsmodels.stats.inter_rater import fleiss_kappa
from nltk.metrics.agreement import AnnotationTask
from nltk.metrics.distance import masi_distance
from sklearn.metrics import cohen_kappa_score


from django.utils.crypto import get_random_string
from django.urls import reverse

from picklefield.fields import PickledObjectField
from polymorphic.models import PolymorphicModel

from time import time

from server.managers import DeleteManager

from al.samplers.constants import get_al_sampler
from al import metrics
from al.models import (
    SklearnTM,
    TorchTM,
    get_model,
    RECURSIVE_MODELS,
    TORCH_MODELS,
    TRANSFORMER_MODELS,
)

from annotation_base.db_utils import (
    get_unselected_split,
    get_labeled_split,
)

from text.preprocessing import (
    make_labeled_dataset,
    make_dataset,
    get_tokenizer,
    get_vectorizer,
    get_posttokenize_hooks,
    get_bert_model_name,
    VectorizedVocab,
)
from podium import Field, LabelField, MultilabelField, Dataset, Iterator
import torch.nn as nn


class Project(PolymorphicModel):
    # General options constants
    # For adding a new project type, add a constant and add it to project choices
    # After that implement concrete instances of classes for the new type:
    # Project, Document, Annotation, DocumentSerializer, AnnotationSerializer
    DOCUMENT_CLASSIFICATION = "Classification"
    SEQUENCE_LABELING = "Sequence Labeling"
    KEX = "Keyphrase Extraction"
    NER = "Named Entity Recognition"
    PROJECT_CHOICES = (
        (DOCUMENT_CLASSIFICATION, "Classification"),
        (SEQUENCE_LABELING, "Sequence Labeling"),
        (KEX, "Keyphrase Extraction"),
        (NER, "Named Entity Recognition"),
    )

    TYPE_MAPPING = {
        DOCUMENT_CLASSIFICATION: "ClassificationProject",
        SEQUENCE_LABELING: "SeqLabelingProject",
        KEX: "SeqLabelingProject",
        NER: "SeqLabelingProject",
    }

    PLAIN = "Plain"
    ACTIVE_LEARNING = "ActiveLearning"

    LEAST_CONFIDENT = "least_conf"
    MARGIN = "margin"
    ENTROPY = "entropy"
    ENTROPY_DENSITY = "entropy_density"
    MULTILABEL_UNCERTAINTY = "multilab_uncert"
    CORE_SET = "core_set"
    BADGE = "badge"
    AL_METHOD_CHOICES = (
        (LEAST_CONFIDENT, "Least confident"),
        (MARGIN, "Margin"),
        (ENTROPY, "Entropy"),
        (ENTROPY_DENSITY, "Entropy + Informative density"),
        (MULTILABEL_UNCERTAINTY, "Multi-label uncertainty"),
        (CORE_SET, "Core-set"),
        (BADGE, "BADGE"),
    )

    LOG_REG = "log_reg"
    LINEAR_SVM = "linear_svm"
    KERNEL_SVM = "kernel_svm"
    RFC = "rfc"
    DUMMY = "dummy"
    MLP = "mlp"
    RNN = "rnn"
    LSTM = "lstm"
    GRU = "gru"
    BERT = "bert"
    MODEL_NAME_CHOICES = (
        (LOG_REG, "Logistic regression"),
        (LINEAR_SVM, "Linear SVM"),
        (KERNEL_SVM, "Kernel SVM"),
        (RFC, "Random forest classifier"),
        (MLP, "Multilayer perceptron"),
        (RNN, "RNN"),
        (LSTM, "LSTM"),
        (GRU, "GRU"),
        (BERT, "BERT"),
    )

    TF_IDF = "tf_idf"
    VEC_AVG = "vec_avg"
    COUNT = "count"
    EMB_MATRX = "emb_matrx"
    VECTORIZER_NAME_CHOICES = (
        (COUNT, "Count"),
        (TF_IDF, "TF-IDF"),
        (VEC_AVG, "Average word vector"),
        (EMB_MATRX, "Embedding matrix"),
    )

    EN = "en"
    HR = "hr"
    LANGUAGE_CHOICES = (
        (EN, "English"),
        (HR, "Croatian"),
    )

    WORDS = "words"
    CHARACTERS = "chars"
    TOKEN_TYPE_CHOICES = ((WORDS, "Words"), (CHARACTERS, "Characters"))

    name = models.CharField(max_length=100, unique=True, null=True, blank=True)
    description = models.CharField(max_length=500)
    guidelines = models.TextField(default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    managers = models.ManyToManyField(User, related_name="manager_in_projects")
    annotators = models.ManyToManyField(User, related_name="annotator_in_projects")
    members = models.ManyToManyField(User, related_name="member_in_projects")
    main_annotator = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="main_annotator_in_projects",
    )
    project_type = models.CharField(max_length=30, choices=PROJECT_CHOICES)
    al_mode = models.BooleanField(null=True, blank=True)
    al_method = models.CharField(max_length=30, choices=AL_METHOD_CHOICES, null=True)
    model_name = models.CharField(max_length=30, choices=MODEL_NAME_CHOICES, null=True)
    vectorizer_name = models.CharField(
        max_length=30, choices=VECTORIZER_NAME_CHOICES, null=True, blank=True
    )
    token_type = models.CharField(
        max_length=30, choices=TOKEN_TYPE_CHOICES, null=True, blank=True
    )
    min_ngram = models.PositiveIntegerField(
        null=True, blank=True, validators=[MinValueValidator(1)]
    )
    max_ngram = models.PositiveIntegerField(
        null=True, blank=True, validators=[MinValueValidator(1)]
    )
    adjustable_vocab = models.BooleanField(null=True, blank=True)
    vocab_max_size = models.PositiveIntegerField(null=True, blank=True)
    vocab_min_freq = models.PositiveIntegerField(
        null=True, blank=True, validators=[MinValueValidator(1)]
    )
    multilabel = models.BooleanField(null=True, blank=True)
    hierarchy = models.BooleanField(null=True, blank=True)
    language = models.CharField(max_length=10, null=True, choices=LANGUAGE_CHOICES)
    gl_enabled = models.BooleanField(default=False, null=True)  # Enable guided learning
    image_url = models.CharField(max_length=50, null=True, blank=True)
    access_code = models.CharField(max_length=32, null=True, blank=True)
    performance = PickledObjectField(null=True, blank=True, editable=True)

    # Virtual deletion
    objects = DeleteManager()  # Filter by is_deleted by default
    objects_with_deleted = DeleteManager(deleted=True)
    is_deleted = models.BooleanField(default=False, null=True)

    @property
    def image(self):
        return staticfiles_storage.url(self.image_url)

    @abstractmethod
    def get_template_name(self):
        raise NotImplementedError

    @abstractmethod
    def filter_docs(self, labeled=True):
        raise NotImplementedError

    @abstractmethod
    def get_document_class(self):
        raise NotImplementedError

    @abstractmethod
    def get_document_serializer(self):
        raise NotImplementedError

    @abstractmethod
    def get_annotation_class(self):
        raise NotImplementedError

    @abstractmethod
    def get_annotation_serializer(self):
        raise NotImplementedError

    @abstractmethod
    def get_csv_header(self, aggregation):
        raise NotImplementedError

    @abstractmethod
    def get_iaa_data(self, users, docs):
        raise NotImplementedError

    @abstractmethod
    def calculate_iaa(self, data, annotators):
        raise NotImplementedError

    @abstractmethod
    def get_aggregated_labels(self, docs):
        raise NotImplementedError

    @abstractmethod
    def related_annotations_name(self):
        raise NotImplementedError

    def is_type_of(self, project_type):
        return project_type == self.project_type

    def get_success_url(self):
        return reverse("upload", args=[self.id])

    def get_absolute_url(self):
        return reverse("upload", args=[self.id])

    def reset_completed_selection(self, user):
        docs = self.documents.filter(selectors=user, completed_by=user)
        user.selected_docs.remove(*docs)

    def get_progress(self, user):
        total_count = self.documents.all().count()
        unlabeled_count = self.documents.exclude(completed_by=user).count()

        total_batch = self.documents.filter(selectors=user)
        unlabeled_batch = total_batch.exclude(completed_by=user)

        return {
            "total": total_count,
            "remaining": unlabeled_count,
            "batch": total_batch.count(),
            "batch_remaining": unlabeled_batch.count(),
        }

    def get_distributed_documents(self, round=None):
        docs = self.documents.filter(is_selected=True)
        if round is None:
            return docs
        return docs.filter(round=round)

    def select_batch(self, **kwargs):
        _select_batch = (
            self._select_al_batch if self.al_mode else self._select_plain_batch
        )
        return _select_batch(**kwargs)

    def _select_al_batch(
        self,
        batch_size,
        test_proportion,
        random_train_proportion,
        model_anno_threshold,
        use_warm_start,
        **kwargs,
    ):
        selected_ids = []
        n_test = int(batch_size * test_proportion)

        # Retrieve unlabeled train and test splits.
        print("Retrieve unlabeled train and test splits.")
        unlab_train, unlab_test = get_unselected_split(self)
        unlab_train_ids = list(
            unlab_train.values_list("id", flat=True)
        )  # [doc.id for doc in unlab_train]
        unlab_test_ids = list(
            unlab_test.values_list("id", flat=True)
        )  # [doc.id for doc in unlab_test]
        print(f"Unlabeled train size: {len(unlab_train_ids)}")
        print(f"Unlabeled test size: {len(unlab_test_ids)}")

        # Retrieve labeled train and test splits.
        print("Retrieve labeled train and test splits.")
        start = time()
        lab_train, lab_test = get_labeled_split(
            self,
            anno_lower_bound=model_anno_threshold,
            use_warm_start=use_warm_start,
        )
        end = time()
        print(f"Get labeled split time: {end - start}")
        # New test documents are retrieved randomly
        # from the predetermined test set.
        print("Retrieve test documents")
        n_test = min(n_test, len(unlab_test_ids))
        print(f"Num test: {n_test}")
        new_test_ids = sample(unlab_test_ids, k=n_test)
        selected_ids.extend(new_test_ids)
        print("New test ids:", new_test_ids)

        # The first part of the new train documents is retrieved
        # randomly from the predetermined train set.
        n_train = batch_size - n_test
        n_train = min(n_train, len(unlab_train_ids))
        n_train_random = int(random_train_proportion * n_train)
        n_train_random = min(n_train_random, n_train)

        # TODO: handle binary, multiclass and multilabel boundary conditions
        al_condition = lab_train  # and self._label_condition(train_labels)
        if not al_condition:
            n_train_random = n_train
        random_train_ids = sample(unlab_train_ids, k=n_train_random)
        selected_ids.extend(random_train_ids)
        print("Random train ids:", random_train_ids)

        n_train_al = n_train - n_train_random

        # The second part of the new train documents is retrieved
        # via AL acquisition function.

        # AL start
        # ====================================================================
        if n_train_al > 0:
            print("AL started...")
            # Step 1: train set preprocessing
            if self.model_name not in TRANSFORMER_MODELS:
                vocab = (
                    VectorizedVocab(
                        max_size=self.vocab_max_size, min_freq=self.vocab_min_freq
                    )
                    if self.adjustable_vocab
                    else VectorizedVocab()
                )

            TEXT_FIELD_DATA = {"name": "text"}
            if self.model_name in TRANSFORMER_MODELS:
                TEXT_FIELD_DATA["tokenizer"] = None
                TEXT_FIELD_DATA["disable_batch_matrix"] = True
            else:
                TEXT_FIELD_DATA["disable_numericalize_caching"] = True
                TEXT_FIELD_DATA["tokenizer"] = get_tokenizer(self.language)
                TEXT_FIELD_DATA["numericalizer"] = vocab
                TEXT_FIELD_DATA["posttokenize_hooks"] = get_posttokenize_hooks(
                    self.language,
                    self.token_type,
                    self.vectorizer_name,
                    self.min_ngram,
                    self.max_ngram,
                )
            TEXT = Field(**TEXT_FIELD_DATA)

            label_field_cls = MultilabelField if self.multilabel else LabelField
            LABEL = label_field_cls(name="label")
            fields = {"text": TEXT, "label": LABEL}

            labels = self.get_aggregated_labels(lab_train + lab_test)
            train_labels, test_labels = (
                labels[: len(lab_train)],
                labels[len(lab_train) :],
            )

            train_set = make_labeled_dataset(lab_train, train_labels, fields)
            train_set.finalize_fields()

            test_set = make_labeled_dataset(lab_test, test_labels, fields)

            # Step 2: train set up vectorizer
            if self.model_name not in TRANSFORMER_MODELS:
                VECTORIZER_DATA = {"vocab": vocab}
                if (
                    self.vectorizer_name == Project.VEC_AVG
                    or self.vectorizer_name == Project.EMB_MATRX
                ):
                    VECTORIZER_DATA["lang"] = self.language
                vectorizer = get_vectorizer(self.vectorizer_name)(**VECTORIZER_DATA)
                vectorizer.fit(train_set, field=TEXT)
                vocab.set_vectorizer(vectorizer)
                TEXT._disable_numericalize_caching = False

            # Step 3: set up train manager
            if self.model_name in TORCH_MODELS:
                criterion = (
                    nn.BCEWithLogitsLoss()
                    if len(self.labels.all()) == 2 or self.multilabel
                    else nn.CrossEntropyLoss()
                )
                train_manager = TorchTM(
                    (train_set, test_set),
                    self.multilabel,
                    criterion,
                )
            else:
                train_manager = SklearnTM((train_set, test_set), self.multilabel)

            # Step 4: fit model
            params = {}
            if self.model_name == Project.MLP:
                params["input_dim"] = (
                    300 if self.vectorizer_name == Project.VEC_AVG else len(vocab.stoi)
                )
                params["output_dim"] = (
                    1
                    if len(self.labels.all()) == 2 and not self.multilabel
                    else len(self.labels.all())
                )
                params["device"] = train_manager.device
            elif self.model_name in TRANSFORMER_MODELS:
                params["output_dim"] = (
                    1
                    if len(self.labels.all()) == 2 and not self.multilabel
                    else len(self.labels.all())
                )
                params["bert_model_name"] = get_bert_model_name(self.language)
                params["device"] = train_manager.device
            elif self.model_name in RECURSIVE_MODELS:
                params["rnn_type"] = self.model_name.upper()
                params["output_dim"] = (
                    1
                    if len(self.labels.all()) == 2 and not self.multilabel
                    else len(self.labels.all())
                )
                params["pretrained_embeddings"] = vectorizer.embeddings
                params["device"] = train_manager.device

            model = get_model(
                self.model_name, params=params, multilabel=self.multilabel
            )
            model.al_step(train_manager)

            # Step 5: test metrics

            # Step 5.1: calculate test metrics
            measure = "f1_samples" if self.multilabel else "f1_micro"
            metric = metrics.calculate_metrics(
                model,
                train_manager.X_train,
                train_manager.y_train,
                train_manager.X_test,
                train_manager.y_test,
                measure=measure,
            )

            # Step 5.2: save calculated metrics
            if not self.performance:
                performance = dict()
                performance["train"] = {
                    "metric": [],
                    "count": [],
                }
                performance["test"] = {
                    "metric": [],
                    "count": [],
                    "ub": [],
                    "lb": [],
                    "bootstrap": [],
                }
                self.performance = performance
            performance = self.performance
            performance["train"]["metric"].append(metric["train"])
            performance["train"]["count"].append(metric["labeled_train"])
            if metric["test"] is not None:
                performance["test"]["metric"].append(metric["test"])
                performance["test"]["count"].append(metric["labeled_train"])
            if metric["conf"] is not None:
                performance["test"]["ub"].append(metric["conf"]["upper"])
                performance["test"]["lb"].append(metric["conf"]["lower"])
                performance["test"]["bootstrap"].append(metric["conf"]["bootstrap"])
            self.performance = performance
            self.save()

            # Step 6: vectorize unlabeled data
            unlab_train_updated = unlab_train.exclude(id__in=random_train_ids)
            unlabeled_set = make_dataset(unlab_train_updated, {"text": TEXT})
            if self.model_name in TORCH_MODELS and self.model_name != "mlp":
                train_inputs, *_ = unlabeled_set.batch(add_padding=True)
            else:
                train_inputs, *_ = unlabeled_set.batch()
            X_unlab = np.array(train_inputs)

            # Step 7: use AL strategy to retreive a sample
            # These inputs will be provided to various different samplers,
            # depending on their respective method signatures.
            al_kwargs = dict(
                X_unlab=X_unlab,
                X_lab=train_manager.X_train,
                model=model,
                batch_size=n_train_al,
                multilabel=self.multilabel,
                iter_lab=train_manager.get_data(shuffle=False)
                if isinstance(train_manager, TorchTM)
                else None,
                iter_unlab=Iterator(
                    unlabeled_set,
                    batch_size=train_manager.batch_size,
                    matrix_class=train_manager.device_tensor,
                    shuffle=False,
                )
                if isinstance(train_manager, TorchTM)
                else None,
                device=train_manager.device
                if isinstance(train_manager, TorchTM)
                else None,
                criterion=train_manager.criterion
                if isinstance(train_manager, TorchTM)
                else None,
            )
            # TODO: discuss seed and randomness (currently seed is not supported)
            al_strategy = get_al_sampler(self.al_method)()
            al_indices = al_strategy.select_batch(**al_kwargs).tolist()
            al_ids = [unlab_train_updated[index].id for index in al_indices]
            selected_ids.extend(al_ids)
            print("AL ids:", al_ids)

            # Step 8: update documents (set is_al flag)
            unlab_train_updated.update(is_al=True)
        print("AL finished...")
        # AL end
        # ====================================================================

        # Update is_selected flag on selected documents.
        self.documents.filter(pk__in=selected_ids).update(is_selected=True)
        return selected_ids

    def _select_plain_batch(self, batch_size, **kwargs):
        available_docs = self.documents.filter(is_selected=False, gl_annotated=False)
        available_ids = [doc.id for doc in available_docs]
        k = min(len(available_ids), batch_size)
        selected_ids = sample(available_ids, k=k)

        # Update is_selected flag on selected documents.
        self.documents.filter(pk__in=selected_ids).update(is_selected=True)
        return selected_ids

    # def upload_data(self, docs):
    #     if self.al_mode:
    #         self._al_upload(docs)
    #     else:
    #         self._plain_upload(docs)

    # def _plain_upload(self, docs):
    #     return

    # def _al_upload(self, docs):
    #     # TODO: set this as an upload form parameter (AL projects only)
    #     test_prop = 0.2
    #     count = len(docs)
    #     test_count = int(test_prop * count)
    #     test_docs = sample(docs, k=test_count)
    #     test_ids = [doc.id for doc in test_docs]
    #     self.documents.filter(pk__in=test_ids).update(is_test=True)

    def get_model_performance(self):
        return self.performance

    def __str__(self):
        return self.name

    def get_all_users(self):
        q1 = self.managers.all()
        q2 = self.annotators.all()
        r = (q1 | q2).distinct()
        return r

    def get_completed_docs(self):
        return None

    @staticmethod
    def get_random_img_url():
        # TODO: make this more generic.
        url = f"images/projects/img-{random.randint(0, 9)}.jpg"
        return url

    @staticmethod
    def generate_access_code():
        length = 32
        return get_random_string(length)


class ClassificationProject(Project):
    def get_template_name(self):
        if self.hierarchy:
            return "annotation/hier_classification.html"
        else:
            return "annotation/document_classification.html"

    def filter_docs(self, labeled=True):
        is_null = not labeled
        return self.documents.filter(doc_annotations__isnull=is_null)

    def get_document_class(self):
        from .classification_document import ClassificationDocument

        return ClassificationDocument

    def get_document_serializer(self):
        from server.serializers import ClassificationDocumentSerializer

        return ClassificationDocumentSerializer

    def get_annotation_class(self):
        from .document_annotation import DocumentAnnotation

        return DocumentAnnotation

    def get_annotation_serializer(self):
        from server.serializers import DocumentAnnotationSerializer

        return DocumentAnnotationSerializer

    def related_annotations_name(self):
        return "doc_annotations"

    def get_csv_header(self, aggregation):
        if aggregation:
            header = ["document_id", "text"]
            labels = sorted([label.text for label in self.labels.all()])
            header += labels + [
                "aggregated_label",
                "num_labels",
                "num_annotators",
                "round",
            ]
            if self.multilabel:
                header += ["MASI_similarity"]
            header += ["AL", "GL"]
            return header
        else:
            return ["document_id", "text", "label", "annotator", "GL"]

    def get_aggregated_labels(self, docs):
        if self.multilabel:
            # TODO: multilabel aggregation methods
            aggregated_labels = []
            for doc in docs:
                gold = doc.get_gold_label()
                if gold:
                    aggregated_labels.append(gold)
                    continue
                doc_labs = []
                annos = doc.get_all_annotations()
                users = {
                    anno.user
                    for anno in annos
                    if doc.completed_by.filter(id=anno.user.id).exists()
                }
                num_users = len(users)
                if num_users == 0 or not self.documents.filter(id=doc.id).exists():
                    aggregated_labels.append(None)
                else:
                    thresh = num_users // 2
                    anno_labs = [anno.label.text for anno in annos]
                    majority_counter = Counter(anno_labs)
                    for label, count in majority_counter.items():
                        if count > thresh:
                            doc_labs.append(label)
                    aggregated_labels.append(sorted(doc_labs))
        else:
            # Empirical Analysis of Aggregation Methods for Collective Annotation
            # Ciyang Qing, Ulle Endriss, Raquel Fernandez and Justin Kruger

            # Make dataset
            data = []
            all_docs = self.documents.annotate(num_anno=Count("completed_by")).filter(
                gl_annotated=False, is_warm_start=False, num_anno__gt=0
            )
            for doc in all_docs:
                doc_annos = doc.get_annotations()
                for doc_anno in doc_annos:
                    completed = doc.completed_by.filter(id__exact=doc_anno.user.id)
                    if completed.exists():
                        data.append([doc_anno.user.id, doc.id, doc_anno.label.text])
            data = pd.DataFrame(data=data, columns=["anno", "doc", "lab"])
            K = self.labels.count()
            # Get plural label for each document (simple plurality rule)
            SPR = {}
            for doc in all_docs:
                df_doc = data.loc[data["doc"] == doc.id]
                counts = df_doc["lab"].value_counts()
                maximum = counts.max()
                SPR[doc.id] = list(counts[counts == maximum].index)
            data["plural"] = data["doc"].map(SPR)

            # Get annotator weights
            anno_weights = Counter()
            for _, row in data.iterrows():
                if row["lab"] in row["plural"]:
                    anno_weights[row["anno"]] += 1
            for anno in anno_weights.keys():
                agr = (anno_weights[anno] + 0.5) / (
                    len(data.loc[data["anno"] == anno]) + 1
                )
                anno_weights[anno] = np.log(((K - 1) * agr) / (1 - agr))

            aggregated_labels = []
            for doc in docs:
                gold = doc.get_gold_label()
                if gold:
                    aggregated_labels.append(gold[0])
                    continue
                df_doc = data.loc[data["doc"] == doc.id]
                if df_doc.empty:
                    aggregated_labels.append(None)
                else:
                    label_weights = Counter()
                    for _, row in df_doc.iterrows():
                        label_weights[row["lab"]] += anno_weights[row["anno"]]
                    aggregated_labels.append(max(label_weights, key=label_weights.get))

        return aggregated_labels

    def get_iaa_data(self, users, docs):
        labels = [label.text for label in self.labels.all()]
        label_map = dict(zip(labels, range(len(labels))))
        user_map = dict(zip(users, range(users.count())))

        annotations = []
        for _ in range(users.count()):
            annotations.append([])

        max_annotations = 0

        for doc in docs:
            doc_annos = doc.get_annotations()
            count_annotations = 0
            for user in users:
                if self.multilabel:
                    anno_object = doc_annos.filter(user__id__exact=user.id)
                else:
                    anno_object = doc_annos.filter(user__id__exact=user.id).first()
                completed = doc.completed_by.filter(id__exact=user.id)
                if anno_object and completed:
                    user_lab = (
                        frozenset([anno.label.text for anno in anno_object])
                        if self.multilabel
                        else label_map[anno_object.label.text]
                    )
                    count_annotations += 1
                else:
                    user_lab = None
                user_list = annotations[user_map[user]]
                user_list.append(user_lab)
                max_annotations = (
                    count_annotations
                    if count_annotations > max_annotations
                    else max_annotations
                )

        data = {"annotations": annotations, "max_annotations": max_annotations}
        return data

    def calculate_iaa(self, data, annotators):
        k = len(annotators)
        max_annotations = data["max_annotations"]
        data = data["annotations"]
        arr = np.array(data)
        if self.multilabel:
            keep_column = arr != None
            keep_column = keep_column.sum(axis=0)
            keep_column = keep_column >= 2
            arr = arr[:, keep_column]
            # Pairwise
            pairwise_scores = {}
            for i, j in combinations(range(k), 2):
                ann1 = arr[i]
                ann2 = arr[j]
                mask1 = ann1 != None
                mask2 = ann2 != None
                joint_mask = mask1 & mask2
                if (joint_mask == False).all():
                    continue
                pair_data = np.vstack((ann1[joint_mask], ann2[joint_mask]))
                pair_data_alpha = []
                for anno, doc in np.ndindex(pair_data.shape):
                    pair_data_alpha.append((anno, doc, pair_data[anno, doc]))
                task = AnnotationTask(distance=masi_distance)
                task.load_array(pair_data_alpha)
                alpha_score = task.alpha()
                pairwise_scores[(i, j)] = round(alpha_score, 4)
            # Joint
            data_alpha = []
            for anno, doc in np.ndindex(arr.shape):
                if arr[anno, doc] is not None:
                    data_alpha.append((anno, doc, arr[anno, doc]))
            task = AnnotationTask(distance=masi_distance)
            task.load_array(data_alpha)
            try:
                masi_alpha = round(task.alpha(), 4)
            except Exception:
                masi_alpha = None
            joint_score = {"score": masi_alpha, "used_docs": None}

        else:
            cohen_scores = {}
            # Cohen's kappa
            for pair in combinations(range(k), 2):
                i, j = pair
                ann1 = arr[i]
                ann2 = arr[j]
                mask1 = ann1 != None
                mask2 = ann2 != None
                joint_mask = mask1 & mask2
                if (joint_mask == False).all():
                    continue

                ann1_filt = ann1[joint_mask].astype(np.int)
                ann2_filt = ann2[joint_mask].astype(np.int)

                ck_score = cohen_kappa_score(ann1_filt, ann2_filt)
                if np.isnan(ck_score):
                    cohen_scores[pair] = 1.0
                else:
                    cohen_scores[pair] = round(ck_score, 4)
            pairwise_scores = cohen_scores
            # Fleiss' kappa
            fleiss_data = []
            count_used_docs = 0
            for doc in range(arr.shape[1]):
                count_anno = 0
                category_count = [0 for _ in range(len(self.labels.all()))]
                for anno in range(k):
                    if data[anno][doc] is not None:
                        category_count[data[anno][doc]] += 1
                        count_anno += 1
                if count_anno == max_annotations:
                    fleiss_data.append(category_count)
                    count_used_docs += 1
            try:
                fleiss_score = round(fleiss_kappa(fleiss_data), 4)
                if np.isnan(fleiss_score):
                    fleiss_score = 1.0
            except Exception:
                fleiss_score = None
            joint_score = {"score": fleiss_score, "used_docs": count_used_docs}

        return pairwise_scores, joint_score


class SeqLabelingProject(Project):
    def get_template_name(self):
        return "annotation/sequence_labeling.html"

    def filter_docs(self, labeled=True):
        is_null = not labeled
        return self.documents.filter(seq_annotations__isnull=is_null)

    def get_document_class(self):
        from .sequence_document import SequenceDocument

        return SequenceDocument

    def get_document_serializer(self):
        from server.serializers import SequenceDocumentSerializer

        return SequenceDocumentSerializer

    def get_annotation_class(self):
        from .sequence_annotation import SequenceAnnotation

        return SequenceAnnotation

    def get_annotation_serializer(self):
        from server.serializers import SequenceAnnotationSerializer

        return SequenceAnnotationSerializer

    def related_annotations_name(self):
        return "seq_annotations"

    def get_csv_header(self, *args, **kwargs):
        return ["document_id", "text", "label", "start", "end", "annotator", "GL"]

    def get_iaa_data(self, users, docs):
        print("tu sam")
        return None

    def calculate_iaa(self, data, annotators):
        return None

    def get_aggregated_labels(self, docs):
        raise NotImplementedError


class ProjectFactory:
    @staticmethod
    def create_subclass(project):
        project_type = project.project_type
        factory_class = eval(Project.TYPE_MAPPING[project_type])

        new_project = factory_class(
            name=project.name,
            description=project.description,
            guidelines=project.guidelines,
            created_at=project.created_at,
            updated_at=project.updated_at,
            project_type=project.project_type,
            al_mode=project.al_mode,
            language=project.language,
            al_method=project.al_method,
            model_name=project.model_name,
            multilabel=project.multilabel,
            hierarchy=project.hierarchy,
            image_url=Project.get_random_img_url(),
            access_code=Project.generate_access_code(),
            vectorizer_name=project.vectorizer_name,
            adjustable_vocab=project.adjustable_vocab,
            vocab_max_size=project.vocab_max_size,
            vocab_min_freq=project.vocab_min_freq,
            token_type=project.token_type,
            min_ngram=project.min_ngram,
            max_ngram=project.max_ngram,
        )
        return new_project
