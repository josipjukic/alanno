from django.test import TestCase
from server.models import (
    Project,
    Label,
    ClassificationDocument,
    DocumentAnnotation,
    ClassificationProject,
    Round,
)
from django.contrib.auth.models import User
from celery_progress.backend import ProgressRecorder
from server.services.export_dataset import *
import numpy as np


class MockProgressRecorder(ProgressRecorder):
    def __init__(self, mock_task=None):
        super().__init__(mock_task)

    def set_progress(self, current, total, description=""):
        print(f"\r{description} {current}/{total}          ", end="")


class ClassificationAggregateSingleLabel(TestCase):
    def setUp(self):
        """
        Created dataset:

            | anno1 | anno2 | anno3 | anno4
             ------------------------------
        doc1|  no   |  yes  |  yes  |  yes
        doc2|  yes  |  no   |  yes  |  yes
        doc3|  yes  |  yes  |  no   |  yes
        doc4|  yes  |  yes  |  no   |  no

        The only annotator who has always annotated with majority label is annotator 4,
        so he should have the biggest weight.
        Document 4 is used for testing tie-breakers and we expect label of annotator 4 to win.
        """
        objects = []
        p = ClassificationProject(
            name="test",
            description="test",
            project_type=Project.DOCUMENT_CLASSIFICATION,
            multilabel=False,
        )
        p.save()
        objects.append(p)
        u1 = User(username="user1")
        u2 = User(username="user2")
        u3 = User(username="user3")
        u4 = User(username="user4")
        for u in [u1, u2, u3, u4]:
            u.save()
        p.annotators.add(u1, u2, u3, u4)
        objects.extend([u1, u2, u3, u4])
        l1 = Label(text="yes", project=p)
        l2 = Label(text="no", project=p)
        objects.extend([l1, l2])
        d1 = ClassificationDocument(text="doc1", raw_html="doc1", project=p)
        d2 = ClassificationDocument(text="doc2", raw_html="doc2", project=p)
        d3 = ClassificationDocument(text="doc3", raw_html="doc3", project=p)
        d1.save()
        d1.selectors.add(u1, u2, u3, u4)
        d1.completed_by.add(u1, u2, u3, u4)
        d2.save()
        d2.selectors.add(u1, u2, u3, u4)
        d2.completed_by.add(u1, u2, u3, u4)
        d3.save()
        d3.selectors.add(u1, u2, u3, u4)
        d3.completed_by.add(u1, u2, u3, u4)
        for i, user in enumerate([u1, u2, u3, u4]):
            for j, doc in enumerate([d1, d2, d3]):
                if i == j:
                    a = DocumentAnnotation(user=user, document=doc, label=l2)
                else:
                    a = DocumentAnnotation(user=user, document=doc, label=l1)
                objects.append(a)
        d4 = ClassificationDocument(text="doc4", raw_html="doc4", project=p)
        d4.save()
        d4.selectors.add(u1, u2, u3, u4)
        d4.completed_by.add(u1, u2, u3, u4)
        a1 = DocumentAnnotation(user=u1, document=d4, label=l1)
        objects.append(a1)
        a2 = DocumentAnnotation(user=u2, document=d4, label=l1)
        objects.append(a2)
        a3 = DocumentAnnotation(user=u3, document=d4, label=l2)
        objects.append(a3)
        a4 = DocumentAnnotation(user=u4, document=d4, label=l2)
        objects.append(a4)
        for obj in objects:
            obj.save()

    def test_tieBreak_allDocuments(self):
        p = ClassificationProject.objects.all().first()

        expected_labels = ["yes", "yes", "yes", "no"]
        aggregated_labels = p.get_aggregated_labels(
            list(ClassificationDocument.objects.all().order_by("id"))
        )
        self.assertEqual(expected_labels, aggregated_labels)

    def test_tieBreak_twoDocuments(self):
        """
        Testing if annotator weights are calculated over all documents (expected) or just input documents.
        For that we are using only documents 3 and 4.
        In document 3 all annotator annotated with majority label except annotator 3.
        Document 4 is tie-breaker. Annotators in agreement are: 1 and 2, 3 and 4.

        If annotator weights were to be calculated over input documents only,
        it would be expected that the chosen label would be the label from annotators 1 and 2 (label yes),
        because annotators 1, 2 and 4 would have the same weights, while annotator 3 would have smaller weight.
        """
        p = ClassificationProject.objects.all().first()

        expected_labels = ["yes", "no"]
        aggregated_labels = p.get_aggregated_labels(
            list(ClassificationDocument.objects.all().order_by("id"))[-2:]
        )
        self.assertEqual(expected_labels, aggregated_labels)

    def test_noDocuments(self):
        p = ClassificationProject.objects.all().first()
        aggregated_labels = p.get_aggregated_labels([])
        self.assertEqual([], aggregated_labels)

    def test_documentNotInProject(self):
        p1 = ClassificationProject.objects.all().first()
        users = list(User.objects.all().order_by("id"))
        l1 = Label.objects.get(text="yes")
        l2 = Label.objects.get(text="no")
        p2 = ClassificationProject(
            name="test2",
            description="test",
            project_type=Project.DOCUMENT_CLASSIFICATION,
            multilabel=False,
        )
        p2.save()
        d5 = ClassificationDocument(text="doc5", raw_html="doc5", project=p2)
        d5.save()
        d5.selectors.add(*users)
        d5.completed_by.add(*users)
        d6 = ClassificationDocument(text="doc6", raw_html="doc6", project=p2)
        d6.save()
        d6.selectors.add(*users)
        d6.completed_by.add(*users)
        for user in users:
            for doc in [d5, d6]:
                if user.username == "user4":
                    a = DocumentAnnotation(user=user, document=doc, label=l2)
                else:
                    a = DocumentAnnotation(user=user, document=doc, label=l1)
                a.save()

        expected_labels = ["yes", "yes", "yes", "no", None, None]
        aggregated_labels = p1.get_aggregated_labels(
            list(ClassificationDocument.objects.all().order_by("id"))
        )
        self.assertEqual(expected_labels, aggregated_labels)

        expected_labels = ["yes", "yes", "yes", "no"]
        doc_list = sorted(list(p1.documents.all()), key=lambda x: x.id)
        aggregated_labels = p1.get_aggregated_labels(doc_list)
        self.assertEqual(expected_labels, aggregated_labels)

    def test_usingOnlyCompleted(self):
        p = ClassificationProject.objects.all().first()
        users = list(User.objects.all().order_by("id"))
        l1 = Label.objects.get(text="yes")
        l2 = Label.objects.get(text="no")
        d5 = ClassificationDocument(text="doc5", raw_html="doc5", project=p)
        d5.save()
        d5.selectors.add(*users)
        d6 = ClassificationDocument(text="doc6", raw_html="doc6", project=p)
        d6.save()
        d6.selectors.add(*users)
        for user in users:
            for doc in [d5, d6]:
                if user.username == "user4":
                    a = DocumentAnnotation(user=user, document=doc, label=l2)
                else:
                    a = DocumentAnnotation(user=user, document=doc, label=l1)
                a.save()

        expected_labels = ["yes", "yes", "yes", "no", None, None]
        aggregated_labels = p.get_aggregated_labels(
            list(ClassificationDocument.objects.all().order_by("id"))
        )
        self.assertEqual(expected_labels, aggregated_labels)

        expected_labels = ["yes", "yes", "yes", "no"]
        doc_list = list(ClassificationDocument.objects.all().order_by("id"))[:-2]
        aggregated_labels = p.get_aggregated_labels(doc_list)
        self.assertEqual(expected_labels, aggregated_labels)

    def test_goldLabeledDocument(self):
        p = ClassificationProject.objects.all().first()
        l1 = Label.objects.get(text="yes")
        d5 = ClassificationDocument(text="doc5", raw_html="doc5", project=p)
        d5.save()
        a = DocumentAnnotation(user=None, document=d5, label=l1)
        a.save()
        expected_labels = ["yes", "yes", "yes", "no", "yes"]
        aggregated_labels = p.get_aggregated_labels(
            list(ClassificationDocument.objects.all().order_by("id"))
        )
        self.assertEqual(expected_labels, aggregated_labels)


class ClassificationAggregateMultiLabel(TestCase):
    # TODO: Write tests for multi-label aggregation when we implement it
    pass

    def setUp(self):
        pass

    def test_tieBreak_allDocuments(self):
        pass

    def test_tieBreak_twoDocuments(self):
        """
        Testing if annotator weights are calculated over all documents (expected) or just input documents.
        Could not be needed. Depends on the implementation.
        """
        pass

    def test_noDocuments(self):
        pass

    def test_documentNotInProject(self):
        pass

    def test_usingOnlyCompleted(self):
        pass


class ClassificationExport(TestCase):
    """
    p1 -> single-label project
    p2 -> multi-label project

    Notation: Ax_y
        - A -> object type
        - x -> object number
        - y -> project number

    u1 -> annotator
    u2 -> annotator
    u3 -> annotator who has not completed his annotations

    d1_x -> document
    d2_x -> document that has only be annotated by one annotator and has document_id
    d3_x -> document with no annotations

    Created dataset:

              |  anno1  |  anno2  |  anno3
               ------------------------------
        doc1_1|    1    |    1    |    2
        doc2_1|    2    |    /    |    /
        doc3_1|    /    |    /    |    /


              |  anno1  |  anno2  |  anno3
               ------------------------------
        doc1_2|   1,2   |   1,2   |    2
        doc2_2|    2    |    /    |    /
        doc3_2|    /    |    /    |    /
    """

    def setUp(self):
        self.mpr = MockProgressRecorder()
        self.p1 = ClassificationProject(
            name="test1",
            description="test",
            project_type=Project.DOCUMENT_CLASSIFICATION,
            multilabel=False,
        )
        self.p1.save()
        self.p2 = ClassificationProject(
            name="test2",
            description="test",
            project_type=Project.DOCUMENT_CLASSIFICATION,
            multilabel=True,
        )
        self.p2.save()

        self.u1 = User(username="user1")
        self.u2 = User(username="user2")
        self.u3 = User(username="user3")
        self.u1.save()
        self.u2.save()
        self.u3.save()
        self.p1.annotators.add(self.u1, self.u2, self.u3)
        self.p2.annotators.add(self.u1, self.u2, self.u3)

        self.r1 = Round(number=0, project=self.p1)
        self.r2 = Round(number=1, project=self.p2)
        self.r1.save()
        self.r2.save()

        self.d1_1 = ClassificationDocument(
            text="doc1", raw_html="doc1", project=self.p1, round=self.r1
        )
        self.d2_1 = ClassificationDocument(
            text="doc2",
            raw_html="doc2",
            project=self.p1,
            document_id=1234,
            round=self.r1,
        )
        self.d3_1 = ClassificationDocument(
            text="doc3", raw_html="doc3", project=self.p1, round=self.r1
        )

        self.d1_2 = ClassificationDocument(
            text="doc1", raw_html="doc1", project=self.p2, round=self.r2
        )
        self.d2_2 = ClassificationDocument(
            text="doc2",
            raw_html="doc2",
            project=self.p2,
            document_id=1234,
            round=self.r2,
        )
        self.d3_2 = ClassificationDocument(
            text="doc3", raw_html="doc3", project=self.p2, round=self.r2
        )
        for d in [self.d1_1, self.d2_1, self.d1_2, self.d2_2]:
            d.save()
            d.selectors.add(self.u1, self.u2, self.u3)
            d.completed_by.add(self.u1, self.u2)
        self.d3_1.save()
        self.d3_2.save()
        self.d3_1.selectors.add(self.u1, self.u2, self.u3)
        self.d3_2.selectors.add(self.u1, self.u2, self.u3)

        self.l1_1 = Label(text="lab1", project=self.p1)
        self.l2_1 = Label(text="lab2", project=self.p1)
        self.l1_2 = Label(text="lab1", project=self.p2)
        self.l2_2 = Label(text="lab2", project=self.p2)
        for l in [self.l1_1, self.l2_1, self.l1_2, self.l2_2]:
            l.save()

        self.a1_1 = DocumentAnnotation(
            user=self.u1, document=self.d1_1, label=self.l1_1
        )
        self.a1_1.save()
        self.a2_1 = DocumentAnnotation(
            user=self.u2, document=self.d1_1, label=self.l1_1
        )
        self.a2_1.save()
        self.a3_1 = DocumentAnnotation(
            user=self.u3, document=self.d1_1, label=self.l2_1
        )
        self.a3_1.save()
        self.a4_1 = DocumentAnnotation(
            user=self.u1, document=self.d2_1, label=self.l2_1
        )
        self.a4_1.save()

        self.a1_2 = DocumentAnnotation(
            user=self.u1, document=self.d1_2, label=self.l1_2
        )
        self.a1_2.save()
        self.a2_2 = DocumentAnnotation(
            user=self.u1, document=self.d1_2, label=self.l2_2
        )
        self.a2_2.save()
        self.a3_2 = DocumentAnnotation(
            user=self.u2, document=self.d1_2, label=self.l1_2
        )
        self.a3_2.save()
        self.a4_2 = DocumentAnnotation(
            user=self.u2, document=self.d1_2, label=self.l2_2
        )
        self.a4_2.save()
        self.a5_2 = DocumentAnnotation(
            user=self.u3, document=self.d1_2, label=self.l2_2
        )
        self.a5_2.save()
        self.a6_2 = DocumentAnnotation(
            user=self.u1, document=self.d2_2, label=self.l2_2
        )
        self.a6_2.save()

    # CSV and single-label
    def test_csv_singleLabel_isAggregated_notUnlabeled(self):
        p = self.p1
        aggregation = True
        unlabeled = False
        docs = [self.d1_1, self.d2_1]

        expected_header = [
            "document_id",
            "text",
            "lab1",
            "lab2",
            "aggregated_label",
            "num_labels",
            "num_annotators",
            "round",
            "AL",
            "GL",
        ]
        self.assertEqual(expected_header, p.get_csv_header(aggregation))

        expected_representation1 = [
            [self.d1_1.id, "doc1", 1.0, 0, 2, 2, 1, False, False],
            [1234, "doc2", 0, 1.0, 1, 1, 1, False, False],
        ]
        doc_csv_representation1 = representation_csv(
            docs, aggregation, unlabeled, self.mpr, 1
        )
        self.assertEqual(expected_representation1, doc_csv_representation1)

        expected_representation2 = [
            [self.d1_1.id, "doc1", 1.0, 0, "lab1", 2, 2, 1, False, False],
            [1234, "doc2", 0, 1.0, "lab2", 1, 1, 1, False, False],
        ]
        doc_csv_representation2 = label_aggregation_csv(
            p, docs, doc_csv_representation1, self.mpr, 1
        )
        self.assertEqual(expected_representation2, doc_csv_representation2)

    def test_csv_singleLabel_isAggregated_isUnlabeled(self):
        p = self.p1
        aggregation = True
        unlabeled = True
        docs = [self.d1_1, self.d2_1, self.d3_1]

        expected_header = [
            "document_id",
            "text",
            "lab1",
            "lab2",
            "aggregated_label",
            "num_labels",
            "num_annotators",
            "round",
            "AL",
            "GL",
        ]
        self.assertEqual(expected_header, p.get_csv_header(aggregation))

        expected_representation1 = [
            [self.d1_1.id, "doc1", 1.0, 0, 2, 2, 1, False, False],
            [1234, "doc2", 0, 1.0, 1, 1, 1, False, False],
            [self.d3_1.id, "doc3", "", "", "", "", "", False, False],
        ]
        doc_csv_representation1 = representation_csv(
            docs, aggregation, unlabeled, self.mpr, 1
        )
        self.assertEqual(expected_representation1, doc_csv_representation1)

        expected_representation2 = [
            [self.d1_1.id, "doc1", 1.0, 0, "lab1", 2, 2, 1, False, False],
            [1234, "doc2", 0, 1.0, "lab2", 1, 1, 1, False, False],
            [self.d3_1.id, "doc3", "", "", "", "", "", "", False, False],
        ]
        doc_csv_representation2 = label_aggregation_csv(
            p, docs, doc_csv_representation1, self.mpr, 1
        )
        self.assertEqual(expected_representation2, doc_csv_representation2)

    def test_csv_singleLabel_notAggregated_notUnlabeled(self):
        p = self.p1
        aggregation = False
        unlabeled = False
        docs = [self.d1_1, self.d2_1]

        expected_header = ["document_id", "text", "label", "annotator", "GL"]
        self.assertEqual(expected_header, p.get_csv_header(aggregation))

        expected_representation1 = [
            [
                [self.d1_1.id, "doc1", "lab1", "user1", False],
                [self.d1_1.id, "doc1", "lab1", "user2", False],
            ],
            [[1234, "doc2", "lab2", "user1", False]],
        ]
        doc_csv_representation1 = representation_csv(
            docs, aggregation, unlabeled, self.mpr, 1
        )
        self.assertEqual(expected_representation1, doc_csv_representation1)

        expected_representation2 = [
            [self.d1_1.id, "doc1", "lab1", "user1", False],
            [self.d1_1.id, "doc1", "lab1", "user2", False],
            [1234, "doc2", "lab2", "user1", False],
        ]
        doc_csv_representation2 = unpack_entries_csv(
            doc_csv_representation1, self.mpr, 1
        )
        self.assertEqual(expected_representation2, doc_csv_representation2)

    def test_csv_singleLabel_notAggregated_isUnlabeled(self):
        p = self.p1
        aggregation = False
        unlabeled = True
        docs = [self.d1_1, self.d2_1, self.d3_1]

        expected_header = ["document_id", "text", "label", "annotator", "GL"]
        self.assertEqual(expected_header, p.get_csv_header(aggregation))

        expected_representation1 = [
            [
                [self.d1_1.id, "doc1", "lab1", "user1", False],
                [self.d1_1.id, "doc1", "lab1", "user2", False],
            ],
            [[1234, "doc2", "lab2", "user1", False]],
            [[self.d3_1.id, "doc3", "", "", ""]],
        ]
        doc_csv_representation1 = representation_csv(
            docs, aggregation, unlabeled, self.mpr, 1
        )
        self.assertEqual(expected_representation1, doc_csv_representation1)

        expected_representation2 = [
            [self.d1_1.id, "doc1", "lab1", "user1", False],
            [self.d1_1.id, "doc1", "lab1", "user2", False],
            [1234, "doc2", "lab2", "user1", False],
            [self.d3_1.id, "doc3", "", "", ""],
        ]
        doc_csv_representation2 = unpack_entries_csv(
            doc_csv_representation1, self.mpr, 1
        )
        self.assertEqual(expected_representation2, doc_csv_representation2)

    # CSV and multi-label
    def test_csv_multiLabel_isAggregated_notUnlabeled(self):
        p = self.p2
        aggregation = True
        unlabeled = False
        docs = [self.d1_2, self.d2_2]

        expected_header = [
            "document_id",
            "text",
            "lab1",
            "lab2",
            "aggregated_label",
            "num_labels",
            "num_annotators",
            "round",
            "MASI_similarity",
            "AL",
            "GL",
        ]
        self.assertEqual(expected_header, p.get_csv_header(aggregation))

        expected_representation1 = [
            [self.d1_2.id, "doc1", 0.5, 0.5, 4, 2, 2, 1.0, False, False],
            [1234, "doc2", 0, 1.0, 1, 1, 2, 1.0, False, False],
        ]
        doc_csv_representation1 = representation_csv(
            docs, aggregation, unlabeled, self.mpr, 1
        )
        self.assertEqual(expected_representation1, doc_csv_representation1)

        expected_representation2 = [
            [self.d1_2.id, "doc1", 0.5, 0.5, "lab1;lab2", 4, 2, 2, 1.0, False, False],
            [1234, "doc2", 0, 1.0, "lab2", 1, 1, 2, 1.0, False, False],
        ]
        doc_csv_representation2 = label_aggregation_csv(
            p, docs, doc_csv_representation1, self.mpr, 1
        )
        self.assertEqual(expected_representation2, doc_csv_representation2)

    def test_csv_multiLabel_isAggregated_isUnlabeled(self):
        p = self.p2
        aggregation = True
        unlabeled = True
        docs = [self.d1_2, self.d2_2, self.d3_2]

        expected_header = [
            "document_id",
            "text",
            "lab1",
            "lab2",
            "aggregated_label",
            "num_labels",
            "num_annotators",
            "round",
            "MASI_similarity",
            "AL",
            "GL",
        ]
        self.assertEqual(expected_header, p.get_csv_header(aggregation))

        expected_representation1 = [
            [self.d1_2.id, "doc1", 0.5, 0.5, 4, 2, 2, 1.0, False, False],
            [1234, "doc2", 0, 1.0, 1, 1, 2, 1.0, False, False],
            [self.d3_2.id, "doc3", "", "", "", "", "", "", False, False],
        ]
        doc_csv_representation1 = representation_csv(
            docs, aggregation, unlabeled, self.mpr, 1
        )
        self.assertEqual(expected_representation1, doc_csv_representation1)

        expected_representation2 = [
            [self.d1_2.id, "doc1", 0.5, 0.5, "lab1;lab2", 4, 2, 2, 1.0, False, False],
            [1234, "doc2", 0, 1.0, "lab2", 1, 1, 2, 1.0, False, False],
            [self.d3_2.id, "doc3", "", "", "", "", "", "", "", False, False],
        ]
        doc_csv_representation2 = label_aggregation_csv(
            p, docs, doc_csv_representation1, self.mpr, 1
        )
        self.assertEqual(expected_representation2, doc_csv_representation2)

    def test_csv_multiLabel_notAggregated_notUnlabeled(self):
        p = self.p2
        aggregation = False
        unlabeled = False
        docs = [self.d1_2, self.d2_2]

        expected_header = ["document_id", "text", "label", "annotator", "GL"]
        self.assertEqual(expected_header, p.get_csv_header(aggregation))

        expected_representation1 = [
            [
                [self.d1_2.id, "doc1", "lab1", "user1", False],
                [self.d1_2.id, "doc1", "lab1", "user2", False],
                [self.d1_2.id, "doc1", "lab2", "user1", False],
                [self.d1_2.id, "doc1", "lab2", "user2", False],
            ],
            [[1234, "doc2", "lab2", "user1", False]],
        ]
        doc_csv_representation1 = representation_csv(
            docs, aggregation, unlabeled, self.mpr, 1
        )
        self.assertEqual(expected_representation1, doc_csv_representation1)

        expected_representation2 = [
            [self.d1_2.id, "doc1", "lab1", "user1", False],
            [self.d1_2.id, "doc1", "lab1", "user2", False],
            [self.d1_2.id, "doc1", "lab2", "user1", False],
            [self.d1_2.id, "doc1", "lab2", "user2", False],
            [1234, "doc2", "lab2", "user1", False],
        ]
        doc_csv_representation2 = unpack_entries_csv(
            doc_csv_representation1, self.mpr, 1
        )
        self.assertEqual(expected_representation2, doc_csv_representation2)

    def test_csv_multiLabel_notAggregated_isUnlabeled(self):
        p = self.p2
        aggregation = False
        unlabeled = True
        docs = [self.d1_2, self.d2_2, self.d3_2]

        expected_header = ["document_id", "text", "label", "annotator", "GL"]
        self.assertEqual(expected_header, p.get_csv_header(aggregation))

        expected_representation1 = [
            [
                [self.d1_2.id, "doc1", "lab1", "user1", False],
                [self.d1_2.id, "doc1", "lab1", "user2", False],
                [self.d1_2.id, "doc1", "lab2", "user1", False],
                [self.d1_2.id, "doc1", "lab2", "user2", False],
            ],
            [[1234, "doc2", "lab2", "user1", False]],
            [[self.d3_2.id, "doc3", "", "", ""]],
        ]
        doc_csv_representation1 = representation_csv(
            docs, aggregation, unlabeled, self.mpr, 1
        )
        self.assertEqual(expected_representation1, doc_csv_representation1)

        expected_representation2 = [
            [self.d1_2.id, "doc1", "lab1", "user1", False],
            [self.d1_2.id, "doc1", "lab1", "user2", False],
            [self.d1_2.id, "doc1", "lab2", "user1", False],
            [self.d1_2.id, "doc1", "lab2", "user2", False],
            [1234, "doc2", "lab2", "user1", False],
            [self.d3_2.id, "doc3", "", "", ""],
        ]
        doc_csv_representation2 = unpack_entries_csv(
            doc_csv_representation1, self.mpr, 1
        )
        self.assertEqual(expected_representation2, doc_csv_representation2)

    # JSON and single-label
    def test_json_singleLabel_isAggregated_notUnlabeled(self):
        p = self.p1
        aggregation = True
        unlabeled = False
        docs = [self.d1_1, self.d2_1]

        expected_representation1 = [
            {
                "document_id": self.d1_1.id,
                "text": "doc1",
                "lab1": 1.0,
                "lab2": 0,
                "num_labels": 2,
                "num_annotators": 2,
                "round": 1,
                "AL": False,
                "GL": False,
            },
            {
                "document_id": 1234,
                "text": "doc2",
                "lab1": 0,
                "lab2": 1.0,
                "num_labels": 1,
                "num_annotators": 1,
                "round": 1,
                "AL": False,
                "GL": False,
            },
        ]
        json_list1 = representation_json(docs, aggregation, unlabeled, self.mpr, 1)
        self.assertEqual(expected_representation1, json_list1)

        expected_representation2 = expected_representation1.copy()
        expected_representation2[0]["aggregated_label"] = "lab1"
        expected_representation2[1]["aggregated_label"] = "lab2"
        json_list2 = label_aggregation_json(p, docs, json_list1, self.mpr, 1)
        self.assertEqual(expected_representation2, json_list2)

    def test_json_singleLabel_isAggregated_isUnlabeled(self):
        p = self.p1
        aggregation = True
        unlabeled = True
        docs = [self.d1_1, self.d2_1, self.d3_1]

        expected_representation1 = [
            {
                "document_id": self.d1_1.id,
                "text": "doc1",
                "lab1": 1.0,
                "lab2": 0,
                "num_labels": 2,
                "num_annotators": 2,
                "round": 1,
                "AL": False,
                "GL": False,
            },
            {
                "document_id": 1234,
                "text": "doc2",
                "lab1": 0,
                "lab2": 1.0,
                "num_labels": 1,
                "num_annotators": 1,
                "round": 1,
                "AL": False,
                "GL": False,
            },
            {
                "document_id": self.d3_1.id,
                "text": "doc3",
                "lab1": "",
                "lab2": "",
                "num_labels": "",
                "num_annotators": "",
                "round": "",
                "AL": False,
                "GL": False,
            },
        ]
        json_list1 = representation_json(docs, aggregation, unlabeled, self.mpr, 1)
        self.assertEqual(expected_representation1, json_list1)

        expected_representation2 = expected_representation1.copy()
        expected_representation2[0]["aggregated_label"] = "lab1"
        expected_representation2[1]["aggregated_label"] = "lab2"
        expected_representation2[2]["aggregated_label"] = ""
        json_list2 = label_aggregation_json(p, docs, json_list1, self.mpr, 1)
        self.assertEqual(expected_representation2, json_list2)

    def test_json_singleLabel_notAggregated_notUnlabeled(self):
        p = self.p1
        aggregation = False
        unlabeled = False
        docs = [self.d1_1, self.d2_1]

        expected_representation = [
            {
                "document_id": self.d1_1.id,
                "text": "doc1",
                "label": "lab1",
                "annotator": "user1",
                "GL": False,
            },
            {
                "document_id": self.d1_1.id,
                "text": "doc1",
                "label": "lab1",
                "annotator": "user2",
                "GL": False,
            },
            {
                "document_id": 1234,
                "text": "doc2",
                "label": "lab2",
                "annotator": "user1",
                "GL": False,
            },
        ]
        json_list = representation_json(docs, aggregation, unlabeled, self.mpr, 1)
        for expect in expected_representation:
            self.assertIn(expect, json_list)
        self.assertEqual(len(expected_representation), len(json_list))

    def test_json_singleLabel_notAggregated_isUnlabeled(self):
        p = self.p1
        aggregation = False
        unlabeled = True
        docs = [self.d1_1, self.d2_1, self.d3_1]

        expected_representation = [
            {
                "document_id": self.d1_1.id,
                "text": "doc1",
                "label": "lab1",
                "annotator": "user1",
                "GL": False,
            },
            {
                "document_id": self.d1_1.id,
                "text": "doc1",
                "label": "lab1",
                "annotator": "user2",
                "GL": False,
            },
            {
                "document_id": 1234,
                "text": "doc2",
                "label": "lab2",
                "annotator": "user1",
                "GL": False,
            },
            {
                "document_id": self.d3_1.id,
                "text": "doc3",
                "label": "",
                "annotator": "",
                "GL": "",
            },
        ]
        json_list = representation_json(docs, aggregation, unlabeled, self.mpr, 1)
        for expect in expected_representation:
            self.assertIn(expect, json_list)
        self.assertEqual(len(expected_representation), len(json_list))

    # JSON and multi-label
    def test_json_multiLabel_isAggregated_notUnlabeled(self):
        p = self.p2
        aggregation = True
        unlabeled = False
        docs = [self.d1_2, self.d2_2]

        expected_representation1 = [
            {
                "document_id": self.d1_2.id,
                "text": "doc1",
                "lab1": 0.5,
                "lab2": 0.5,
                "num_labels": 4,
                "num_annotators": 2,
                "round": 2,
                "MASI_similarity": 1.0,
                "AL": False,
                "GL": False,
            },
            {
                "document_id": 1234,
                "text": "doc2",
                "lab1": 0,
                "lab2": 1.0,
                "num_labels": 1,
                "num_annotators": 1,
                "round": 2,
                "MASI_similarity": 1.0,
                "AL": False,
                "GL": False,
            },
        ]
        json_list1 = representation_json(docs, aggregation, unlabeled, self.mpr, 1)
        self.assertEqual(expected_representation1, json_list1)

        expected_representation2 = expected_representation1.copy()
        expected_representation2[0]["aggregated_label"] = ["lab1", "lab2"]
        expected_representation2[1]["aggregated_label"] = ["lab2"]
        json_list2 = label_aggregation_json(p, docs, json_list1, self.mpr, 1)
        self.assertEqual(expected_representation2, json_list2)

    def test_json_multiLabel_isAggregated_isUnlabeled(self):
        p = self.p2
        aggregation = True
        unlabeled = True
        docs = [self.d1_2, self.d2_2, self.d3_2]

        expected_representation1 = [
            {
                "document_id": self.d1_2.id,
                "text": "doc1",
                "lab1": 0.5,
                "lab2": 0.5,
                "num_labels": 4,
                "num_annotators": 2,
                "round": 2,
                "MASI_similarity": 1.0,
                "AL": False,
                "GL": False,
            },
            {
                "document_id": 1234,
                "text": "doc2",
                "lab1": 0,
                "lab2": 1.0,
                "num_labels": 1,
                "num_annotators": 1,
                "round": 2,
                "MASI_similarity": 1.0,
                "AL": False,
                "GL": False,
            },
            {
                "document_id": self.d3_2.id,
                "text": "doc3",
                "lab1": "",
                "lab2": "",
                "num_labels": "",
                "num_annotators": "",
                "round": "",
                "MASI_similarity": "",
                "AL": False,
                "GL": False,
            },
        ]
        json_list1 = representation_json(docs, aggregation, unlabeled, self.mpr, 1)
        self.assertEqual(expected_representation1, json_list1)

        expected_representation2 = expected_representation1.copy()
        expected_representation2[0]["aggregated_label"] = ["lab1", "lab2"]
        expected_representation2[1]["aggregated_label"] = ["lab2"]
        expected_representation2[2]["aggregated_label"] = ""
        json_list2 = label_aggregation_json(p, docs, json_list1, self.mpr, 1)
        self.assertEqual(expected_representation2, json_list2)

    def test_json_multiLabel_notAggregated_notUnlabeled(self):
        p = self.p1
        aggregation = False
        unlabeled = False
        docs = [self.d1_2, self.d2_2]

        expected_representation = [
            {
                "document_id": self.d1_2.id,
                "text": "doc1",
                "label": "lab1",
                "annotator": "user1",
                "GL": False,
            },
            {
                "document_id": self.d1_2.id,
                "text": "doc1",
                "label": "lab1",
                "annotator": "user2",
                "GL": False,
            },
            {
                "document_id": self.d1_2.id,
                "text": "doc1",
                "label": "lab2",
                "annotator": "user1",
                "GL": False,
            },
            {
                "document_id": self.d1_2.id,
                "text": "doc1",
                "label": "lab2",
                "annotator": "user2",
                "GL": False,
            },
            {
                "document_id": 1234,
                "text": "doc2",
                "label": "lab2",
                "annotator": "user1",
                "GL": False,
            },
        ]
        json_list = representation_json(docs, aggregation, unlabeled, self.mpr, 1)
        for expect in expected_representation:
            self.assertIn(expect, json_list)
        self.assertEqual(len(expected_representation), len(json_list))

    def test_json_multiLabel_notAggregated_isUnlabeled(self):
        p = self.p1
        aggregation = False
        unlabeled = True
        docs = [self.d1_2, self.d2_2, self.d3_2]

        expected_representation = [
            {
                "document_id": self.d1_2.id,
                "text": "doc1",
                "label": "lab1",
                "annotator": "user1",
                "GL": False,
            },
            {
                "document_id": self.d1_2.id,
                "text": "doc1",
                "label": "lab1",
                "annotator": "user2",
                "GL": False,
            },
            {
                "document_id": self.d1_2.id,
                "text": "doc1",
                "label": "lab2",
                "annotator": "user1",
                "GL": False,
            },
            {
                "document_id": self.d1_2.id,
                "text": "doc1",
                "label": "lab2",
                "annotator": "user2",
                "GL": False,
            },
            {
                "document_id": 1234,
                "text": "doc2",
                "label": "lab2",
                "annotator": "user1",
                "GL": False,
            },
            {
                "document_id": self.d3_2.id,
                "text": "doc3",
                "label": "",
                "annotator": "",
                "GL": "",
            },
        ]
        json_list = representation_json(docs, aggregation, unlabeled, self.mpr, 1)
        for expect in expected_representation:
            self.assertIn(expect, json_list)
        self.assertEqual(len(expected_representation), len(json_list))
