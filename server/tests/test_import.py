from django.test import TestCase
from server.tests.utils import init_clx_project, MockProgressRecorder

from server.models import ClassificationProject, DocumentAnnotation, Document
from server.services.import_dataset import *


class ImportCsv(TestCase):

    def setUp(self):
        self.project = ClassificationProject(name="Test project", description="Description")
        self.project.save()
        self.mpr = MockProgressRecorder()

    def test_textHeaderMissing_raisesError(self):
        docs = [
            "document_id,label,annotator",
            "12345,label1,user1",
            "54321,label2,user2"
        ]

        self.assertRaisesMessage(ValueError,
                                 "The 'text' header must be present.",
                                 lambda: import_csv(self.project, docs, self.mpr))


class AggregateCsv(TestCase):

    IDX_ALL = [0, 1, 2, 3]

    IDX_NO_DOCID = [-1, 0, 1, 2]
    IDX_NO_TEXT = [0, -1, 1, 2]
    IDX_NO_LABEL = [0, 1, -1, 2]
    IDX_NO_ANNOT = [0, 1, 2, -1]
    IDX_TEXT_AND_DOCID = [0, 1, -1, -1]
    IDX_TEXT_ONLY = [-1, 0, -1, -1]

    def setUp(self):
        self.mpr = MockProgressRecorder()

    def test_allHeaders_allPresent_noDuplicates_correct(self):
        entries = [
            ["12345", "text1", "label1", "user1"],
            ["54321", "text2", "label2", "user2"]
        ]
        data = aggregate_csv(entries, AggregateCsv.IDX_ALL, self.mpr)

        self.assertIn("12345", data)
        self.assertIn("54321", data)
        self.assertEqual(2, len(data))

        text = data["12345"][K_TXT]
        self.assertEqual("text1", text)

        self.assertIn(K_ANNOTS, data["54321"])
        annots = data["54321"][K_ANNOTS]
        self.assertEqual(1, len(annots))

        self.assertIn("user2", annots)
        labels = annots["user2"]
        self.assertEqual(1, len(labels))
        self.assertIn("label2", labels)

    def test_allHeaders_allPresent_duplicateLabels_duplicatesIgnored(self):
        entries = [
            ["12345", "text1", "label1", "user1"],
            ["12345", "text1", "label1", "user1"]
        ]
        data = aggregate_csv(entries, AggregateCsv.IDX_ALL, self.mpr)
        self.assertEqual(1, len(data))

        annots = data["12345"][K_ANNOTS]
        self.assertEqual(1, len(annots))
        self.assertIn("user1", annots)

        labels = annots["user1"]
        self.assertEqual(1, len(labels))
        self.assertIn("label1", labels)

    def test_allHeaders_allPresent_sameDocumentDifferentText_firstCounts(self):
        entries = [
            ["12345", "ABCDE", "label1", "user1"],
            ["12345", "vwxyz", "label2", "user2"]
        ]
        data = aggregate_csv(entries, AggregateCsv.IDX_ALL, self.mpr)

        self.assertEqual("ABCDE", data["12345"][K_TXT])
        self.assertEqual(2, len(data["12345"][K_ANNOTS]))
        self.assertIn("label1", data["12345"][K_ANNOTS]["user1"])
        self.assertIn("label2", data["12345"][K_ANNOTS]["user2"])

    def test_allHeaders_allPresent_oneUserMoreLabels_correctData(self):
        entries = [
            ["12345", "text1", "label1", "user1"],
            ["12345", "text2", "label2", "user1"]
        ]
        data = aggregate_csv(entries, AggregateCsv.IDX_ALL, self.mpr)
        annots = data["12345"][K_ANNOTS]
        self.assertEqual(1, len(annots))
        self.assertEqual(2, len(annots["user1"]))

    def test_allHeaders_allPresent_oneDocMoreUsers_correctData(self):
        entries = [
            ["12345", "text1", "label1", "user1"],
            ["12345", "text1", "label2", "user2"],
            ["12345", "text1", "label3", "user3"],
            ["12345", "text1", "label2", "user3"]
        ]
        data = aggregate_csv(entries, AggregateCsv.IDX_ALL, self.mpr)
        annots = data["12345"][K_ANNOTS]
        self.assertEqual(3, len(annots))
        self.assertIn("user1", annots)
        self.assertIn("user2", annots)
        self.assertIn("user3", annots)

        self.assertEqual(1, len(annots["user1"]))
        self.assertEqual(1, len(annots["user2"]))
        self.assertEqual(2, len(annots["user3"]))
        self.assertIn("label2", annots["user3"])
        self.assertIn("label3", annots["user3"])

    def test_allHeaders_blankDocId_raisesError(self):
        entries = [
            ["12345", "text1", "label1", "user1"],
            ["", "text2", "label2", "user1"]
        ]
        self.assertRaisesMessage(ValueError,
                                 "If the 'document_id' header exists, all entries must have a defined ID.",
                                 lambda: aggregate_csv(entries, AggregateCsv.IDX_ALL, self.mpr))

    def test_allHeaders_blankTextPreviouslyDefined_noError(self):
        entries = [
            ["12345", "text1", "label1", "user1"],
            ["12345", "", "label2", "user1"]
        ]
        data = aggregate_csv(entries, AggregateCsv.IDX_ALL, self.mpr)
        self.assertEqual("text1", data["12345"][K_TXT])

    def test_allHeaders_blankTextNotDefined_raisesError(self):
        entries = [
            ["12345", "", "label2", "user1"],
            ["12345", "text1", "label1", "user1"]
        ]
        self.assertRaisesMessage(ValueError,
                                 "All new document rows must have a non-blank 'text' field.",
                                 lambda: aggregate_csv(entries, AggregateCsv.IDX_ALL, self.mpr))

    def test_allHeaders_blankLabel_noLabelAndAnnotatorIgnored(self):
        entries = [
            ["12345", "text1", "label1", "user1"],
            ["12345", "text1", "", "user2"],
            ["54321", "text2", "label2", "user2"]
        ]
        data = aggregate_csv(entries, AggregateCsv.IDX_ALL, self.mpr)

        self.assertNotIn("user2", data["12345"][K_ANNOTS])
        self.assertEqual(1, len(data["12345"][K_ANNOTS]))

        self.assertIn("user1", data["12345"][K_ANNOTS])
        self.assertIn("user2", data["54321"][K_ANNOTS])

    def test_allHeaders_blankAnnotator_labelAdded(self):
        entries = [
            ["12345", "text1", "label1", "user1"],
            ["12345", "text1", "label1", ""]
        ]
        data = aggregate_csv(entries, AggregateCsv.IDX_ALL, self.mpr)

        self.assertNotIn("user2", data["12345"][K_ANNOTS])
        self.assertEqual(2, len(data["12345"][K_ANNOTS]))
        self.assertIn(None, data["12345"][K_ANNOTS])
        self.assertIn("label1", data["12345"][K_ANNOTS][None])

    def test_noDocIdHeader_allPresent_documentsNotGrouped(self):
        entries = [
            ["text", "label", "user"],
            ["text", "label", "user"],
            ["text", "label", "user"],
            ["text", "label", "user"]
        ]
        data = aggregate_csv(entries, AggregateCsv.IDX_NO_DOCID, self.mpr)

        self.assertEqual(4, len(data))
        self.assertIn("0", data)
        self.assertIn("1", data)
        self.assertIn("2", data)
        self.assertIn("3", data)

        self.assertEqual(1, len(data["0"][K_ANNOTS]))
        self.assertIn("user", data["0"][K_ANNOTS])
        self.assertEqual(1, len(data["0"][K_ANNOTS]["user"]))
        self.assertIn("label", data["0"][K_ANNOTS]["user"])

    def test_noDocIdHeader_blankText_raisesError(self):
        entries = [
            ["text", "label1", "user1"],
            ["", "label2", "user2"]
        ]
        self.assertRaisesMessage(ValueError,
                                 "All new document rows must have a non-blank 'text' field.",
                                 lambda: aggregate_csv(entries, AggregateCsv.IDX_NO_DOCID, self.mpr))

    def test_noLabelHeader_allPresent_annotatorsIgnored(self):
        entries = [
            ["12345", "text1", "user1"],
            ["12345", "text1", "user2"],
            ["54321", "text2", "user1"],
            ["54321", "text2", "user2"]
        ]
        data = aggregate_csv(entries, AggregateCsv.IDX_NO_LABEL, self.mpr)

        self.assertEqual(2, len(data))
        self.assertEqual(0, len(data["12345"][K_ANNOTS]))
        self.assertEqual(0, len(data["54321"][K_ANNOTS]))

    def test_noAnnotatorHeader_allPresent_labelsAdded(self):
        entries = [
            ["12345", "text1", "label1"],
            ["12345", "text1", "label2"],
            ["54321", "text2", "label2"],
            ["54321", "text2", "label3"]
        ]
        data = aggregate_csv(entries, AggregateCsv.IDX_NO_ANNOT, self.mpr)

        self.assertEqual(2, len(data))

        self.assertEqual(1, len(data["12345"][K_ANNOTS]))
        self.assertIn(None, data["12345"][K_ANNOTS])
        self.assertEqual(2, len(data["12345"][K_ANNOTS][None]))
        self.assertIn("label1", data["12345"][K_ANNOTS][None])
        self.assertIn("label2", data["12345"][K_ANNOTS][None])

        self.assertEqual(1, len(data["54321"][K_ANNOTS]))
        self.assertIn(None, data["54321"][K_ANNOTS])
        self.assertEqual(2, len(data["54321"][K_ANNOTS][None]))
        self.assertIn("label2", data["54321"][K_ANNOTS][None])
        self.assertIn("label3", data["54321"][K_ANNOTS][None])

    def test_onlyTextAndDocId_allPresent_correctData(self):
        entries = [
            ["12345", "text1"],
            ["12345", ""],
            ["54321", "text2"],
            ["33333", "text3"]
        ]
        data = aggregate_csv(entries, AggregateCsv.IDX_TEXT_AND_DOCID, self.mpr)

        self.assertEqual(3, len(data))
        self.assertEqual("text1", data["12345"][K_TXT])
        self.assertEqual("text2", data["54321"][K_TXT])
        self.assertEqual("text3", data["33333"][K_TXT])
        self.assertEqual(0, len(data["12345"][K_ANNOTS]))
        self.assertEqual(0, len(data["54321"][K_ANNOTS]))
        self.assertEqual(0, len(data["33333"][K_ANNOTS]))

    def test_onlyText_allPresent_correctData(self):
        entries = [["text1"], ["text2"], ["text1"]]
        data = aggregate_csv(entries, AggregateCsv.IDX_TEXT_ONLY, self.mpr)
        self.assertEqual(3, len(data))
        self.assertIn("0", data)
        self.assertIn("1", data)
        self.assertIn("2", data)
        self.assertEqual(data["0"][K_TXT], data["2"][K_TXT])
        self.assertEqual("text2", data["1"][K_TXT])


class BuildCsv(TestCase):

    def setUp(self):
        project, users, labels, _ = init_clx_project(user_range=(1, 3), label_range=(1, 4))
        self.project = project
        self.user1, self.user2 = users
        self.lbl1, self.lbl2, self.lbl3 = labels

        self.template1 = {K_TXT: "text1", K_ANNOTS: {"user1": ["label1"], "user2": ["label2"]}}
        self.template2 = {K_TXT: "text2", K_ANNOTS: {"user1": ["label2"], "user2": ["label3"]}}

        self.mpr = MockProgressRecorder()

    def test_hasDocId_correctCase(self):
        data = {"12345": self.template1, "54321": self.template2}
        docs, label_stats = build_documents(data, self.project, True, self.mpr)
        self.assertEqual(2, len(docs))
        self.assertEqual(2, label_stats[0])
        self.assertEqual(4, label_stats[1])

        doc_ids = [d.document_id for d in docs]
        self.assertIn("12345", doc_ids)
        self.assertIn("54321", doc_ids)

        idx1 = doc_ids.index("12345")
        doc1 = docs[idx1]
        self.assertEqual("text1", doc1.text)
        self.assertEqual(2, DocumentAnnotation.objects.filter(document_id=doc1.id).count())
        self.assertEqual(1, DocumentAnnotation.objects.filter(document_id=doc1.id, label_id=self.lbl1.id).count())
        self.assertEqual(0, DocumentAnnotation.objects.filter(document_id=doc1.id, label_id=self.lbl3.id).count())

        doc2 = docs[1 - idx1]
        annotation2 = DocumentAnnotation.objects.filter(document_id=doc2.id, label_id=self.lbl2.id).first()
        annotation3 = DocumentAnnotation.objects.filter(document_id=doc2.id, label_id=self.lbl3.id).first()
        self.assertEqual(self.user1.id, annotation2.user_id)
        self.assertEqual(self.user2.id, annotation3.user_id)

        self.assertIn(self.user1, doc1.completed_by.all())
        self.assertIn(self.user2, doc1.completed_by.all())
        self.assertIn(self.user1, doc2.completed_by.all())
        self.assertIn(self.user2, doc2.completed_by.all())

    def test_noDocId_internalIdIndependent_docIdIsNone(self):
        data = {"0": self.template1, "1": self.template2}
        docs, label_stats = build_documents(data, self.project, False, self.mpr)

        ids = [d.id for d in docs]
        doc_ids = [d.document_id for d in docs]
        self.assertEqual(2, len(ids))
        self.assertEqual(2, len(doc_ids))
        self.assertNotIn("0", ids)
        self.assertNotIn("1", ids)
        self.assertNotIn("0", doc_ids)
        self.assertNotIn("1", doc_ids)
        self.assertEqual(None, doc_ids[0])
        self.assertEqual(None, doc_ids[1])

    def test_withDocId_noAnnotations_correctDocs(self):
        entry1 = self.template1
        entry2 = self.template2
        entry1[K_ANNOTS] = {}
        entry2[K_ANNOTS] = {}
        data = {"12345": entry1, "54321": entry2}

        docs, label_stats = build_documents(data, self.project, True, self.mpr)
        self.assertEqual(0, label_stats[0])
        self.assertEqual(0, label_stats[1])
        self.assertEqual(2, len(docs))
        self.assertEqual(0, len(docs[0].get_all_annotations()))

    def test_oneAnnotatorTwoLabels_ignoredForSingleLabelProject(self):
        entry = self.template1
        del entry[K_ANNOTS]["user2"]
        entry[K_ANNOTS]["user1"].append("label2")

        data = {"12345": entry, "54321": self.template2}
        docs, label_stats = build_documents(data, self.project, True, self.mpr)

        self.assertEqual(1, label_stats[0])
        self.assertEqual(2, label_stats[1])

        doc12345 = Document.objects.filter(document_id="12345").first()
        doc54321 = Document.objects.filter(document_id="54321").first()
        self.assertEqual(0, DocumentAnnotation.objects.filter(document_id=doc12345.id).count())
        self.assertEqual(2, DocumentAnnotation.objects.filter(document_id=doc54321.id).count())

    def test_noneAnnotatorTwoLabels_accepted(self):
        entry = self.template1
        entry[K_ANNOTS][None] = entry[K_ANNOTS]["user1"] + entry[K_ANNOTS]["user2"]
        del entry[K_ANNOTS]["user1"]
        del entry[K_ANNOTS]["user2"]

        data = {"12345": entry, "54321": self.template2}
        docs, label_stats = build_documents(data, self.project, True, self.mpr)

        self.assertEqual(1, label_stats[0])
        self.assertEqual(2, label_stats[1])

        doc12345 = Document.objects.filter(document_id="12345").first()
        self.assertEqual(0, DocumentAnnotation.objects.filter(document_id=doc12345.id).count())

    def test_nonexistentAnnotator_labelAdded_annotatorIgnored(self):
        entry = self.template1
        entry[K_ANNOTS]["who_is_this"] = entry[K_ANNOTS]["user1"]
        del entry[K_ANNOTS]["user1"]

        data = {"12345": entry, "54321": self.template2}
        docs, label_stats = build_documents(data, self.project, True, self.mpr)

        self.assertEqual(2, label_stats[0])
        self.assertEqual(3, label_stats[1])

        doc12345 = Document.objects.filter(document_id="12345").first()
        self.assertEqual(1, DocumentAnnotation.objects.filter(document_id=doc12345.id).count())
        self.assertEqual(0, DocumentAnnotation.objects.filter(document_id=doc12345.id, user=None).count())

    def test_nonexistentLabel_annotationIgnored(self):
        entry = self.template1
        entry[K_ANNOTS]["user1"] = ["unknown_label"]

        data = {"12345": entry, "54321": self.template2}
        docs, label_stats = build_documents(data, self.project, True, self.mpr)

        self.assertEqual(2, label_stats[0])     # first doc labeled by user 2, second doc labeled by both
        self.assertEqual(3, label_stats[1])     # first doc has 1 correct annotation, second has both correct

        doc12345 = Document.objects.filter(document_id="12345").first()
        self.assertEqual(1, DocumentAnnotation.objects.filter(document_id=doc12345.id).count())
