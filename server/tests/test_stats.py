from django.test import TestCase
from server.tests.utils import init_clx_project, MockProgressRecorder, add_clx_annotations

from server.services.statistics import *


class LabelCounts(TestCase):
    def setUp(self):
        project, users, labels, docs = init_clx_project(user_range=(0, 3), label_range=(0, 3), document_range=(0, 4))
        self.project = project
        self.users = users
        self.labels = labels
        self.docs = docs

        annotations = [(0, 0, 0), (0, 1, 0), (0, 2, 1),
                       (1, 0, 1), (1, 1, 2), (1, 2, 1),
                       (2, 0, 2), (2, 1, 2), (2, 2, 1)]
        # Document 0 has annotations with all three labels
        # Document 1 only has label 0 or 2 annotations
        # Document 2 only has label 1 annotations
        # Document 3 has no annotations
        add_clx_annotations(self.users, self.docs, self.labels, annotations)

        self.mpr = MockProgressRecorder()

    def test_allDocs_correct(self):
        label_counts = get_label_counts([lbl.text for lbl in self.labels], self.docs)
        self.assertEqual(3, len(label_counts))
        self.assertEqual(2, label_counts[0])
        self.assertEqual(4, label_counts[1])
        self.assertEqual(3, label_counts[2])

    def test_docWithAllLabelsInThreeAnnots_correct(self):
        label_counts = get_label_counts([lbl.text for lbl in self.labels], [self.docs[0]])
        self.assertEqual(3, len(label_counts))
        self.assertEqual(1, label_counts[0])
        self.assertEqual(1, label_counts[1])
        self.assertEqual(1, label_counts[2])

    def test_docWithTwoLabelsInThreeAnnots_correct(self):
        label_counts = get_label_counts([lbl.text for lbl in self.labels], [self.docs[1]])
        self.assertEqual(3, len(label_counts))
        self.assertEqual(1, label_counts[0])
        self.assertEqual(0, label_counts[1])
        self.assertEqual(2, label_counts[2])

    def test_docWithSingleLabelInThreeAnnots_correct(self):
        label_counts = get_label_counts([lbl.text for lbl in self.labels], [self.docs[2]])
        self.assertEqual(3, len(label_counts))
        self.assertEqual(0, label_counts[0])
        self.assertEqual(3, label_counts[1])
        self.assertEqual(0, label_counts[2])

    def test_docWithNoAnnotations_correct(self):
        label_counts = get_label_counts([lbl.text for lbl in self.labels], [self.docs[3]])
        self.assertEqual(3, len(label_counts))
        self.assertEqual(0, label_counts[0])
        self.assertEqual(0, label_counts[1])
        self.assertEqual(0, label_counts[2])

    def test_noDocs_correct(self):
        label_counts = get_label_counts([lbl.text for lbl in self.labels], [])
        self.assertEqual(3, len(label_counts))
        self.assertEqual(0, label_counts[0])
        self.assertEqual(0, label_counts[1])
        self.assertEqual(0, label_counts[2])

    def test_noLabels_emptyList(self):
        label_counts = get_label_counts([], self.docs)
        self.assertEqual(0, len(label_counts))


class UserStats(TestCase):
    def setUp(self):
        project, users, labels, docs = init_clx_project(user_range=(0, 4), label_range=(0, 3), document_range=(0, 4))
        self.project = project
        self.users = users
        self.labels = labels
        self.docs = docs

        for idx_user in range(3):
            for idx_doc in range(3):
                self.docs[idx_doc].is_selected = True
                self.docs[idx_doc].save()
                self.users[idx_user].selected_docs.add(self.docs[idx_doc])
        # Distributed each document 0, 1, 2 to each user 0, 1, 2
        # Document 3 not distributed
        # User 3 not given any documents

        annotations = [(0, 0, 0), (0, 1, 0), (0, 2, 1),             # user 0: annotated all documents
                       (1, 0, 1), (1, 1, 2),            (1, 3, 1),  # user 1: annotated two selected and one extra
                                  (2, 1, 2)]                        # user 2: annotated only one
        add_clx_annotations(self.users, self.docs, self.labels, annotations)

        self.mpr = MockProgressRecorder()

    def test_getDistributedWorks(self):
        docs = self.project.get_distributed_documents()  # regardless of round
        self.assertEqual(3, len(docs))

        for doc in docs:
            self.assertTrue(doc.is_selected)
            self.assertIn(self.users[0], doc.selectors.all())
            self.assertIn(self.users[1], doc.selectors.all())
            self.assertIn(self.users[2], doc.selectors.all())

        self.assertNotIn(self.docs[3], docs)

    def test_allDistributedDocs_allUsers_correct(self):
        user_lists, user_dicts = get_user_stats(self.users, self.project, self.project.get_distributed_documents())
        self.assertEqual(4, len(user_lists["username"]))
        idx_0 = user_lists["username"].index("user0")
        idx_1 = user_lists["username"].index("user1")
        idx_2 = user_lists["username"].index("user2")
        idx_3 = user_lists["username"].index("user3")

        # Check the data
        self.assertEqual(0, user_lists["active"][idx_0])
        self.assertEqual(1, user_lists["active"][idx_1])     # annotation of non-selected doc (3) doesn't count
        self.assertEqual(2, user_lists["active"][idx_2])
        self.assertEqual(0, user_lists["active"][idx_3])

        self.assertEqual(3, user_lists["completed"][idx_0])
        self.assertEqual(2, user_lists["completed"][idx_1])  # annotation of non-selected doc (3) doesn't count
        self.assertEqual(1, user_lists["completed"][idx_2])
        self.assertEqual(0, user_lists["completed"][idx_3])

        self.assertEqual(3, user_lists["total"][idx_0])
        self.assertEqual(3, user_lists["total"][idx_1])
        self.assertEqual(3, user_lists["total"][idx_2])
        self.assertEqual(0, user_lists["total"][idx_3])

        # Check conversion to dict
        self.assertEqual(4, len(user_dicts))
        user0_dict = user_dicts[idx_0]
        self.assertEqual("user0", user0_dict["username"])
        self.assertEqual(0, user0_dict["active"])
        self.assertEqual(3, user0_dict["completed"])
        self.assertEqual(3, user0_dict["total"])

    def test_someDistributedDocs_allUsers_correct(self):
        user_lists, user_dicts = get_user_stats(self.users, self.project, [self.docs[0], self.docs[2]])
        self.assertEqual(4, len(user_lists["username"]))
        idx_0 = user_lists["username"].index("user0")
        idx_1 = user_lists["username"].index("user1")
        idx_2 = user_lists["username"].index("user2")
        idx_3 = user_lists["username"].index("user3")

        # Check the data
        self.assertEqual(0, user_lists["active"][idx_0])
        self.assertEqual(1, user_lists["active"][idx_1])  # annotation of non-selected doc (3) doesn't count
        self.assertEqual(2, user_lists["active"][idx_2])
        self.assertEqual(0, user_lists["active"][idx_3])

        self.assertEqual(2, user_lists["completed"][idx_0])
        self.assertEqual(1, user_lists["completed"][idx_1])  # annotation of non-selected doc (3) doesn't count
        self.assertEqual(0, user_lists["completed"][idx_2])
        self.assertEqual(0, user_lists["completed"][idx_3])

        self.assertEqual(2, user_lists["total"][idx_0])
        self.assertEqual(2, user_lists["total"][idx_1])
        self.assertEqual(2, user_lists["total"][idx_2])
        self.assertEqual(0, user_lists["total"][idx_3])

    def test_someDistributedDocs_someUsers_correct(self):
        user_lists, user_dicts = get_user_stats(self.users[1:], self.project, [self.docs[0], self.docs[2]])
        self.assertEqual(3, len(user_lists["username"]))
        idx_1 = user_lists["username"].index("user1")
        idx_2 = user_lists["username"].index("user2")
        idx_3 = user_lists["username"].index("user3")

        # Check the data
        self.assertEqual(1, user_lists["active"][idx_1])  # annotation of non-selected doc (3) doesn't count
        self.assertEqual(2, user_lists["active"][idx_2])
        self.assertEqual(0, user_lists["active"][idx_3])

        self.assertEqual(1, user_lists["completed"][idx_1])  # annotation of non-selected doc (3) doesn't count
        self.assertEqual(0, user_lists["completed"][idx_2])
        self.assertEqual(0, user_lists["completed"][idx_3])

        self.assertEqual(2, user_lists["total"][idx_1])
        self.assertEqual(2, user_lists["total"][idx_2])
        self.assertEqual(0, user_lists["total"][idx_3])
