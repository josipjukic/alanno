from itertools import combinations
from collections import Counter
from nltk.metrics.distance import masi_distance

from .document import Document


class ClassificationDocument(Document):
    def get_annotations(self, is_null=False):
        return self.doc_annotations.filter(user__isnull=is_null)

    def get_all_annotations(self):
        return self.doc_annotations.all()

    def get_gold_label(self):
        gold_labels = None
        annotations = self.get_annotations(is_null=True)
        if annotations:
            gold_labels = sorted([anno.label.text for anno in annotations])
        else:
            annotations = self.get_gl_annotations()
            if annotations:
                gold_labels = sorted([anno.label.text for anno in annotations])
        return gold_labels

    def get_gl_annotations(self):
        # Check if current main annotator's guided learning annotations exist
        annotations = self.doc_annotations.filter(
            gl_annotation=True, user=self.project.main_annotator
        )
        if not annotations.exists():
            # Check if any annotator's guided learning annotations exist
            user = self.doc_annotations.values_list("user", flat=True).first()
            annotations = self.doc_annotations.filter(
                gl_annotation=True, user=user)
        return annotations

    def make_dataset(self, aggregation, unlabeled, *args, **kwargs):
        if self.gl_annotated:
            annotations = self.get_gl_annotations()
        else:
            annotations = self.get_annotations()
        id_property = self.id if self.document_id is None else self.document_id
        if aggregation:
            labels = sorted(
                [label.text for label in self.project.labels.all()])
            round_number = self.round.number + 1 if self.round is not None else ""
            dataset = [id_property, self.raw_html]
            if len(annotations) > 0:
                count = Counter()
                multi_sets = {}
                for a in annotations:
                    if (
                        self.completed_by.filter(id__exact=a.user.id).exists()
                        or self.gl_annotated
                    ):
                        count[a.label.text] += 1
                        if a.user.id in multi_sets:
                            multi_sets[a.user.id].add(a.label.text)
                        else:
                            multi_sets[a.user.id] = {a.label.text}
                num_labels = sum(count.values())
                if num_labels == 0:
                    if unlabeled:
                        dataset += ["" for _ in labels]
                        dataset += (
                            ["", "", "", ""]
                            if self.project.multilabel
                            else ["", "", ""]
                        )
                        dataset += [self.is_al, self.gl_annotated]
                        return dataset
                    else:
                        return []
                dataset += [
                    round(count[label] / num_labels,
                          4) if label in count else 0
                    for label in labels
                ] + [num_labels, len(multi_sets), round_number]
                if self.project.multilabel:
                    if len(multi_sets) > 1:
                        num_combinations = 0
                        masi = 0
                        for i, j in combinations(multi_sets.keys(), 2):
                            masi += 1 - \
                                masi_distance(multi_sets[i], multi_sets[j])
                            num_combinations += 1
                        dataset += [round(masi / num_combinations, 4)]
                    else:
                        dataset += [1.0]
                dataset += [self.is_al, self.gl_annotated]
            else:
                dataset += ["" for _ in labels]
                dataset += ["", "", "",
                            ""] if self.project.multilabel else ["", "", ""]
                dataset += [self.is_al, self.gl_annotated]
        else:
            if len(annotations) > 0:
                dataset = [
                    [
                        id_property,
                        self.raw_html,
                        a.label.text,
                        a.user.username,
                        self.gl_annotated,
                    ]
                    for a in annotations
                    if self.completed_by.filter(id__exact=a.user.id).exists()
                    or self.gl_annotated
                ]
            else:
                dataset = [[id_property, self.raw_html, "", "", ""]]
        return dataset

    def make_dataset_json(self, aggregation, unlabeled, *args, **kwargs):
        if self.gl_annotated:
            annotations = self.get_gl_annotations()
        else:
            annotations = self.get_annotations()
        id_property = self.id if self.document_id is None else self.document_id
        if aggregation:
            labels = [label.text for label in self.project.labels.all()]
            round_number = self.round.number + 1 if self.round is not None else ""
            dataset = {"document_id": id_property, "text": self.raw_html}
            if len(annotations) > 0:
                count = Counter()
                multi_sets = {}
                for a in annotations:
                    if (
                        self.completed_by.filter(id__exact=a.user.id).exists()
                        or self.gl_annotated
                    ):
                        count[a.label.text] += 1
                        if a.user.id in multi_sets:
                            multi_sets[a.user.id].add(a.label.text)
                        else:
                            multi_sets[a.user.id] = {a.label.text}
                num_labels = sum(count.values())
                if num_labels == 0:
                    if unlabeled:
                        dataset = {label: "" for label in labels}
                        dataset["document_id"] = id_property
                        dataset["text"] = self.raw_html
                        dataset["num_labels"] = ""
                        dataset["num_annotators"] = ""
                        dataset["round"] = ""
                        dataset["AL"] = self.is_al
                        dataset["GL"] = self.gl_annotated
                        if self.project.multilabel:
                            dataset["MASI_similarity"] = ""
                    else:
                        return []
                for label in labels:
                    dataset[label] = (
                        round(count[label] / num_labels,
                              4) if label in count else 0
                    )
                dataset["num_labels"] = num_labels
                dataset["num_annotators"] = len(multi_sets)
                dataset["round"] = round_number
                if self.project.multilabel:
                    if len(multi_sets) > 1:
                        num_combinations = 0
                        masi = 0
                        for i, j in combinations(multi_sets.keys(), 2):
                            masi += 1 - \
                                masi_distance(multi_sets[i], multi_sets[j])
                            num_combinations += 1
                        dataset["MASI_similarity"] = round(
                            masi / num_combinations, 4)
                    else:
                        dataset["MASI_similarity"] = 1.0
                dataset["AL"] = self.is_al
                dataset["GL"] = self.gl_annotated
            else:
                dataset = {label: "" for label in labels}
                dataset["document_id"] = id_property
                dataset["text"] = self.raw_html
                dataset["num_labels"] = ""
                dataset["num_annotators"] = ""
                dataset["round"] = ""
                dataset["AL"] = self.is_al
                dataset["GL"] = self.gl_annotated
                if self.project.multilabel:
                    dataset["MASI_similarity"] = ""
            dataset = [dataset]
        else:
            dataset = []
            if len(annotations) > 0:
                for a in annotations:
                    if (
                        self.completed_by.filter(id=a.user.id).exists()
                        or self.gl_annotated
                    ):
                        anno_dp = {
                            "document_id": id_property,
                            "text": self.clean_text(),
                            "label": a.label.text,
                            "annotator": a.user.username,
                            "GL": self.gl_annotated,
                        }
                        dataset.append(anno_dp)
            else:
                dataset.append(
                    {
                        "document_id": id_property,
                        "text": self.clean_text(),
                        "label": "",
                        "annotator": "",
                        "GL": "",
                    }
                )
        return dataset
