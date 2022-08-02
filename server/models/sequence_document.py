from .document import Document


class SequenceDocument(Document):
    def get_annotations(self, is_null=False):
        return self.seq_annotations.filter(user__isnull=is_null)

    def get_all_annotations(self):
        return self.seq_annotations.all()

    def get_gold_label(self):
        gold_labels = None
        annotations = self.get_gl_annotations()
        if annotations:
            gold_labels = sorted(
                [
                    (anno.label.text, anno.start_offset, anno.end_offset)
                    for anno in annotations
                ]
            )
        return gold_labels

    def get_gl_annotations(self):
        # Check if current main annotator's guided learning annotations exist
        annotations = self.seq_annotations.filter(
            gl_annotation=True, user=self.project.main_annotator
        )
        if not annotations.exists():
            # Check if any annotator's guided learning annotations exist
            user = self.seq_annotations.values_list("user", flat=True).first()
            annotations = self.seq_annotations.filter(
                gl_annotation=True, user=user)
        return annotations

    def make_dataset(self, *args, **kwargs):
        if self.gl_annotated:
            annotations = self.get_gl_annotations()
        else:
            annotations = self.get_annotations()
        id_property = self.id if self.document_id is None else self.document_id

        if len(annotations) > 0:
            dataset = [
                [
                    id_property,
                    self.raw_html,
                    a.label.text,
                    a.start_offset,
                    a.end_offset,
                    a.user.username,
                    self.gl_annotated,
                ]
                for a in annotations
                if self.completed_by.filter(id__exact=a.user.id) or self.gl_annotated
            ]
        else:
            dataset = [[id_property, self.raw_html, "", "", "", "", ""]]

        return dataset

    def make_dataset_json(self, *args, **kwargs):
        if self.gl_annotated:
            annotations = self.get_gl_annotations()
        else:
            annotations = self.get_annotations()
        id_property = self.id if self.document_id is None else self.document_id

        dataset = []
        if len(annotations) > 0:
            for a in annotations:
                if self.completed_by.filter(id=a.user.id).exists() or self.gl_annotated:
                    anno_dp = {
                        "document_id": id_property,
                        "text": self.clean_text(),
                        "label": a.label.text,
                        "start": a.start_offset,
                        "end": a.end_offset,
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
                    "start": "",
                    "end": "",
                    "annotator": "",
                    "GL": "",
                }
            )

        return dataset
