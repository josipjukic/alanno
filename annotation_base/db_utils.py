from collections import Counter
from django.db.models import Count


def get_unselected_split(proj):
    docs = proj.documents.filter(
        is_selected=False, gl_annotated=False, is_warm_start=False
    )
    train_docs = docs.filter(is_test=False)
    test_docs = docs.filter(is_test=True)
    return train_docs, test_docs


def get_labeled_split(
    proj,
    anno_lower_bound=None,
    use_warm_start=True,
):
    all_docs = proj.documents.annotate(num_anno=Count("completed_by"))
    regular_docs = all_docs.filter(num_anno__gte=anno_lower_bound)
    gl_docs = all_docs.filter(gl_annotated=True)
    if use_warm_start:
        ws_docs = all_docs.filter(is_warm_start=True)
    else:
        ws_docs = all_docs.none()

    docs = regular_docs | gl_docs | ws_docs  # distinct join
    train_docs = list(docs.filter(is_test=False))
    test_docs = list(docs.filter(is_test=True))
    print("Filtered test_docs", list(test_docs))

    return train_docs, test_docs
