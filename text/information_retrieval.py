from rank_bm25 import BM25Okapi
from text.preprocessing import Indexer


def bm25_search(query, doc_indexing, doc_ids, lang):
    indexer = Indexer(lang)

    doc_indexing = [indexing.split() for indexing in doc_indexing]
    query_indexing = indexer.index([query])[0]

    bm25_obj = BM25Okapi(doc_indexing)
    scores = bm25_obj.get_scores(query_indexing)

    score_idx_pair = list(map(lambda score, idx: (score, idx), scores, doc_ids))
    relevant_only = filter(lambda pair: pair[0] > 0, score_idx_pair)
    score_sorted = sorted(relevant_only, key=lambda item: item[0], reverse=True)
    return list(map(lambda item: item[1], score_sorted))
