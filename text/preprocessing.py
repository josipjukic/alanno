from podium import Dataset
import pandas as pd
import spacy
from podium.vectorizers import GloVe
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import csr_matrix
import string
from nltk import ngrams
import io
import numpy as np
from transformers import AutoTokenizer
from podium import Vocab


def make_labeled_dataset(docs, labs, fields):
    texts = [doc.text for doc in docs]
    df = pd.DataFrame(data={"text": texts, "label": labs})
    dataset = Dataset.from_pandas(df, fields)
    return dataset


def make_dataset(docs, fields):
    texts = [doc.text for doc in docs]
    df = pd.DataFrame(data={"text": texts})
    dataset = Dataset.from_pandas(df, fields)
    return dataset


def get_tokenizer(lang):
    if lang == "en":
        from spacy.lang.en import English

        nlp = English()
    elif lang == "hr":
        from spacy.lang.hr import Croatian

        nlp = Croatian()
    else:
        raise RuntimeError(f"Language {lang} is not supported.")

    tokenizer = nlp.tokenizer

    tokenize = lambda text: [doc.text for doc in tokenizer(text)]
    return tokenize


def get_punctuation():
    return set(string.punctuation)


def get_stemmer(lang):
    if lang == "en":
        from nltk.stem.porter import PorterStemmer

        return PorterStemmer().stem

    elif lang == "hr":
        from text.cro_stemmer import CroStemmer

        return CroStemmer().stem

    else:
        raise RuntimeError(f"Language {lang} is not supported.")


def load_vectors(lang, vocab):
    if lang == "en":
        glove = GloVe()
        embeddings = glove.load_vocab(vocab)
        embeddings[vocab.get_padding_index()] = np.zeros(300)
        return embeddings

    if lang == "hr":
        fname = "text/vector_data/cc.hr.300.vec"

    else:
        raise RuntimeError(f"Language {lang} is not supported.")

    fin = io.open(
        fname,
        "r",
        encoding="utf-8",
        newline="\n",
        errors="ignore",
    )
    n, d = map(int, fin.readline().split())
    # unknown words have a random vector drown from normal distribution
    embeddings = np.random.normal(0, 1, size=(len(vocab), 300))
    # padding vector is a vector of zeros
    embeddings[vocab.get_padding_index()] = np.zeros(300)
    for line in fin:
        tokens = line.rstrip().split(" ")
        if tokens[0] in vocab.stoi:
            embeddings[vocab.stoi[tokens[0]]] = np.array(list(map(float, tokens[1:])))

    return embeddings


class EmbeddingMatrix:
    def __init__(self, lang, vocab=None):
        self.vocab = vocab
        self.lang = lang
        self.embeddings = []

    def fit(self, data, field):
        self.vocab = self.vocab if self.vocab else field.vocab
        self.embeddings = load_vectors(self.lang, self.vocab)

    def transform(self, data):
        return data


class AverageWordVector:
    def __init__(self, lang, vocab=None):
        self.vocab = vocab
        self.lang = lang
        self.embeddings = []

    def fit(self, data, field):
        # Set embedding matrix
        self.vocab = self.vocab if self.vocab else field.vocab
        self.embeddings = load_vectors(self.lang, self.vocab)

    def transform(self, data):
        # Make list of 300 dim word vectors
        transformed = [self.embeddings[x, :] for x in data]
        # Pad vectors with 300 dim nan vectors so they all have the same number of vectors
        max_len = max(transformed, key=lambda x: x.shape[0]).shape[0]
        transformed = [
            np.concatenate(
                (x, np.tile(np.full(300, np.nan), ((max_len - x.shape[0]), 1))), axis=0
            )
            for x in transformed
        ]
        return np.nanmean(np.array(transformed), axis=1)


class CountVectorizer:
    def __init__(self, vocab=None):
        self.vocab = vocab

    def fit(self, data, field):
        self.vocab = self.vocab if self.vocab else field.vocab
        self.vector_size = len(self.vocab.itos)

    def transform(self, data):
        # The standard CSR representation where the column indices for row i are stored in
        # indices[indptr[i]:indptr[i+1]] and their corresponding values are stored in counts[indptr[i]:indptr[i+1]].
        # If the shape parameter is not supplied, the matrix dimensions are inferred from the index arrays.

        indptr = [0]
        indices = []
        counts = []
        for inds in data:
            inds_list = inds.tolist()
            indices.extend(inds_list)
            counts.extend([1 for _ in range(len(inds_list))])
            indptr.append(len(indices))
        return csr_matrix(
            (counts, indices, indptr), dtype=int, shape=(len(data), self.vector_size)
        ).toarray()


class TfIdfVectorizer:
    def __init__(self, vocab=None):
        self.vocab = vocab
        self.count = CountVectorizer(vocab)
        self.tf_idf = TfidfTransformer()

    def fit(self, data, field):
        x = data.batch().text
        self.count.fit(x, field)
        self.tf_idf.fit(self.count.transform(x))

    def transform(self, data):
        counted_data = self.count.transform(data)
        return self.tf_idf.transform(counted_data).toarray()


VECTORIZERS = {
    "tf_idf": TfIdfVectorizer,
    "vec_avg": AverageWordVector,
    "count": CountVectorizer,
    "emb_matrx": EmbeddingMatrix,
}


def get_vectorizer(name):
    try:
        return VECTORIZERS[name]

    except:
        raise ValueError("Vectorizer %s not supported" % (name))


def lowercase(raw, tokenized):
    return raw, [token.lower() for token in tokenized]


def remove_punct(raw, tokenized):
    punct = get_punctuation()
    return raw, [token for token in tokenized if token not in punct]


# def remove_punct(raw, tokenized):
#    punct = set(string.punctuation)
#    return raw, [token for token in tokenized if set(token).isdisjoint(punct)]


class WordNGramHook:
    def __init__(self, min_n, max_n):
        self.min_n = min_n
        self.max_n = max_n

    def __call__(self, raw, tokenized):
        tokenized_ngrams = []
        for n in range(self.min_n, self.max_n + 1):
            tokenized_ngrams.extend(ngrams(tokenized, n))
        return raw, tokenized_ngrams


class CharNGramHook:
    def __init__(self, min_n, max_n):
        self.min_n = min_n
        self.max_n = max_n

    def __call__(self, raw, tokenized):
        tokenized = "".join(tokenized)
        tokenized_ngrams = []
        for n in range(self.min_n, self.max_n + 1):
            tokenized_ngrams.extend(ngrams(list(tokenized), n))
        return raw, tokenized_ngrams


def get_stop_words(lang):
    if lang == "en":
        from spacy.lang.en import English

        nlp = English()
    elif lang == "hr":
        from spacy.lang.hr import Croatian

        nlp = Croatian()
    else:
        raise RuntimeError(f"Language {lang} is not supported.")

    return nlp.Defaults.stop_words


class StopWordsHook:
    def __init__(self, lang):

        self.stop_words = get_stop_words(lang)

    def __call__(self, raw, tokenized):
        return raw, [token for token in tokenized if token not in self.stop_words]


def get_posttokenize_hooks(lang, token_type, vectorizer_name, min_ngram, max_ngram):
    TOKENIZER_TYPE = {"words": WordNGramHook, "chars": CharNGramHook}

    pipeline = [
        remove_punct,
        lowercase,
        StopWordsHook(lang),
    ]

    if vectorizer_name == "count" or vectorizer_name == "tf_idf":
        pipeline.append(TOKENIZER_TYPE[token_type](min_ngram, max_ngram))

    return pipeline


def get_bert_model_name(lang):
    if lang == "en":
        return "distilbert-base-uncased"

    elif lang == "hr":
        return "classla/bcms-bertic"

    else:
        raise RuntimeError(f"Language {lang} is not supported with BERT.")


def get_bert_vectorizer(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def func(x):
        return tokenizer(
            x, truncation=True, padding=True, return_tensors="pt"
        ).to(device)

    return func


class VectorizedVocab(Vocab):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vectorizer = None

    def numericalize(self, data):
        token_indices = super().numericalize(data)
        if self.vectorizer is None:
            return token_indices
        else:
            tfidf_x = self.vectorizer.transform([token_indices])
            return tfidf_x[0]

    def set_vectorizer(self, vectorizer):
        self.vectorizer = vectorizer


class Indexer:
    def __init__(self, lang):
        self.lang = lang
        self.punct = get_punctuation()
        self.tokenizer = get_tokenizer(lang)
        self.stemmer = get_stemmer(lang)
        self.stop = get_stop_words(lang)

    def index(self, texts):
        return [
            [
                self.stemmer(word)
                for word in self.tokenizer(text)
                if word not in self.stop and set(word).isdisjoint(self.punct)
            ]
            for text in texts
        ]
