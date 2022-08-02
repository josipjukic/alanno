import spacy
import spacy_udpipe

##########################
##### META CLASS #########


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


# TODO: disable unnecessary pipeline parts, keep lemmatizer only


##########################
######## CLASSES #########


class CroatianLemmatizer:
    def __init__(self):
        # download HR model
        spacy_udpipe.download("hr")
        self.nlp = spacy_udpipe.load("hr")

    def __call__(self, text):
        return self.nlp(text)


class CroatianLemmatizerSingleton(CroatianLemmatizer, metaclass=Singleton):
    pass


class EnglishLemmatizer:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def __call__(self, text):
        return self.nlp(text)


class EnglishLemmatizerSingleton(EnglishLemmatizer, metaclass=Singleton):
    pass
