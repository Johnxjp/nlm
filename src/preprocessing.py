"""Data preprocessing functionality"""

import nltk
import re


def tokenise_sentences(doc):
    """Tokenises strings"""

    try:
        return nltk.sent_tokenize(doc)
    except LookupError as e:
        nltk.download('punkt')
        return nltk.sent_tokenize(doc)


def remove_nonalpha(sent):
    return re.sub("\W", " ", sent)


def remove_blanks(sent):
    return re.sub("\s+", " ", sent)


def tokenise(sentence):
    return nltk.tokenize.WhitespaceTokenizer().tokenize(sentence)