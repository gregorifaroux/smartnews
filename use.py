import pickle, gensim
from operator import itemgetter

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


# Text Cleaning
import spacy
spacy.load('en')
from spacy.lang.en import English
parser = English()
def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens

#We use NLTK’s Wordnet to find the meanings of words, synonyms, antonyms, and more. In addition, we use WordNetLemmatizer to get the root word.
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return get_lemma2(word)
    else:
        return get_lemma2(lemma)
    
from nltk.stem.wordnet import WordNetLemmatizer
def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)

# stop words
nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))

def prepare_text_for_lda(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens

# Load a potentially pretrained model from disk.
dictionary = gensim.corpora.Dictionary.load('data/dictionary.gensim')
corpus = pickle.load(open('data/corpus.pkl', 'rb'))
ldamodel = gensim.models.ldamodel.LdaModel.load('data/model.gensim')

def topic_for_doc(new_doc):
    print("*** topic_for_doc ***")
    print(new_doc)
    new_doc = prepare_text_for_lda(new_doc)
    new_doc_bow = dictionary.doc2bow(new_doc)
    for idx, topic in ldamodel.get_document_topics(new_doc_bow, minimum_probability=0.1):
        print('Topic: ' + str(ldamodel.show_topic(idx, 3)))
        print(topic)
    print("")

topic_for_doc('CRISPR causes significantly greater DNA damage than previously thought, study finds')
topic_for_doc('At least 107 measles cases confirmed across 21 states')
topic_for_doc('Why your mother age could be the key to longevity')
