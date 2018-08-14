# Topic Modeling
# 
import numpy as np
import pandas as pd
import feedparser
from io import StringIO
from pprint import pprint
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn
import os
import os.path
import wget
import zipfile

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

def get_lemma(word):
    """Use NLTK’s Wordnet to find the meanings of words, synonyms, antonyms, and more"""
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma

# Root word
def get_root_word(word):
    """Use WordNetLemmatizer to get the root word"""
    return WordNetLemmatizer().lemmatize(get_lemma(word))

# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[get_root_word(word) for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

# Get Sample data
PATH='data/million-headlines.zip'
if not os.path.isfile(PATH):
    pprint("File does not exist")
    wget.download('https://www.kaggle.com/therohk/million-headlines/downloads/million-headlines.zip/6', PATH)
    zip_ref = zipfile.ZipFile(PATH, 'r')
    zip_ref.extractall('data/')
    zip_ref.close()

# Import Dataset
#d = feedparser.parse('https://news.google.com/news/rss/search/section/q/life%20science?ned=us&gl=US&hl=en')
#d = feedparser.parse('https://news.google.com/news/rss/')

#titles = StringIO()
#print('### Titles ###')
#for entry in d['entries']:
#  print(entry.title)
#  titles.write(entry.title)
#  titles.write('=')

print('1. Read Title')
#output = StringIO(titles.getvalue())
#df = pd.read_csv(output, sep='~', header=None)
#output.close()
df = pd.read_csv('data/data/million-headlines.csv')
print(df.head())

# Convert to list
data = df.values.tolist()

# Tokenize words and Clean-up text
print("2. Tokezine words")

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(data))

print(data_words[:1])

# Creating Bigram and Trigram Models
# Build the bigram and trigram models
print("3. Trigram example")
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
print(trigram_mod[bigram_mod[data_words[0]]])

# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load('en', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
print("4. Lemmatized")
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print(data_lemmatized[:1])

# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
print("5. View corpus")
corpus = [id2word.doc2bow(text) for text in texts]

# View
print(corpus[:1])

# Human readable format of corpus (term-frequency)
print([[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]])

# Build LDA model
print("6. Build LDA model")
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=20, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)

# Print the Keyword in the 10 topics
print("Print the Keyword in the 10 topics")
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]

# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)