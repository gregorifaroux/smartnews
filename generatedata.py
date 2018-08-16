# import libraries
import urllib.request
from bs4 import BeautifulSoup
from bs4.element import Comment
import feedparser
import pickle

# Topic Modeling
# 
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

#
import numpy as np
from io import StringIO
from pprint import pprint
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn
import os
import os.path
import wget
import zipfile
import tarfile,sys

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.test.utils import datapath

# spacy for lemmatization
import spacy


# Download documents
def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

def text_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)  
    return u" ".join(t.strip() for t in visible_texts)

def sciencedaily_text_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')
    for div in soup.find_all('div', id='text'):
      return(div.get_text())
    return ""

def get_google_documents(list, url):
  """Download a Google RSS feed links content to populate the list """
  d = feedparser.parse(url)
  for entry in d['entries']:
    try:
      page = urllib.request.urlopen(entry.link)
      print('  ' + entry.title)
      print("")
      list.append(text_from_html(page))
    except urllib.error.HTTPError as e:
      print(e.reason)
    except urllib.error.URLError as e:
      print(e.reason)

def get_sciencedaily_documents(list, url):
  """Download Science Daily RSS feed links content to populate the list """
  d = feedparser.parse(url)
  for entry in d['entries']:
    try:
      page = urllib.request.urlopen(urllib.request.Request(entry.link, data=None, headers={ 'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'}))
      print('  ' + entry.title)
      print("")
      list.append(sciencedaily_text_from_html(page))
    except urllib.error.HTTPError as e:
      print(e.reason)
    except urllib.error.URLError as e:
      print(e.reason)

# Process text
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

# Get documents

print('1. Get content')
documents = []

#Science Daily
with open('rss_science.txt') as f:
   for url in f:
       print(url)
       get_sciencedaily_documents(documents, url)


#Google News
with open('rss_google_words.txt') as f:
   for word in f:
       print(word)
       get_google_documents(documents, 'https://news.google.com/news/rss/search/section/q/' + word + '?ned=us&gl=US&hl=en')

print('Number of documents ' + str(len(documents)))

# Tokenize words and Clean-up text

print("2. Tokezine words")

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(documents))

#print(data_words[:1])

# Creating Bigram and Trigram Models
# Build the bigram and trigram models
print("3. Trigram example")
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
#print(trigram_mod[bigram_mod[data_words[0]]])

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

#print(data_lemmatized[:1])

# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
print("5. View corpus")
corpus = [id2word.doc2bow(text) for text in texts]

#save the dictionary and corpus for future use.
pickle.dump(corpus, open('data/corpus.pkl', 'wb'))
id2word.save('data/dictionary.gensim')

# View
#print(corpus[:1])

# Human readable format of corpus (term-frequency)
#print([[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]])

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
# Save model to disk.
lda_model.save('data/model.gensim')

# Print the Keyword in the topics
#print("Print the Keyword in the topics")
#pprint(lda_model.print_topics())
#print("Print the Keyword in the topics ; num_words = 1")
#pprint(lda_model.print_topics(num_words=1))
doc_lda = lda_model[corpus]

# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

pprint("Top Topics")
#pprint(lda_model.show_topics(num_topics=20, num_words=1, log=False, formatted=False))
#pprint(lda_model.top_topics(corpus, topn=1))

for topic in lda_model.top_topics(corpus, topn=1):
  pprint('Topic: ' + str(topic[0][0][1]))

