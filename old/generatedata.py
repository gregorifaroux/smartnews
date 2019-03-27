TOPICS = 30


# import libraries
import urllib.request
import pickle

# Topic Modeling
# 
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.filterwarnings("ignore", message="For a faster implementation")

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
import operator
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel, LdaModel
from gensim.test.utils import datapath

# spacy for lemmatization
import spacy

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

def lemmatization(texts, allowed_postags=['NOUN']):
#def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

# Get documents

print('1. Get content')
documents = []
#documents = pickle.load(open('data/sciencedocuments.pkl', 'rb'))
documents = pickle.load(open('data/newsdocuments.pkl', 'rb'))
#documents = pickle.load(open('data/googledocuments.pkl', 'rb'))


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
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN'])

print(data_lemmatized[:1])

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
print([[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]])

# Build LDA model
print("6. Build LDA model")
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=TOPICS, 
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

pprint("Top Topics")
#pprint(lda_model.show_topics(num_topics=20, num_words=1, log=False, formatted=False))
#pprint(lda_model.top_topics(corpus, topn=1))

for topic in lda_model.top_topics(corpus, topn=1):
  pprint('Topic: ' + str(topic[0][0][1]))


doc_lda = lda_model[corpus]

# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

def ret_top_model():
    """
    Since LDAmodel is a probabilistic model, it comes up different topics each time we run it. To control the
    quality of the topic model we produce, we can see what the interpretability of the best topic is and keep
    evaluating the topic model until this threshold is crossed. 
    
    Returns:
    -------
    lm: Final evaluated topic model
    top_topics: ranked topics in decreasing order. List of tuples
    """
    top_topics = [(0, 0)]
    while top_topics[0][1] < 0.55:
        print('\ntop_topics[0][1]')
        print(top_topics[0][1])
        lm = LdaModel(corpus=corpus, id2word=id2word)
        coherence_values = {}
        for n, topic in lm.show_topics(num_topics=-1, formatted=False):
            topic = [word for word, _ in topic]
            cm = CoherenceModel(topics=[topic], texts=data_lemmatized, dictionary=id2word, window_size=10)
            coherence_values[n] = cm.get_coherence()
            print(coherence_values[n])
        top_topics = sorted(coherence_values.items(), key=operator.itemgetter(1), reverse=True)
    return lm, top_topics
    
print('\n control the quality of the topic model we produce')
lm, top_topics = ret_top_model()
print(top_topics[:5])
pprint([lm.show_topic(topicid) for topicid, c_v in top_topics[:10]])
lda_lsi_topics = [[word for word, prob in lm.show_topic(topicid)] for topicid, c_v in top_topics]
lda_lsi_coherence = CoherenceModel(topics=lda_lsi_topics[:10], texts=data_lemmatized, dictionary=id2word, window_size=10).get_coherence()
print('\nCoherence Score LDA LSI: ', lda_lsi_coherence)


