import os, sys, json

# NLTK Stop words
from nltk.corpus import stopwords

stop = stopwords.words("english")
from string import punctuation
from nltk.tokenize import word_tokenize, sent_tokenize

#
import pandas as pd
from functools import reduce

pd.options.display.max_columns = 200
pd.options.mode.chained_assignment = None
from datetime import datetime


from collections import Counter
import re
import numpy as np

import matplotlib

matplotlib.use("TkAgg", warn=True, force=False)
print("Using: " + matplotlib.get_backend())
import matplotlib.pyplot as plt


#
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

# Perform LDA

import gensim
import gensim.corpora as corpora
from gensim import matutils
from gensim.models import CoherenceModel


"""
    Data discovery
"""


def _removeNonAscii(s):
    return "".join(i for i in s if ord(i) < 128)


def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = text.replace("(ap)", "")
    text = re.sub(r"\'s", " is ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\W+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\\", "", text)
    text = re.sub(r"\'", "", text)
    text = re.sub(r"\"", "", text)
    text = re.sub("[^a-zA-Z ?!]+", "", text)
    text = _removeNonAscii(text)
    text = text.strip()
    return text


def tokenizer(text):
    text = clean_text(text)
    tokens = [word_tokenize(sent) for sent in sent_tokenize(text)]
    tokens = list(reduce(lambda x, y: x + y, tokens))
    tokens = list(filter(lambda token: token not in (stop + list(punctuation)), tokens))
    return tokens


def keywords(data, category):
    tokens = data[data["category"] == category]["tokens"]
    alltokens = []
    for token_list in tokens:
        alltokens += token_list
    counter = Counter(alltokens)
    return counter.most_common(10)


def plot_word_cloud(terms):
    text = terms.index
    text = " ".join(list(text))
    # lower max_font_size
    wordcloud = WordCloud(max_font_size=40).generate(text)
    plt.figure(figsize=(25, 25))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


def LDA_model(corpus, id2word, num_topics, passes=1):
    return gensim.models.ldamodel.LdaModel(
        corpus=corpus,
        id2word=id2word,
        num_topics=num_topics,
        random_state=100,
        eval_every=10,
        chunksize=2000,
        passes=passes,
        per_word_topics=True,
    )


def compute_coherence(model, aux, id2word):
    coherence = CoherenceModel(
        model=model,
        texts=aux["tokens_bigram"].values,
        dictionary=id2word,
        coherence="c_v",
    )
    return coherence.get_coherence()


def display_topics(model):
    topics = model.show_topics(
        num_topics=model.num_topics, formatted=False, num_words=10
    )
    topics = map(lambda c: map(lambda cc: cc[0], c[1]), topics)
    df = pd.DataFrame(topics)
    df.index = ["topic_{0}".format(i) for i in range(model.num_topics)]
    df.columns = ["keyword_{0}".format(i) for i in range(1, 10 + 1)]
    return df


def explore_models(df, rg=range(5, 25)):
    id2word = corpora.Dictionary(df["tokens_bigram"])
    texts = df["tokens_bigram"].values
    corpus = [id2word.doc2bow(text) for text in texts]
    models = []
    coherences = []

    for num_topics in rg:
        lda_model = LDA_model(corpus, id2word, num_topics, passes=5)
        models.append(lda_model)
        coherence = compute_coherence(lda_model, df, id2word)
        coherences.append(coherence)

    fig = plt.figure(figsize=(15, 5))
    plt.title("Choosing the optimal number of topics")
    plt.xlabel("Number of topics")
    plt.ylabel("Coherence")
    plt.grid(True)
    plt.plot(rg, coherences)

    return coherences, models


def get_document_topic_matrix(corpus, best_model, num_topics):
    matrix = []
    for row in corpus:
        output = np.zeros(num_topics)
        doc_proba = best_model[row][0]
        for doc, proba in doc_proba:
            output[doc] = proba
        matrix.append(output)
    matrix = np.array(matrix)
    return matrix


def main():

    print("Data discovery ... ")
    data = pd.read_csv("./data/news.csv")
    print(data.shape)
    print(data.head(3))
    data.category.value_counts(normalize=True).plot(
        kind="bar", grid=True, figsize=(16, 9)
    )
    pdf_plot = (
        data.category.value_counts(normalize=True)
        .plot(kind="bar", grid=True, figsize=(16, 9))
        .get_figure()
    )
    pdf_plot.savefig("data/category.pdf")
    #   plt.xticks(rotation=25)
    #   plt.show()

    # process the descriptions of each article
    data = data.drop_duplicates("description")
    data = data[~data["description"].isnull()]
    data = data[(data.description.map(len) > 140) & (data.description.map(len) <= 300)]
    data.reset_index(inplace=True, drop=True)
    data.description.map(len).hist(figsize=(15, 5), bins=100)
    pdf_plot = data.description.map(len).hist(figsize=(15, 5), bins=100).get_figure()
    pdf_plot.savefig("data/description.pdf")
    #    plt.show()

    print("Text processing : tokenization")
    data["tokens"] = data["description"].map(lambda d: tokenizer(d))

    for description, tokens in zip(data["description"].head(5), data["tokens"].head(5)):
        print("description:", description)
        print("tokens:", tokens)
        print()

    for category in set(data["category"]):
        print("category :", category)
        print("top 10 keywords:", keywords(data, category))
        print("---")

    print("Term frequencey-inverse document frequency")
    vectorizer = TfidfVectorizer(
        min_df=5, analyzer="word", ngram_range=(1, 2), stop_words="english"
    )
    vz = vectorizer.fit_transform(
        list(data["tokens"].map(lambda tokens: " ".join(tokens)))
    )
    print(vz.shape)
    tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
    tfidf = pd.DataFrame(columns=["tfidf"]).from_dict(dict(tfidf), orient="index")
    tfidf.columns = ["tfidf"]

    #   plot_word_cloud(tfidf.sort_values(by=["tfidf"], ascending=True).head(40))
    #  plot_word_cloud(tfidf.sort_values(by=["tfidf"], ascending=False).head(40))

    print(" topic modeling - LDA")
    aux = data.copy()

    bigram = gensim.models.Phrases(aux["tokens"], min_count=5, threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    aux["tokens_bigram"] = aux["tokens"].map(lambda tokens: bigram_mod[tokens])
    id2word = corpora.Dictionary(aux["tokens_bigram"])
    texts = aux["tokens_bigram"].values
    corpus = [id2word.doc2bow(text) for text in texts]
    coherences, models = explore_models(aux, rg=range(5, 85, 5))
    best_model = LDA_model(corpus, id2word, num_topics=55, passes=5)
    print(display_topics(model=best_model))

    matrix = get_document_topic_matrix(corpus, best_model, best_model.num_topics)
    doc_topic = best_model.get_document_topics(corpus)
    lda_keys = []
    for i, desc in enumerate(data["description"]):
        lda_keys.append(np.argmax(matrix[i, :]))
        print(desc)
        print(best_model.show_topic(np.argmax(matrix[i, :])))
        print()
    #    plt.show()

    print("Done")


if __name__ == "__main__":
    main()
