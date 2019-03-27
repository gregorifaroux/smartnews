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

import matplotlib.pyplot as plt

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
    data["tokens"] = data["content"].map(lambda d: tokenizer(d))

    for descripition, tokens in zip(data["content"].head(5), data["tokens"].head(5)):
        print("description:", descripition)
        print("tokens:", tokens)
        print()

    for category in set(data["category"]):
        print("category :", category)
        print("top 10 keywords:", keywords(data, category))
        print("---")

    print("Done")


if __name__ == "__main__":
    main()
