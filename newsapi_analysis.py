import os, sys, json
from string import punctuation
import re

# NLTK Stop words
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from functools import reduce

stop_words = stopwords.words("english")

"""
    Analyze all the news downloaded from newsapi
"""


def getListOfFiles(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles


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
    tokens = list(
        filter(lambda token: token not in (stop_words + list(punctuation)), tokens)
    )
    return tokens


def main():

    dirName = "data/"

    # Get the list of all files in directory tree at given path
    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk(dirName):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]

    # Process the file content
    content = {}
    content["description"] = {}
    content["tokens"] = {}
    for elem in listOfFiles:
        print(elem)
        with open(elem, "r") as read_file:
            try:
                data = json.load(read_file)
                data["tokens"] = tokenizer(data["description"])
                content["description"].update(data["description"])
                content["tokens"].update(data["tokens"])
            except:
                print("Unexpected error:", sys.exc_info()[0])

    for descripition, tokens in zip(
        content["description"].head(5), content["tokens"].head(5)
    ):
        print("description:", descripition)
        print("tokens:", tokens)
        print()


if __name__ == "__main__":
    main()
