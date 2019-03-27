import os, sys, json

#
import pandas as pd
from datetime import datetime


"""
    Merge news to a csv file
"""


def main():

    dirName = "data/"

    # Get the list of all files in directory tree at given path
    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk(dirName):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]

    # Process the file content
    responses = []
    for elem in listOfFiles:
        print(elem)
        with open(elem, "r") as read_file:
            try:
                data = json.load(read_file)
                responses.append(data)
            except:
                print("Unexpected error:", sys.exc_info()[0])

    news = pd.DataFrame(responses)
    news = news.dropna()
    news = news.drop_duplicates()
    news.reset_index(inplace=True, drop=True)
    news["scraping_date"] = datetime.now()
    print(news)

    try:
        aux = pd.read_csv("./data/news.csv")
        aux = aux.append(news)
        aux = aux.drop_duplicates("url")
        aux.reset_index(inplace=True, drop=True)
        aux.to_csv("./data/news.csv", encoding="utf-8", index=False)
    except:
        news.to_csv("./data/news.csv", index=False, encoding="utf-8")

    print("Done")


if __name__ == "__main__":
    main()
