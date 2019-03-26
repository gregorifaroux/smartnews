# import packages
from __future__ import print_function
import sys
import requests
import pandas as pd
from datetime import datetime
from tqdm import tqdm, tqdm_notebook
from matplotlib import pyplot as plt
from pathlib import Path


def getSources(category: str):
    source_url = f"https://newsapi.org/v1/sources?category={category}&language=en&apiKey={API_KEY}"
    response = requests.get(source_url).json()
    sources = []
    for source in response["sources"]:
        sources.append(source["id"])
    return sources


def mapping():
    d = {}
    response = requests.get(
        "https://newsapi.org/v1/sources?language=en&apiKey={API_KEY}"
    )
    response = response.json()
    for s in response["sources"]:
        d[s["id"]] = s["category"]
    return d


def getNews(category: str):
    sources = getSources(category)
    responses = []
    for i, source in tqdm_notebook(enumerate(sources), total=len(sources)):
        print(
            f"https://newsapi.org/v1/articles?apiKey={API_KEY}&language=en&source={source}&sortBy=latest"
        )
        response = requests.get(
            f"https://newsapi.org/v1/articles?apiKey={API_KEY}&language=en&source={source}&sortBy=latest"
        )
        r = response.json()
        print(r)
        if r["status"] == "error":
            sys.exit(r["message"])
        for article in r["articles"]:
            article["source"] = source
            responses.append(r)
            print(r)


# Check if there is an API key
config = Path("./newsapi.key")
if not config.is_file():
    sys.exit("newsapi.org requires a key: please create a key 'newsapi.key'")
API_KEY = config.read_text().rstrip()
print(f"newsapi.org key is: {API_KEY}")

# Sources in English
sources = getSources("business")
sources.extend(getSources("health"))
sources.extend(getSources("science"))
sources.extend(getSources("technology"))
print("number of sources :", len(sources))

# Categories
m = mapping()
for source in sources:
    print(f"{source}: {m[source]}")

# Download news
getNews("business")

