# import packages
from __future__ import print_function
import requests
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from matplotlib import pyplot as plt


def getSources(category: str, lang: str):
    source_url = f"https://newsapi.org/v1/sources?category={category}&language={lang}"
    response = requests.get(source_url).json()
    sources = []
    for source in response["sources"]:
        sources.append(source["id"])
    return sources

def mapping():
    d = {}
    response = requests.get('https://newsapi.org/v1/sources?language=en')
    response = response.json()
    for s in response['sources']:
        d[s['id']] = s['category']
    return d


# Sources in English
sources = getSources("business", "en")
sources.extend(getSources("health", "en"))
sources.extend(getSources("science", "en"))
sources.extend(getSources("technology", "en"))
print("number of sources :", len(sources))

# Categories
m = mapping()
for source in sources:
    print(f'{source}: {m[source]}')

