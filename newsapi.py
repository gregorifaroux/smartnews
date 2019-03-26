# import packages
from __future__ import print_function
import sys
import requests
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from matplotlib import pyplot as plt
from pathlib import Path


def getSources(category: str, lang: str):
    source_url = f"https://newsapi.org/v1/sources?category={category}&language={lang}"
    response = requests.get(source_url).json()
    sources = []
    for source in response["sources"]:
        sources.append(source["id"])
    return sources


def mapping():
    d = {}
    response = requests.get("https://newsapi.org/v1/sources?language=en")
    response = response.json()
    for s in response["sources"]:
        d[s["id"]] = s["category"]
    return d


# Check if there is an API key
config = Path("./newsapi.key")
if not config.is_file():
    sys.exit("newsapi.org requires a key: please create a key 'newsapi.key'")
key = config.read_text()
print(f"newsapi.org key is: {key}")

# Sources in English
sources = getSources("business", "en")
sources.extend(getSources("health", "en"))
sources.extend(getSources("science", "en"))
sources.extend(getSources("technology", "en"))
print("number of sources :", len(sources))

# Categories
m = mapping()
for source in sources:
    print(f"{source}: {m[source]}")

