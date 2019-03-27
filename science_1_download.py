# import packages
import sys, requests
from pathlib import Path
import hashlib, json, os
import urllib
from urllib.parse import urlparse

import feedparser, pickle, urllib.request
from bs4 import BeautifulSoup

# Download HTML
from bs4 import BeautifulSoup
from tqdm import tqdm


def sciencedaily_text_from_html(body):
    soup = BeautifulSoup(body, "html.parser")
    for div in soup.find_all("div", id="text"):
        return div.get_text()
    return ""


def get_sciencedaily_documents(url):
    """Download Science Daily RSS feed links content to populate the list """
    d = feedparser.parse(url)
    # Generate domain based on the feed URL and drop the .xml
    category = os.path.basename(os.path.dirname(url))
    folder = f"data/{category}"
    if not os.path.exists(folder):
        os.makedirs(folder)
    for entry in tqdm(d["entries"]):
        try:
            # Generate unique filename
            h = hashlib.new("ripemd160")
            h.update(entry["link"].encode("utf-8"))
            filename = f"{folder}/{h.hexdigest()}.json"
            if Path(filename).exists():
                continue
            # Download content
            page = urllib.request.urlopen(
                urllib.request.Request(
                    entry.link,
                    data=None,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36"
                    },
                )
            )
            json_entry = {}
            json_entry["title"] = entry.title
            json_entry["link"] = entry.link
            json_entry["description"] = entry.description
            json_entry["title"] = entry.title
            json_entry["content"] = sciencedaily_text_from_html(page)

            # Save RSS entry as a JSON file
            with open(filename, "w") as outfile:
                json.dump(entry, outfile)

        except urllib.error.HTTPError as e:
            print(e.reason)
        except urllib.error.URLError as e:
            print(e.reason)


print("1. Get content")

# Science Daily
with open("rss_science.txt") as f:
    for url in f:
        print(url)
        get_sciencedaily_documents(url)

