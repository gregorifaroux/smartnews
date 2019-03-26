# import packages
import sys, requests
from pathlib import Path
import hashlib, json, os
import urllib

# Download HTML
from bs4 import BeautifulSoup
from bs4.element import Comment
from tqdm import tqdm

# List of categories to download
CATEGORIES = ["business", "science", "technology"]
BLACKLIST = ["bloomberg", "national-geographic"]
# news API get English sources
def getSources(category: str):
    source_url = f"https://newsapi.org/v1/sources?category={category}&language=en"
    response = requests.get(source_url).json()
    sources = []
    for source in response["sources"]:
        if source["id"] not in BLACKLIST:
            sources.append(source["id"])
    return sources


# news API Category
def mapping():
    d = {}
    response = requests.get("https://newsapi.org/v1/sources?language=en")
    response = response.json()
    for s in response["sources"]:
        d[s["id"]] = s["category"]
    return d


def tag_visible(element):
    if element.parent.name in [
        "style",
        "script",
        "head",
        "title",
        "meta",
        "[document]",
    ]:
        return False
    if isinstance(element, Comment):
        return False
    return True


def text_from_html(body):
    soup = BeautifulSoup(body, "html.parser")
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)
    return " ".join(t.strip() for t in visible_texts)


def get_news_content(url):
    """Download RSS feed content"""
    try:
        page = urllib.request.urlopen(
            urllib.request.Request(
                url,
                data=None,
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36"
                },
            )
        )
        return text_from_html(page.read().decode(page.headers.get_content_charset()))
    except urllib.error.HTTPError as e:
        print(e.reason)
    except urllib.error.URLError as e:
        print(e.reason)


# Download news API feeds
def getNews(category: str):
    sources = getSources(category)
    folder = f"data/{category}"
    if not os.path.exists(folder):
        os.makedirs(folder)
    for source in tqdm(sources):
        response = requests.get(
            f"https://newsapi.org/v1/articles?apiKey={API_KEY}&language=en&source={source}&sortBy=latest"
        )
        r = response.json()
        if r["status"] == "error":
            print(
                f"https://newsapi.org/v1/articles?apiKey={API_KEY}&language=en&source={source}&sortBy=latest"
            )
            print(f"Error: source={source}")
            sys.exit(r["message"])
        for article in r["articles"]:
            # Download URL content
            try:
                article["content"] = get_news_content(article["url"])
                article["category"] = category
                # Generate unique filename
                h = hashlib.new("ripemd160")
                h.update(article["url"].encode("utf-8"))
                with open(f"{folder}/{h.hexdigest()}.json", "w") as outfile:
                    json.dump(article, outfile, ensure_ascii=False)
            except:
                print("Unexpected error:", sys.exc_info()[0])
                print(f"Source: {source} Url was: {article['url']}")


# Check if there is an API key
config = Path("./newsapi.key")
if not config.is_file():
    sys.exit("newsapi.org requires a key: please create a key 'newsapi.key'")
API_KEY = config.read_text().rstrip()
print(f"newsapi.org key is: {API_KEY}")

# Sources in English
sources = []
for category in CATEGORIES:
    sources.extend(getSources(category))
print("number of sources :", len(sources))

# Categories
m = mapping()
for source in sources:
    print(f"{source}: {m[source]}")

# Download news
for category in CATEGORIES:
    print(f"Downloading {category} news")
    getNews(category)

