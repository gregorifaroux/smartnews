import feedparser, pickle, urllib.request
from bs4 import BeautifulSoup

# Download science daily RSS feeds

def sciencedaily_text_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')
    for div in soup.find_all('div', id='text'):
      return(div.get_text())
    return ""

def get_sciencedaily_documents(list, url):
  """Download Science Daily RSS feed links content to populate the list """
  d = feedparser.parse(url)
  for entry in d['entries']:
    try:
      page = urllib.request.urlopen(urllib.request.Request(entry.link, data=None, headers={ 'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'}))
      print('  ' + entry.title)
      print("")
      list.append(sciencedaily_text_from_html(page))
    except urllib.error.HTTPError as e:
      print(e.reason)
    except urllib.error.URLError as e:
      print(e.reason)

print('1. Get content')
documents = []

#Science Daily
with open('rss_science.txt') as f:
   for url in f:
       print(url)
       get_sciencedaily_documents(documents, url)

pickle.dump(documents, open('data/sciencedocuments.pkl', 'wb'))

