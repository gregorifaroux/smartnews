import feedparser, pickle, urllib.request
from bs4 import BeautifulSoup
from bs4.element import Comment

# Download Google RSS feeds
# Download documents
def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

def text_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)  
    return u" ".join(t.strip() for t in visible_texts)

def get_google_documents(list, url):
  """Download a Google RSS feed links content to populate the list """
  d = feedparser.parse(url)
  for entry in d['entries']:
    try:
      page = urllib.request.urlopen(urllib.request.Request(entry.link, data=None, headers={ 'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'}))
#      page = urllib.request.urlopen(entry.link)
      print('  ' + entry.title)
      print("")
      list.append(text_from_html(page))
    except urllib.error.HTTPError as e:
      print(e.reason)
    except urllib.error.URLError as e:
      print(e.reason)



print('1. Get content')
documents = []

#Google News
with open('rss_google_words.txt') as f:
   for word in f:
       print(word)
       get_google_documents(documents, 'https://news.google.com/news/rss/search/section/q/' + word + '?ned=us&gl=US&hl=en')


pickle.dump(documents, open('data/googledocuments.pkl', 'wb'))

