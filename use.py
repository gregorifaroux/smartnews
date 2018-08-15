from gensim.test.utils import datapath

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.test.utils import datapath


# Load a potentially pretrained model from disk.
lda_model = LdaModel.load(datapath("model"))

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result
    

unseen_document = 'How a Pentagon deal became an identity crisis for Google'
bow_vector = gensim.corpora.Dictionary.doc2bow(preprocess(unseen_document))
for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))