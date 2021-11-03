import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import re
import nltk
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('stopwords')
nltk.download('punkt')
Stop_Words = stopwords.words("english")


def tokenize(text):
    clean_txt = re.sub('[^a-z\s]+',' ',text)  # replacing spcl chars, punctuations by space
    clean_txt = re.sub('(\s+)',' ',clean_txt)  # replacing multiple spaces by single space
    min_length = 3
    words = map(lambda word: word.lower(), word_tokenize(clean_txt))  # tokenizing, lowercase
    words = [word for word in words if word not in Stop_Words]  # filtering stopwords
    words = filter(lambda t: len(t)>=min_length, words)  # filtering words of length <=2
    tokens =(list(map(lambda token: PorterStemmer().stem(token),words)))  # stemming tokens
    return tokens


def processD(article):
    article = article.split('\n.T\n')[1]
    T, _, article = article.partition('\n.A\n')
    A, _, article = article.partition('\n.B\n')
    B, _, W = article.partition('\n.W\n')
    return {'T':T, 'A':A, 'B':B, 'W':W}


def process(query):
    query = query.split('\n.W\n')[1]
    W, _, s = query.partition('\n.W\n')
    return W


keyFile = open('cran.qry.txt', 'r')
key = keyFile.readlines()
for ii in key:
    print(ii+'\n')
# 365 queries

responseFile = open('cranqrel.txt', 'r')
response = responseFile.readlines()
for i in response:
    print(i+'\n')
# query_number(1-225)  document_number  relevance_rank(-1-5)

keyF = open('cran.all.1400.txt', 'r')
key2 = keyF.readlines()
for iii in key2:
    print(iii+'\n')

# 각 문서 나눠서 list에 저장
with open('cran.all.1400.txt') as f:
    articles = f.read().split('\n.I')

data = {(i+1):processD(article) for i,article in enumerate(articles)}
docs = [data[index]['W'] for index in range(1, 1401)]

# 각 query 나눠서 list에 저장
with open('cran.qry.txt') as fq:
    queries = fq.read().split('\n.I')

dataq = {(ind+1):process(query) for ind,query in enumerate(queries)}
query = list(dataq.values())

query_tokens = []
for i in query:
    query_tokens.append(tokenize(i))

print(query_tokens[0])
