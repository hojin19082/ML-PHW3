import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import re
import numpy as np
import nltk
import itertools

"""
nltk.download('stopwords')
nltk.download('punkt')
"""
Stop_Words = stopwords.words("english")


def tokenize(text):
    clean_txt = re.sub('[^a-z\s]+', ' ', text)  # replacing spcl chars, punctuations by space
    clean_txt = re.sub('(\s+)', ' ', clean_txt)  # replacing multiple spaces by single space
    min_length = 3
    words = map(lambda word: word.lower(), word_tokenize(clean_txt))  # tokenizing, lowercase
    words = [word for word in words if word not in Stop_Words]  # filtering stopwords
    words = filter(lambda t: len(t) >= min_length, words)  # filtering words of length <=2
    tokens =(list(map(lambda token: PorterStemmer().stem(token), words)))  # stemming tokens
    return tokens


def processD(article):
    article = article.split('\n.T\n')[1]
    T, _, article = article.partition('\n.A\n')
    A, _, article = article.partition('\n.B\n')
    B, _, W = article.partition('\n.W\n')
    return {'T': T, 'A': A, 'B': B, 'W': W}


def process(query):
    query = query.split('\n.W\n')[1]
    W, _, s = query.partition('\n.W\n')
    return W

'''
keyFile = open('cran.qry.txt', 'r')
key = keyFile.readlines()
for ii in key:
    print(ii+'\n')

responseFile = open('cranqrel.txt', 'r')
response = responseFile.readlines()
for i in response:
    print(i+'\n')
# query_number(1-225)  document_number  relevance_rank(-1-5)

keyF = open('cran.all.1400.txt', 'r')
key2 = keyF.readlines()
for iii in key2:
    print(iii+'\n')
'''

# 각 문서 나눠서 list에 저장
with open('cran.all.1400.txt') as f:
    articles = f.read().split('\n.I')

data = {(i+1): processD(article) for i, article in enumerate(articles)}
docs = [data[index]['W'] for index in range(1, 1401)]

# 각 문서 토큰화 후 list에 저장
docs_tokens = []
for i in docs:
    docs_tokens.append(tokenize(i))

# 각 쿼리 나누기
with open('cran.qry.txt') as fq:
    queries = fq.read().split('\n.I')


dataq = {(ind+1): process(query) for ind, query in enumerate(queries)}
query = list(dataq.values())
query_tokens = []
# 쿼리 token화
for i in query:
    query_tokens.append(tokenize(i))

total_query_tokens = []
# 전체 쿼리 하나의 list에 넣고 중복 제거. (inverted index용 )
total_query = list(itertools.chain.from_iterable(query_tokens))
for v in total_query:
    if v not in total_query_tokens:
        # total_query_tokens에 전체 쿼리에 대한 token이 하나의 list에 들어가 있음.
        total_query_tokens.append(v)

# Inverted Index
dictionary = {}
for i in range(len(docs)):
    check = docs_tokens[i]
    for item in total_query_tokens:
        if item in check:
            if item not in dictionary:
                dictionary[item] = []
            if item in dictionary:
                dictionary[item].append(i+1)

dictionary = sorted(dictionary.items())
dictionary = dict(dictionary)


vectorizer = CountVectorizer(tokenizer=tokenize, min_df=1,max_df=0.5, ngram_range=(1,2))
transformer = TfidfTransformer(smooth_idf=False)

i = 0
for query_token in query_tokens:
    qvec = query[i]
    simi_dict = {}
    if i == 5:
        break
    for token in query_token:
        value = dictionary[token]
        for doc_id in value:
            if doc_id not in simi_dict:
                simi_dict[doc_id] = 1
            if doc_id in simi_dict:
                simi_dict[doc_id] += 1

    doc_list_to_compute_similarity = []
    for key, value in simi_dict.items():
        if value >= 5:
            doc_list_to_compute_similarity.append(key)

    doc_list_to_compute_similarity = sorted(doc_list_to_compute_similarity)

    temp= list()
    for item in doc_list_to_compute_similarity:
        temp.append(docs[item])

    tfid_docs = vectorizer.fit_transform(temp)
    qvec = vectorizer.transform(query)
    tfid_docs_weight = transformer.fit_transform(tfid_docs)
    qvec_weight = transformer.fit_transform(qvec)

    similarity = cosine_similarity(tfid_docs_weight, qvec_weight)
    print(similarity)
    i += 1