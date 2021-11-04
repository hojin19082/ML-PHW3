import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import re, sys, os, nltk, itertools
import math
import numpy as np

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


responseFile = open('cranqrel.txt', 'r')
response = responseFile.readlines()

l_l = [response[c].split() for c in range(len(response))]
lol = [p[:2] for p in l_l]  # lol = listOf [query no. , relevant doc_no.]

qrel_list = []
for i in range(225):
    qrel_list.append([])
    for v in range(len(lol)):
        if lol[v][0] == str(i+1):
            qrel_list[i].append(lol[v][1])


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
                dictionary[item].append(i)

dictionary = sorted(dictionary.items())
dictionary = dict(dictionary)

vectorizer = TfidfVectorizer(tokenizer=tokenize, use_idf=True, sublinear_tf=True, norm='l2')

for k_value in range(4, 7):
    os.system('cls')
    k = 'k-value' + str(k_value) + ".txt"
    sys.stdout = open(k, "w")
    print("When k-value is: " + str(k_value))
    precision = 0
    recall = 0
    i = 0
    for query_token in query_tokens:
        print("Query[" + str(i+1) + "]")
        qvec = query[i]
        qvec = [qvec]
        simi_dict = {}
        for token in query_token:
            if token in dictionary:
                value = dictionary[token]
                for doc_id in value:
                    if doc_id not in simi_dict:
                        simi_dict[doc_id] = 1
                    if doc_id in simi_dict:
                        simi_dict[doc_id] += 1

        doc_list_to_compute_similarity = []
        for key, value in simi_dict.items():
            if value >= k_value:
                doc_list_to_compute_similarity.append(key)

        doc_list_to_compute_similarity = sorted(doc_list_to_compute_similarity)
        if len(doc_list_to_compute_similarity) != 0:
            temp = list()
            for item in doc_list_to_compute_similarity:
                temp.append(docs[item])

            tfid_docs = vectorizer.fit_transform(temp)
            tfid_query = vectorizer.transform(qvec)

            similarity = cosine_similarity(tfid_docs, tfid_query)
            similarity_df = pd.DataFrame(similarity, columns=['cosSim'])
            similarity_df['Document id'] = \
                [doc_list_to_compute_similarity[i]+1 for i in range(len(doc_list_to_compute_similarity))]
            similarity_df.set_index('Document id', inplace=True)
            similarity_df = similarity_df.sort_values(by=['cosSim'], axis=0,  ascending=False)
            print(similarity_df.round(5).head(5))

            list1 = sorted(list(map(int, qrel_list[i])))
            list2 = [doc_list_to_compute_similarity[i]+1 for i in range(len(doc_list_to_compute_similarity))]

            tp = set(list1) & set(list2)
            precision += round(len(tp) / len(list2), 2)
            recall += round(len(tp) / len(list1), 2)

            print("Precision: ", round(len(tp)/len(list2), 2))
            print("Recall:    ", round(len(tp)/len(list1), 2))
            print("\n")

        else:
            print("Precision: ", 0)
            print("Recall:    ", 0)
            print("\n")

        i += 1

    print('Mean precision: ', round((precision/225.0), 4))
    print('Mean recall:    ', round((recall/225.0), 4))

    sys.stdout.close()
