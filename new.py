import nltk
import numpy as np
import math
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

df=open("cran.all.1400.txt")
dff=open("cran.qry.txt")
t=False

def read(df):
    data = []
    sent = ''
    for line in df.readlines():  # 한줄씩 읽어서

        if line.startswith(".I"):  # .I나오면 끝난거고
            t = False
        elif line.startswith(".W"):  # .W나오면 다음줄부터 시작이니
            t = True
            continue
        if t:
            sent = sent + line  # 시작되는 줄부터 이어주고
        else:
            if sent != '':
                sent = sent.replace('\n', ' ')  # 쓸데없는 엔터 -> space로 대체하고
                data.append(sent)  # append 해준다.
                sent = ''  # 초기화
    return data
def tokenize(paragraph): #토큰화
   split_sentence=paragraph.split('.')
   toke_word=[]
   for sent in split_sentence:
      toke_word.append(word_tokenize(sent))
   return toke_word[0] #[[a,b,c],[]]이런식으로 2차원배열로 나와서 1차만 호출

def stop_punc_lemm(data): #lemm
    ps = WordNetLemmatizer()
    result=[]
    for data_split in data: #단락을 문장으로 나누고
        #문장을 단어로 쪼개서 stopword
        tokens_without_sw = [word for word in tokenize(data_split) if not word in stopwords.words()]


        punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        for erase in tokens_without_sw: #punc인거 잇으면 지우기
            if erase in punc:
                tokens_without_sw.remove(erase)

        tokens_lemm = [] #temp
        for lemm in tokens_without_sw: #lemm처리
            lemm = ps.lemmatize(lemm)
            tokens_lemm.append(lemm) #temp에 저장
        result.append(tokens_lemm) #결과 저장
    return result

def stop_punc_stem(data): #stem
    ps = PorterStemmer()
    result=[]
    for data_split in data: #단락을 문장으로 나누고
        # 문장을 단어로 쪼개서 stopword
        tokens_without_sw = [ word for word in tokenize(data_split) if not word in stopwords.words()]

        punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        for erase in tokens_without_sw: #punc인거 잇으면 지우기
            if erase in punc:
                tokens_without_sw.remove(erase)

        tokens_stem = [] #temp
        for stem in tokens_without_sw: #stem 처리
            stem = ps.stem(stem)
            tokens_stem.append(stem) #temp에 저장
        result.append(tokens_stem) #결과 저장
    return result

def inverted(data):
    dict = pd.DataFrame() #빈 데이터프레임
    for sent in data: #단락에서 문장 불러오고
        for item in sent: #문장에서 단어 불러와서
            if item not in dict.columns: #컬럼 이름 없으면 만들고
                dict[item]=[1]
            if item in dict.columns: #컬럼 이름 있으면 1 더해준다.
                    dict[item].loc[0] = dict[item].loc[0]+1
    return dict #.transpose() #보기 편하게 행렬 전환

#https://www.geeksforgeeks.org/measuring-the-document-similarity-in-python/
def dotProduct(D1, D2):
    Sum = 0.0

    for key in D1:

        if key in D2:
            Sum += (D1[key] * D2[key])

    return Sum

def cos_sim(D1,D2):
    numerator = dotProduct(D1, D2)
    denominator = math.sqrt(dotProduct(D1, D1) * dotProduct(D2, D2))

    return math.acos(numerator / denominator)

data=read(df)
dfdf=stop_punc_stem(data)

dat=read(dff)
dfdfdf=stop_punc_stem(dat)
print(cos_sim(inverted(dfdf), inverted(dfdfdf)))