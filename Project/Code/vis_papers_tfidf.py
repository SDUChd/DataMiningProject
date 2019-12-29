import codecs as  cs
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


visdata=pd.read_csv("vis_papers.csv")
punct=[',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%','>','<','...','…','’','”','“','‘','\'','..']

papers={}


def is_number(uchar):
    """判断一个unicode是否是数字"""
    if uchar >= u'\u0030' and uchar <= u'\u0039':
        return True
    else:
        return False


def is_alphabet(uchar):
    """判断一个unicode是否是英文字母"""
    if (uchar >= u'\u0041' and uchar <= u'\u005a') or (uchar >= u'\u0061' and uchar <= u'\u007a'):
        return True
    else:
        return False


def isValid(str):
    for i in range(len(str)):
        if (not is_number(str[i])) and (not is_alphabet(str[i])):
            return False
    return True

abstracts=[]

for index,row in visdata.iterrows():
    if row['Conference']=='InfoVis' and not row['Paper type: C=conference paper, J = journal paper, M=miscellaneous (capstone, keynote, VAST challenge, panel, poster, ...)']=='M':
        if not row['Year'] in papers.keys():
            papers[row['Year']]=[]

        abstract=row['Abstract']
        #print(abstract)
        if abstract!="":
            abstracts.append(abstract)

print(abstracts)


tfidf2 = TfidfVectorizer()
tfidf2.fit_transform(abstracts)
print(tfidf2.get_feature_names())
print (tfidf2.fit_transform(abstracts).toarray())


