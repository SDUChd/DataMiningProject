#coding=utf-8
import codecs, sys
sys.stdout = codecs.getwriter('utf8')(sys.stdout.buffer)

import pandas as pd


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import nltk

visdata=pd.read_csv("vis_papers.csv")
punct=["'s",',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%','>','<','...','…','’','”','“','‘','\'','..']

papers={}

lema=nltk.WordNetLemmatizer()
s = nltk.stem.snowball.EnglishStemmer()
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
    allnumber=True
    for i in range(len(str)):
        if (not is_number(str[i])) and (not is_alphabet(str[i])):
            return False
        if not is_number(str[i]):
            allnumber=False
    if allnumber:
        return False
    else:
        return True


paperinYear={}
with codecs.open("bongshin Lee Publications.txt","r",encoding='utf-8') as f:
    nowyear=-1
    for r in f:
        isyear=True
        r=r.replace("\n","")
        r=r.replace("\r","")
        if len(r)==0:
            continue
        for i in range(len(r)):
            if not is_number(r[i]):
                isyear=False
                break
        if isyear:
            paperinYear[r]=[]
            nowyear=r
        else:
            text=word_tokenize(text=r, language="english")
            print (text)
            for w in text:
                if w.lower() not in stopwords.words('english') and w not in punct:
                    paperinYear[nowyear].append(lema.lemmatize(w).capitalize())
    print(paperinYear)


paperwordcount={}
for k in paperinYear.keys():
    if(k=='2002' or k=='2003' or k=='2004' or k=='2005' or k=='2006' or k=='2007' or k=='2008'):
        if '2002-2008' not in paperwordcount.keys():
            paperwordcount['2002-2008']={}
        tagged = nltk.pos_tag(paperinYear[k])
        print(tagged)
        for w in tagged:
            if 'JJ' in w[1] or 'NN' in w[1]:
                if w[0] in paperwordcount['2002-2008'].keys():
                    paperwordcount['2002-2008'][w[0]]+=1
                else:
                    paperwordcount['2002-2008'][w[0]]=1
    elif k=='2013' or k=='2014':
        if '2013-2014' not in paperwordcount.keys():
            paperwordcount['2013-2014']={}
        tagged = nltk.pos_tag(paperinYear[k])
        print(tagged)
        for w in tagged:
            if 'JJ' in w[1] or 'NN' in w[1]:
                if w[0] in paperwordcount['2013-2014'].keys():
                    paperwordcount['2013-2014'][w[0]]+=1
                else:
                    paperwordcount['2013-2014'][w[0]]=1
    elif k=='2015' or k=='2016':
        if '2015-2016' not in paperwordcount.keys():
            paperwordcount['2015-2016']={}
        tagged = nltk.pos_tag(paperinYear[k])
        print(tagged)
        for w in tagged:
            if 'JJ' in w[1] or 'NN' in w[1]:
                if w[0] in paperwordcount['2015-2016'].keys():
                    paperwordcount['2015-2016'][w[0]]+=1
                else:
                    paperwordcount['2015-2016'][w[0]]=1
    else:
        paperwordcount[k]={}
        tagged = nltk.pos_tag(paperinYear[k])
        print(tagged)
        for w in tagged:
            if 'JJ' in w[1] or 'NN' in w[1]:
                if w[0] in paperwordcount[k].keys():
                    paperwordcount[k][w[0]]+=1
                else:
                    paperwordcount[k][w[0]]=1

for k in paperwordcount.keys():
    paperwordcount[k]=sorted(paperwordcount[k].items(),key=lambda d: d[1], reverse=True)

paperwordcount=sorted(paperwordcount.items(), key=lambda d: d[0], reverse=False)
print(paperwordcount)






with codecs.open("output/Bongshin Lee Publications.txt",'w',encoding="utf-8") as f:
    for oneyear in paperwordcount:
        f.write(str(oneyear[0]) + "%\n")
        num=0
        for w in oneyear[1]:
            f.write(w[0] + "," + str(w[1]) + "\n")
            num += 1
            if num==50:
                break

        f.write("***")









