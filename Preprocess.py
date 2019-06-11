# -*- coding: utf-8 -*-
import nltk
nltk.set_proxy('http://web-proxy.ind.hp.com:8080')
#nltk.download('stopwords') 
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from nltk.stem import PorterStemmer 
#nltk.download('wordnet')
   
stop_words = stopwords.words('english')


def removePuncts(sentences):
    return pd.Series(sentences).str.replace("[^a-zA-Z]", " ")


def makeLowerCase(punctsRemoved):
    lowerCase = [s.lower() for s in punctsRemoved]
    #Remove addtional white spaces
    return [s.strip() for s in lowerCase]


def removeStopWords(sen):
    noStopWords = " ".join([i for i in sen if i not in stop_words])
    return noStopWords


def removeDuplicate(sentences):
    lines_seen = list()
    
    for line in sentences:
        if (line not in lines_seen) and not(len(line)==0):
            lines_seen.append(line)
        else:
            pass
            print("Duplicate line: '"+line,"' is removed.")
    return lines_seen

def lemmatizeSentences(sentArray):
    
    lemmatizer = WordNetLemmatizer()
    
            
    lemSentences = []
    for sent in sentArray:
        if len(sent)!=0:
            cleanOne = " ".join(lemmatizer.lemmatize(word) for word in sent.split())
            lemSentences.append(cleanOne)
        
    return lemSentences
    
    