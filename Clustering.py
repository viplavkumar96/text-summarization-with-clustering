# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 11:52:29 2019

@author: kumavipl
"""
from gensim.models import Word2Vec
from sklearn import cluster
from sklearn import metrics
from nltk.cluster import KMeansClusterer
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
 
  
def sent_vectorizer(sent, model):
    sent_vec =[]
    numw = 0
    for w in sent:
        try:
            if numw == 0:
                sent_vec = model[w]
            else:
                sent_vec = np.add(sent_vec, model[w])
            numw+=1
        except:
            pass
     
    return np.asarray(sent_vec) / numw
  
  
def Clustering(lemSentences,numCategory):
    model = Word2Vec(lemSentences, min_count=1)
    X=[]
    for sentence in lemSentences:
        X.append(sent_vectorizer(sentence, model))  
        
    #print(X)   
    NUM_CLUSTERS = numCategory
    
    kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25,avoid_empty_clusters=True)
    
    assigned_clusters = kclusterer.cluster(X, assign_clusters=True)
    
    return assigned_clusters

#print (assigned_clusters)