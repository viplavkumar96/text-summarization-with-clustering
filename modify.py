import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from Utils import readCSV
from Preprocess import removePuncts,makeLowerCase,removeStopWords,removeDuplicate,lemmatizeSentences
from sentenceVectors import getSentenceVector
from Clustering import Clustering
from os import sys
#nltk.set_proxy('http://web-proxy.ind.hp.com:8080')
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

import networkx as nx
#nltk.download('punkt')
import re


categoryData,numCategory = readCSV()


#print("Enter category")
#category = input()
#Split text into sentences
sentences = sent_tokenize(categoryData)

#print(sentences[1:10])




#------------Start of Preprocessing----------------
punctsRemoved = removePuncts(sentences)
#print(punctsRemoved[1:10])

lowerCaseSent = makeLowerCase(punctsRemoved)
#print(lowerCaseSent)

remDuplicateLines = removeDuplicate(lowerCaseSent)
#print(remDuplicateLines)

remStopWords = [removeStopWords(sentence.split()) for sentence in lowerCaseSent]
#print(remStopWords[1:10])

#Sentence Lemmatization
lemSentences = lemmatizeSentences(remDuplicateLines)
#print(lemSentences[1:10])

print("Enter category:")
topic  = input()
topic = topic.lower()


#-----------End of Preprocessing-------------------

#-----------Start of Clustering---------
#lemSentences = np.asarray(lemSentences)

assigned_clusters = Clustering(lemSentences,numCategory)

clusteredSentences = {}
originalClusterd = {}

index = -1
topicFound = False

categoryCnt = {}
for i in range(numCategory):
    clusteredSentences[i] = []
    categoryCnt[i] = 0
for ind, sentence in enumerate(lemSentences):    
    #print (str(assigned_clusters[index]) + ":" + str(sentence))
    if topic in sentence:
        categoryCnt[int(assigned_clusters[ind])] += 1
        clusteredSentences[int(assigned_clusters[ind])].append(sentence)
    #originalClusterd[int(assigned_clusters[ind])].append(sentences[ind])


maxCount = 0
if len(categoryCnt.keys())==0:
    print("Internal error: Clustering not possible on topic '%s'"%(topic))
    sys.exit()
    

    
for key in categoryCnt.keys():
    if maxCount < categoryCnt[key]:
        maxCount = categoryCnt[key]
        index = key
        #print(key,maxCount)
#print("Cluster chosen:",index,maxCount)
#machine print(clusteredSentences[index])   


#for i in range(len(originalClusterd)):
#    for j in range(len(originalClusterd[i])):
#        file.write(str(i)+"    "+originalClusterd[i][j]+"\n")
#        file1.write(str(i)+"    "+clusteredSentences[i][j]+"\n")


sentenceVectors = getSentenceVector(clusteredSentences[index])
#print(sentenceVectors[1])

#-----------End of Clustering-----------

sim_mat = np.zeros([len(clusteredSentences[index]), len(clusteredSentences[index])])

for i in range(len(clusteredSentences[index])):
  for j in range(len(clusteredSentences[index])):
    try:
        if i != j:
          sim_mat[i][j] = cosine_similarity(sentenceVectors[i].reshape(1,100), sentenceVectors[j].reshape(1,100))[0,0]
    except IndexError:
        pass

#Creating graph from similarity matrix
nx_graph = nx.from_numpy_array(sim_mat)


#Applying page rank and calculating score
scores = nx.pagerank(nx_graph)


ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(clusteredSentences[index])), reverse=True)

print("Summary: ")
for i in range(3):
  print(ranked_sentences[i][1].upper(), end = ". ")


  











        