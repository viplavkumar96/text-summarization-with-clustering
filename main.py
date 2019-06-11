
from gensim.models import Word2Vec
from sklearn import cluster
from sklearn import metrics
from nltk.cluster import KMeansClusterer
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from Utils import readCSV
from Preprocess import removePuncts,makeLowerCase,removeStopWords,removeDuplicate,lemmatizeSentences
from sentenceVectors import getSentenceVector
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
#print(sentences)

model = Word2Vec(sentences, min_count=1)
 
  
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
  
  

X=[]
for sentence in sentences:
    X.append(sent_vectorizer(sentence, model))   
#print(X)   
NUM_CLUSTERS = numCategory
kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=,avoid_empty_clusters=True)
assigned_clusters = kclusterer.cluster(X, assign_clusters=True)
#print (assigned_clusters)

clusteredSentences = []
for i in range(NUM_CLUSTERS):
    clusteredSentences.append([])
    
for index, sentence in enumerate(sentences):    
    #print (str(assigned_clusters[index]) + ":" + str(sentence))
    clusteredSentences[int(assigned_clusters[index])].append(sentence)

print(clusteredSentences)

#------------Start of Preprocessing----------------
punctsRemoved = removePuncts(sentences)
#print(punctsRemoved)

lowerCaseSent = makeLowerCase(punctsRemoved)
#print(lowerCaseSent)

remDuplicateLines = removeDuplicate(lowerCaseSent)
#print(removedDuplicateLines)

remStopWords = [removeStopWords(sentence.split()) for sentence in remDuplicateLines]
#print(remStopWords)

#Sentence Lemmatization
lemSentences = lemmatizeSentences(remStopWords)
#print(lemSentences)


#-----------End of Preprocessing-------------------

#-----------Start of Vector representations---------


sentenceVectors = getSentenceVector(lemSentences)
#print(sentenceVectors[1])

#-----------End of Vector representations-----------

sim_mat = np.zeros([len(sentences), len(sentences)])

for i in range(len(sentences)):
  for j in range(len(sentences)):
    try:
        if i != j:
          sim_mat[i][j] = cosine_similarity(sentenceVectors[i].reshape(1,100), sentenceVectors[j].reshape(1,100))[0,0]
    except IndexError:
        pass

#Creating graph from similarity matrix
nx_graph = nx.from_numpy_array(sim_mat)


#Applying page rank and calculating score
scores = nx.pagerank(nx_graph)


ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)

print("Summary: ")
for i in range(2):
  print(ranked_sentences[i][1], end = " ")


  











        