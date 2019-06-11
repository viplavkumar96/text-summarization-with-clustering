# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 14:32:22 2019

@author: kumavipl
"""
from Utils import importWordEmbeddings
import numpy as np

def getSentenceVector(preprocessedSentences):
    #Importing word embeddings for vector representation
    word_dictionary = importWordEmbeddings()
    #print(len(word_dictionary))
    
    sentence_vectors = []
    for sentence in preprocessedSentences:
      if len(sentence) != 0:
        v = sum([word_dictionary.get(w, np.zeros((100,))) for w in sentence.split()])/(len(sentence.split())+0.001)
      else:
        v = np.zeros((100,))
    
      sentence_vectors.append(v)
    
    return sentence_vectors