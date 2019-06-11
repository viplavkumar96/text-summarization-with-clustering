# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 16:31:00 2019

@author: kumavipl
"""
import numpy as np

def readCSV():
    import csv
    
    csvFile = open("dataset.csv","r")
    dataset = csv.reader(csvFile)
    
    #Store articles by category
    categoryData = ""
    topicSet = set()
    
    #Reading each article and categorize in topics
    for eachRow in dataset:
        topic = eachRow[3]
        topicSet.add(topic)
        article = eachRow[1]
        #if topic not in categoryData.keys() :
        #    categoryData[topic] = article
        #else:
        categoryData = categoryData +(article)
    
    #print(len(categoryData))
    
    #print(categoryData)
    
    csvFile.close()
    #categoryData.pop('topic')
    
    return categoryData,len(topicSet)


def importWordEmbeddings():
    word_embeddings = {}
    
    embeddingFile = open("glove.6B.100d.txt",encoding = "utf-8")
    
    for line in embeddingFile:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    
    embeddingFile.close()
    return word_embeddings

