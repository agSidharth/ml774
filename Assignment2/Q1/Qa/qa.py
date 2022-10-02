# Importing libraries...
import math
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import re
from wordcloud import WordCloud, STOPWORDS

training_dir = str(sys.argv[1])
testing_dir  = str(sys.argv[2])

# Initializing variables..
wordFreq = {}
wordFreq[0] = {}
wordFreq[1] = {}

totalFreq = {}
totalFreq[0] = 0
totalFreq[1] = 0

totalDocs = {}
totalDocs[0] = 0
totalDocs[1] = 1

alpha = 1
posCloudText = ""
negCloudText = ""

# Training the model
def trainModel(path,label,wordFreq,totalFreq,totalDocs):
    
    listOfFiles = os.listdir(path)
    for fileName in listOfFiles:
        
        file = open(os.path.join(path,fileName),"r")
        text = file.read()
        word_list = re.split(' |,|\\.|\n|:|;|"|\'|`|{{|}}|[|]|\)|\(',text)
        for word in word_list:
            if word not in wordFreq[label]:
                wordFreq[label][word] = 0
            wordFreq[label][word] += 1
            
        totalFreq[label] += len(word_list)
    
    totalDocs[label] += len(listOfFiles)

trainModel(training_dir+"/pos",1,wordFreq,totalFreq,totalDocs)
trainModel(training_dir+"/neg",0,wordFreq,totalFreq,totalDocs)    

# Testing the model

def predictLabel(path,wordFreq,totalFreq,totalDocs,isPosCloud):
    global posCloudText
    global negCloudText
    
    file = open(path,"r")
    text = file.read()
    word_list = re.split(' |,|\\.|\n|:|;|"|\'|`|{{|}}|[|]|\)|\(',text)
    
    positive_prob = 0
    negative_prob = 0
    
    for word in word_list:
        
        if isPosCloud:
            posCloudText += word + " "
        else:
            negCloudText += word + " "
        
        posFreq = alpha if word not in wordFreq[1] else wordFreq[1][word] + alpha
        negFreq = alpha if word not in wordFreq[0] else wordFreq[0][word] + alpha
        
        positive_prob += math.log(posFreq) - math.log(totalFreq[1] + alpha*len(wordFreq[1]))
        negative_prob += math.log(negFreq) - math.log(totalFreq[0] + alpha*len(wordFreq[0]))
    
    positive_prob +=  -1*math.log(totalDocs[0])
    negative_prob +=  -1*math.log(totalDocs[1])
    
    if positive_prob>=negative_prob:
        return 1
    return 0
        
    
def prediction(path,wordFreq,totalFreq,totalDocs):
    correct = 0
    total = 0
    
    pos_path = path + "/pos"
    listOfPos = os.listdir(pos_path)
    for fileName in listOfPos:
        thisLabel = predictLabel(os.path.join(pos_path,fileName),wordFreq,totalFreq,totalDocs,True)
        if(thisLabel==1):
            correct += 1
    total += len(listOfPos)
    
    neg_path = path + "/neg"
    listOfNeg = os.listdir(neg_path)
    for fileName in listOfNeg:
        thisLabel = predictLabel(os.path.join(neg_path,fileName),wordFreq,totalFreq,totalDocs,False)
        if(thisLabel==0):
            correct += 1
    total += len(listOfNeg)
    
    print("The accuracy is : "+str(correct/total))

print("For training dataset: ")
prediction(training_dir,wordFreq,totalFreq,totalDocs)
print("For testing dataset: ")
prediction(testing_dir,wordFreq,totalFreq,totalDocs)


# wordclouds
#WordCloud(width = 800, height = 800, background_color ='white',stopwords = set(STOPWORDS),
#          min_font_size = 10).generate(posCloudText)

#WordCloud(width = 800, height = 800, background_color ='white',stopwords = set(STOPWORDS),
#          min_font_size = 10).generate(negCloudText)


