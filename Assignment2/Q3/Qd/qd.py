import math
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import re
import pickle
from scipy.spatial.distance import pdist,squareform,cdist
from sklearn.svm import SVC
from sklearn import metrics
import time

training_file = str(sys.argv[1])+"/"+str(os.listdir(str(sys.argv[1]))[0])
testing_file  = str(sys.argv[2])+"/"+str(os.listdir(str(sys.argv[2]))[0])

EPSILON = 1e-4
gamma = 0.001
C = 1.0
TOTAL_CLASSES = 5

def readData(file_name):
    file = open(file_name,"rb")
    fileDict = pickle.load(file)
    
    X = []
    Y = []

    for idx in range(len(fileDict["data"])):
        X.append(fileDict["data"][idx].reshape(-1))
        Y.append(fileDict["labels"][idx])

    X = np.array(X)
    Y = np.array(Y)
    
    X = X/255
    
    X = X.astype('float64')
    Y = Y.astype('float64')
    
    return X,Y

X,Y = readData(training_file)
X_test,Y_test = readData(testing_file)

from sklearn.utils import shuffle
X,Y = shuffle(X,Y)
X_test,Y_test = shuffle(X,Y)

from sklearn.model_selection import KFold

C_List = [1e-5, 1e-3, 1, 5, 10]

def cross_SVM(X,Y,X_test,Y_test,C_List,gamma,kfolds = 5):

    val_acc = []
    test_acc = []
    kf = KFold(n_splits = kfolds)

    for thisC in C_List:
      print("Currently testing for : "+str(thisC))
      model = SVC(kernel='rbf', gamma=gamma, C=thisC)

      maxAcc = 0
      maxTestAcc = 0
      sumAcc = 0
      count = 0

      for train_index,test_index in kf.split(X):

        print("Started split num : "+str(count))
        
        X_big,X_small = X[train_index],X[test_index]
        Y_big,Y_small = Y[train_index],Y[test_index]

        count += 1

        print("Started training model")
        model.fit(X_big,Y_big)
        print("Finished training model")
        model_pred = model.predict(X_small)
        thisAcc = metrics.accuracy_score(Y_small,model_pred)

        if thisAcc>maxAcc:
          maxAcc = thisAcc

          test_pred = model.predict(X_test)
          maxTestAcc = metrics.accuracy_score(Y_test,test_pred)

        sumAcc += thisAcc
      
      print(sum/kfolds)
      print(maxAccxs)
      val_acc.append(sumAcc/kfolds)
      test_acc.append(maxTestAcc)
    
    return val_acc,test_acc

val_acc,test_acc = cross_SVM(X,Y,X_test,Y_test,C_List,gamma,5)

print(val_acc)
print(test_acc)

c_vals_log = []

for thisC in C_List:
  c_vals_log.append(math.log(thisC))

plt.plot(c_vals_log, val_acc)
plt.plot(c_vals_log, test_acc)
plt.title("Cross validation & test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Log of C")
plt.legend()
plt.savefig("3d.png")

