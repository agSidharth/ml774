import math
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import re
import pickle
import cvxopt
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

def sklearn_svm(X,Y,X_test,Y_test,isLinear,C,gamma):

    if isLinear:
        print("Training the linear classifer")
        clf = SVC(kernel='linear', C=C)
    else:
        print("Training the gaussian classifer")
        clf = SVC(kernel='rbf', gamma=gamma, C=C)
    
    start_time = time.time()
    clf.fit(X,Y)
    print("Total training time: "+str(time.time()-start_time))
    
    pred = clf.predict(X)

    print("Number of support vectors used : "+str(np.sum(clf.n_support_)))
    print("Training accuracy: "+str(metrics.accuracy_score(pred,Y)))

    test_pred = clf.predict(X_test)
    print("Testing accuracy: "+str(metrics.accuracy_score(test_pred,Y_test)))

    if(isLinear and False):
      print("The values of w and b in case of linear are: ")
      print(clf.coef_[0])
      print(clf.intercept_[0])
    
    #print("The confusion matrix on training data is : ")
    #print(metrics.confusion_matrix(Y,pred))

    print("The confusion matrix on testing data is : ")
    print(metrics.confusion_matrix(Y_test,test_pred))

    return test_pred

test_pred = sklearn_svm(X,Y,X_test,Y_test,False,C,gamma)

def missclassified(test_pred,Y_test,X_test):
  count = 5
  for idx in range(len(test_pred)):
    if test_pred[idx]!=Y_test[idx]:

      img = X_test[idx].reshape((32,32,3))
      plt.imshow(img, interpolation='none')
      plt.savefig("3c_img_"+str(idx))
      plt.show()
      count -= 1

      if(count==0): break
  pass