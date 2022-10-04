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

def svm_cvxopt(X,Y,isLinear,C,gamma):
    
    start_time = time.time()

    if isLinear:
        tempProd0 = np.matmul(X,X.T)
        tempProd = np.matmul(Y,Y.T)*tempProd0
        P = cvxopt.matrix(tempProd)
    else:
        tempProd0 = gaussian_kernel(X,X)
        P = cvxopt.matrix(np.matmul(Y,Y.T)*tempProd0)
    
    tempG = np.identity(Y.shape[0])
    G = cvxopt.matrix(np.append(tempG,-1*tempG,axis = 0))
    
    Q = cvxopt.matrix(-1*np.ones((Y.shape[0],1)))
    
    A = cvxopt.matrix(Y.T)
    
    tempH = np.ones((Y.shape[0],1))
    H = cvxopt.matrix(np.append(C * tempH, 0 * tempH, axis=0))
    
    B = cvxopt.matrix(0.0)
    
    sol = cvxopt.solvers.qp(P, Q, G, H, A, B, options={'show_progress': False})
    
    tempX = np.array(sol['x'])
    alpha = np.reshape(tempX, (Y.shape[0],1))
    
    indices = []
    
    for idx in range(Y.shape[0]):
        if alpha[idx] > EPSILON:
            indices.append(idx)

    if isLinear:
      inner_product = np.sum(alpha*Y*tempProd0,0)
    else:
      inner_product = np.sum(alpha*Y*gaussian_kernel(X,X),0)

    M = -float("inf")
    M_index = -1
    m = float("inf")
    m_index = -1

    for idx in range(Y.shape[0]):
        val = -float("inf") if Y[idx]==1 else inner_product[idx]
        if M<val:
          M = val
          M_index = idx
        
        val = float("inf") if Y[idx]==-1 else inner_product[idx]
        if m>val:
          m = val
          m_index = idx
    
    b = -1*(inner_product[M_index]+inner_product[m_index])/2
    
    print("Total training time: "+str(time.time()-start_time))
    return alpha,indices,b

def LinearAccuracy(X,Y,W,b):
    pred = np.where((np.matmul(X,W)+b)>=0,1,-1)
    return (sum(pred==Y)[0]*100)/Y.shape[0]

def gaussian_kernel(x, y):
    return np.exp(-1*gamma * cdist(x, y, "sqeuclidean"))

def eval(x_test, y_test,X_train,Y_train,alpha,b,indices,gamma):
    y_hat = (np.sum(alpha * Y_train * gaussian_kernel(X_train, x_test),axis=0,keepdims=True).T + b)
    return np.sign(y_hat)

def GaussAccuracy(X_test,Y_test,X_train,Y_train,alpha,b,indices,gamma):
    final_pred = eval(X_test,Y_test,X_train[indices],Y_train[indices],alpha[indices],b,indices,gamma)

    return (sum(final_pred==Y_test)[0]*100)/Y_test.shape[0]

def kC2_cvxopt(X,Y,C,gamma):
    classifier = {}

    for idx in range(TOTAL_CLASSES):
      for jdx in range(idx+1,TOTAL_CLASSES):

        indices = np.append(np.where(Y==idx)[0],np.where(Y==jdx)[0])
        Xij = X[indices]
        Yij = np.where(Y[indices]==jdx,1,-1)

        Xij = Xij.astype('float64')
        Yij = Yij.astype('float64')
        
        classifier[idx,jdx] = svm_cvxopt(Xij,Yij,False,C,gamma)
        print(str(idx)+","+str(jdx)+": Model trained")
    
    return classifier

classifier = kC2_cvxopt(X,Y,C,gamma)

def kC2_accuracy(X_test,Y_test,X_train,Y_train,classifier):

  shape = (Y_test.shape[0],TOTAL_CLASSES)
  votes = np.zeros(shape)

  for idx in range(TOTAL_CLASSES):
    for jdx in range(idx+1,TOTAL_CLASSES):
      
      indices = np.append(np.where(Y_train==idx)[0],np.where(Y_train==jdx)[0])
      Xij = X_train[indices]
      Yij = np.where(Y_train[indices]==jdx,1,-1)

      Xij = Xij.astype('float64')
      Yij = Yij.astype('float64')

      ALPHA = classifier[idx,jdx][0]
      INDICES = classifier[idx,jdx][1]
      B = classifier[idx,jdx][2]

      this_pred = eval(X_test,Y_test,Xij[INDICES],Yij[INDICES],ALPHA[INDICES],B,INDICES,gamma)

      for kdx in range(Y_test.shape[0]):
        if this_pred[kdx]==1:
          votes[kdx][jdx] += 1
        else:
          votes[kdx][idx] += 1
      
      print(str(idx)+","+str(jdx)+": prediction done")

  pred = np.reshape(np.argmax(votes,1),(Y_test.shape[0],1))
  acc = (sum(pred==Y_test)[0]*100)/Y_test.shape[0]
  print("Accuracy is : "+str(acc))
  print(metrics.confusion_matrix(Y_test,pred))

  return pred

print("For testing dataset: ")
kc2_test_pred = kC2_accuracy(X_test,Y_test,X,Y,classifier)