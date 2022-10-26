#from google.colab import drive
#drive.mount('/content/drive')

#%cd /content/drive/MyDrive/Assignment3/

# Loading important libraries

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import sklearn.preprocessing as prepro
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import sklearn.tree as tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier

def readData(filename):
  data = np.loadtxt(filename,delimiter = ',')
  X = data[:,:-1]/255
  enc = OneHotEncoder()
  Y = data[:,-1].reshape((-1,1))
  Y = (enc.fit_transform(Y).toarray())
  X,Y = shuffle(X,Y)
  return X.T,Y.T

class NeuralNet:
    def __init__(self,num_features,batch_size,target_classes,learning_rate,hidden_layers,activation = 'sigmoid',max_epochs = 200,learning_rate_type = 'constant'):
        self.num_features = num_features
        self.batch_size = batch_size
        self.target_classes = target_classes
        self.learning_rate = learning_rate
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.max_epochs = max_epochs
        self.adaptive_rate = (learning_rate_type=="adaptive")
        self.EPSILON = EPSILON

        # for norm_factor see this link "https://stats.stackexchange.com/questions/47590/what-are-good-initial-weights-in-a-neural-network"
        # another link "https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79"
        # note we will add one columnn for bias.
        norm_factor = np.sqrt(2/(self.num_features+1))
        self.params = [np.random.randn(self.hidden_layers[0],self.num_features+1)*norm_factor]

        for idx in range(1,len(self.hidden_layers)):
            norm_factor = np.sqrt(2/(self.hidden_layers[idx-1]+1))
            self.params.append(np.random.randn(self.hidden_layers[idx],self.hidden_layers[idx-1]+1)*norm_factor)
        
        norm_factor = np.sqrt(2/(self.hidden_layers[-1]+1))
        self.params.append(np.random.randn(self.target_classes,self.hidden_layers[-1]+1)*norm_factor)
    
    def activate(self,x,layerNum):

        if (layerNum == len(self.params) - 1) or (self.activation=="sigmoid"):
            return self.sigmoid(x)
        return np.maximum(0,x)
    
    def de_activate(self,x,layerNum):
        if (layerNum == len(self.params) - 1) or (self.activation=="sigmoid"):
            return self.de_sigmoid(x)
        return (np.ones_like(x) * (x >= 0))
    
    def sigmoid(self,x):
        return 1/(1 + np.exp(-1*x))
    
    def de_sigmoid(self,x):
        temp = self.sigmoid(x)
        temp = temp*(1-temp)
        return temp

    def forward(self,x,thisAxis = 0):
        self.preactivation = []
        self.output = []
        layerNum = 0

        for layer in self.params:
            bias = np.ones((1,x.shape[1]))
            
            # append the bias to the last row..
            befAct = layer @ (np.append(x,bias,axis = thisAxis))
            self.preactivation.append(befAct)

            # activate the input given and make the output recieved to be the input for next layer.
            x = self.activate(befAct,layerNum)
            layerNum += 1
            self.output.append(x)
        
        return x
    
    def backtrack(self,x,thisAxis = 0):
        self.delta = []

        temp = self.preactivation[len(self.params)-1]
        temp = (x - self.output[len(self.params)-1])*(self.de_activate(temp,len(self.params)-1))

        self.delta.append(temp)

        for idx in range(1,len(self.params)):
            
            """
            print((self.delta[idx-1]).shape)
            print((self.params[len(self.params)-idx][:,0:-1]).shape)
            exit()
            """

            del_theta = (self.params[len(self.params)-idx][:,0:-1].T) @ (self.delta[idx-1])
            temp = self.preactivation[len(self.params)-idx-1]

            self.delta.append(del_theta * self.de_activate(temp,-1))
        
        self.delta.reverse()
    
    def predict(self,x,thisAxis = 0):
        return np.argmax(self.forward(x),axis = thisAxis)

    def cost(self,x,y):
        return np.sum((y - self.forward(x))**2)/(2*y.shape[-1])
    
    def accuracy(self,x,y,thisAxis = 0):
        return np.sum(self.predict(x)==np.argmax(y,axis = thisAxis))/y.shape[1]

    def returnGradients(self,Xb,Yb,thisAxis = 0):
        
        self.forward(Xb)
        self.backtrack(Yb)

        gradient = []
        
        for idx in range(len(self.params)):
            if idx>0: Xb = np.array(self.output[idx-1])

            bias = np.ones((1,Xb.shape[1]))
            Xb = np.array(np.append(Xb,bias,axis = thisAxis))

            temp = (-1*self.delta[idx]) @ Xb.T
            gradient.append(temp/self.batch_size)
        
        return gradient
        
    def train(self,X,Y):

        new_cost = np.inf/2
        count = 0
        curr_learning_rate = self.learning_rate

        while (count<self.max_epochs or (curr_learning_rate)*100>self.learning_rate):
            prev_cost = new_cost

            for batch in range(self.batch_size):
                Xb = X[:,batch*self.batch_size:(batch + 1)*self.batch_size]
                Yb = Y[:,batch*self.batch_size:(batch + 1)*self.batch_size]

                gradient = self.returnGradients(Xb,Yb)

                for idx in range(len(self.params)):
                    self.params[idx] = self.params[idx] - curr_learning_rate*gradient[idx]

            #update parameters
            if self.adaptive_rate:
                curr_learning_rate = self.learning_rate/np.sqrt(count+1)

            new_cost = self.cost(X,Y)
            if(abs(new_cost-prev_cost)<self.EPSILON): break
            count += 1

            if DEBUG: print("New accuracy : "+str(self.accuracy(X,Y)))
        
        print("Training has completed with final train accuracy : "+str(self.accuracy(X,Y)))
        
def plotGraph(values,name,hidden_layer):
    plt.plot(hidden_layer,values)
    plt.scatter(hidden_layer,values)
    plt.xlabel("hidden layer size")
    plt.ylabel(name)
    plt.title(name+" vs hidden layer size")
    plt.savefig("2_"+name+".png")
    plt.figure()

def trainTestModel(X_train,Y_train,X_test,Y_test,hidden_layer,learning_rate = 0.1,learning_rate_type="constant",activation = "sigmoid",verbose = False,thisAxis = 0):
    thisModel = NeuralNet(X_train.shape[0],100,10,learning_rate,hidden_layer,learning_rate_type=learning_rate_type,activation=activation)
    start_time = time.time()

    thisModel.train(X_train,Y_train)
    trainTime = time.time()-start_time
    
    trainAcc = thisModel.accuracy(X_train,Y_train)
    testAcc = (thisModel.accuracy(X_test,Y_test))

    confMatrix = confusion_matrix(np.argmax(Y_test,axis = thisAxis),thisModel.predict(X_test))

    if verbose:
        print("For hidden layers : "+str(hidden_layer))
        print("For learningRateType : "+learning_rate_type+" , activation : "+activation)
        print("Training time : "+str(trainTime))
        print("Training accuracy : "+str(trainAcc))
        print("Testing accurcay : "+str(testAcc))
        print("The confusion matrix : ")
        print(confMatrix)
        print('\n')

    return trainTime,trainAcc,testAcc,confMatrix


def partBC(X_train,Y_train,X_test,Y_test,hidden_layer = [5,10,15,20,25],learning_rate = 0.1,learning_rate_type = "constant",activation = "sigmoid",partD = False):

    training_time = []
    acc_train = []
    acc_test  = []

    for thisH in hidden_layer:
        print("For size : "+str(thisH))

        hidden2pass = [thisH]
        if partD:
            hidden2pass = thisH
        thisTime,thisTrain,thisTest,thisConf = trainTestModel(X_train,Y_train,X_test,Y_test,hidden2pass,learning_rate,learning_rate_type,activation,verbose = True)
            
        training_time.append(thisTime)
        acc_train.append(thisTrain)
        acc_test.append(thisTest)
        
        """
        print("The confusion matrix for size "+str(thisH)+" is")
        print(thisConf)
        print("\n\n")
        """
    
    return training_time,acc_train,acc_test

SEED = 50661
np.random.seed(SEED)

trainFile = "COL774_fmnist/fmnist_train.csv"
testFile = "COL774_fmnist/fmnist_test.csv"

DEBUG = False
EPSILON = 1e-4

X_train,Y_train = readData(trainFile)
X_test,Y_test = readData(testFile)


"""
# Part B
print("For part B")
hidden_layer = [5,10,15,20]
training_time,acc_train,acc_test = partBC(X_train,Y_train,X_test,Y_test,hidden_layer)

plotGraph(training_time,"b_time",hidden_layer)
plotGraph(acc_train,"b_train_acc",hidden_layer)
plotGraph(acc_test,"b_test_acc",hidden_layer)



# Part C
print("For Part C")
hidden_layer = [5,10,15,20]
training_time,acc_train,acc_test = partBC(X_train,Y_train,X_test,Y_test,hidden_layer,learning_rate_type="adaptive")

plotGraph(training_time,"c_time",hidden_layer)
plotGraph(acc_train,"c_train_acc",hidden_layer)
plotGraph(acc_test,"c_test_acc",hidden_layer)


# Part D

print("For part D:\n")
hidden_layer = [100,100]
trainTime,trainAcc,testAcc,confMatrix = trainTestModel(X_train,Y_train,X_test,Y_test,hidden_layer,learning_rate_type="adaptive",activation = "relu",verbose = True)
trainTime,trainAcc,testAcc,confMatrix = trainTestModel(X_train,Y_train,X_test,Y_test,hidden_layer,learning_rate_type="adaptive",activation = "sigmoid",verbose = True)


# Part E
print("For part E:\n")
hidden_layer = [[50,50],[50,50,50],[50,50,50,50],[50,50,50,50,50]]
len_layers = [2,3,4,5]

training_time,acc_train,acc_test = partBC(X_train,Y_train,X_test,Y_test,hidden_layer,learning_rate_type="adaptive",activation = "relu",partD = True)

plotGraph(training_time,"e_relu_time",len_layers)
plotGraph(acc_train,"e_relu_train_acc",len_layers)
plotGraph(acc_test,"e_relu_test_acc",len_layers)

training_time,acc_train,acc_test = partBC(X_train,Y_train,X_test,Y_test,hidden_layer,learning_rate_type="adaptive",activation = "sigmoid",partD = True)

plotGraph(training_time,"e_sig_time",len_layers)
plotGraph(acc_train,"e_sig_train_acc",len_layers)
plotGraph(acc_test,"e_sig_test_acc",len_layers)

"""
# Part g
print("For part G:\n")
clf = MLPClassifier([50,50,50],'relu',solver = 'sgd',batch_size = 100,learning_rate_init = 0.1,max_iter = 200,random_state = SEED,verbose = False)

start_time = time.time()
clf.fit(X_train.T,Y_train.T)
trainTime = time.time() - start_time

pred_train = np.argmax(clf.predict_proba(X_train.T),axis = 1)
pred_test  = np.argmax(clf.predict_proba(X_test.T),axis = 1)

trainAcc = np.sum(pred_train==np.argmax(Y_train.T,axis = 1))/Y_train.shape[1]

testAcc  = np.sum(pred_test==np.argmax(Y_test.T,axis = 1))/Y_test.shape[1]

print("Total time taken : "+str(time.time()-start_time))
print("Training accuracy : "+str(trainAcc))
print("Testing accuracy : "+str(testAcc))

