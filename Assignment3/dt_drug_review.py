# Loading important libraries

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import sklearn.preprocessing as prepro
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import sklearn.tree as tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import time
from nltk.corpus import stopwords
import os
#import lightgbm as lgb

def returnDataText(filename,isTrain = False):
    df_temp = pd.read_csv(filename)
    df_temp.replace(np.nan,"",inplace = True)
    
    if SAMPLE_SIZE==None or isTrain==False:
      df = df_temp
    else:
      df = df_temp.sample(SAMPLE_SIZE,random_state = SEED)
    
    text_vector = df["condition"] + " " + df["review"] + " "+ df["date"]
    #print(text_vector.shape)
    
    if(isTrain):
        vectorizer.fit(text_vector)
        
    X = vectorizer.transform(text_vector)
    
    #print(len(vectorizer.vocabulary_))
    #print(len(vectorizer.stop_words_))
    Y = np.array(df["rating"],dtype = int)
    
    return X,Y

def printAccuracies(clf):
    
    Y_train_pred = clf.predict(X_train)
    print("Training accuracy : "+str(accuracy_score(Y_train,Y_train_pred)))

    Y_test_pred = clf.predict(X_test)
    print("Testing accuracy : "+str(accuracy_score(Y_test,Y_test_pred)))

    Y_val_pred = clf.predict(X_val)
    print("Validation accuracy : "+str(accuracy_score(Y_val,Y_val_pred)))

def doGridSearch(ourEstimator,ourParamGrid):
    grid_search = GridSearchCV(estimator = ourEstimator,param_grid = ourParamGrid)
    grid_search.fit(X_train,Y_train)

    clf = grid_search.best_estimator_
    print("Optimal parameters obtained are : "+str(grid_search.best_params_))
    printAccuracies(clf)
    
    return clf

def plotGraph(pruneDict,values,name):
    plt.plot(pruneDict['ccp_alphas'],values)
    plt.xlabel('alphas')
    plt.ylabel(name)
    plt.title(name+' vs alpha')

    path = OUTPUT_DIR
    path = os.path.join(path,'1.2.c.'+name+'.png')
    plt.savefig(path)
    plt.figure()

def handlePart(X_train,X_test,X_val,Y_train,Y_test,Y_val,qPart):

    print("Runing for part : "+qPart)
    if qPart=="a":
        clf = DecisionTreeClassifier(random_state = SEED)
        start_time = time.time()
        clf.fit(X_train,Y_train)
        print("Time taken : "+str(time.time()-start_time))
        printAccuracies(clf)
        if(VISUALIZE): tree.plot_tree(clf)

    elif qPart=="b":
        param_grid = {'max_depth' : [2,4,8],
             'min_samples_split': [2,4,8],
             'min_samples_leaf': [2,4,8]}

        start_time = time.time()
        clf = doGridSearch(DecisionTreeClassifier(random_state = SEED),param_grid)
        print("Time taken : "+str(time.time()-start_time))

        if(VISUALIZE): tree.plot_tree(clf)
    elif qPart=="c":
        clf = DecisionTreeClassifier(random_state = SEED)
        pruneDict = clf.cost_complexity_pruning_path(X = X_train,y = Y_train)

        plotGraph(pruneDict,pruneDict['impurities'],'impurities')

        numNodesList = []
        depthList = []
        trainList = []
        testList = []
        valList = []

        best_clf = None
        best_val_acc = -1

        start_time = time.time()
        for thisAlpha in pruneDict["ccp_alphas"]:
            clf = DecisionTreeClassifier(ccp_alpha = thisAlpha,random_state = SEED)
            clf.fit(X_train,Y_train)
            
            numNodesList.append(clf.tree_.node_count)
            depthList.append(clf.tree_.max_depth)
            
            Y_train_pred = clf.predict(X_train)
            trainList.append(accuracy_score(Y_train,Y_train_pred))

            Y_test_pred = clf.predict(X_test)
            testList.append(accuracy_score(Y_test,Y_test_pred))

            Y_val_pred = clf.predict(X_val)
            valList.append(accuracy_score(Y_val,Y_val_pred))
            
            if accuracy_score(Y_val,Y_val_pred)>best_val_acc:
                best_val_acc = accuracy_score(Y_val,Y_val_pred)
                best_clf = clf

        print("Time taken : "+str(time.time()-start_time))


        plotGraph(pruneDict,numNodesList,'num_nodes')
        plotGraph(pruneDict,depthList,'max_depth')
        plotGraph(pruneDict,trainList,'train_accu')
        plotGraph(pruneDict,testList,'test_accu')
        plotGraph(pruneDict,valList,'val_accu')

        printAccuracies(best_clf)
    elif qPart=="d":
        # 1.d

        param_grid = {'n_estimators' : [50,100,150,200,250,300,350,400,450],
                    'max_features' : [0.4,0.5,0.6,0.7,0.8],
                    'min_samples_split': [2,4,6,8,10]}

        start_time = time.time()
        best_clf = doGridSearch(RandomForestClassifier(random_state = SEED,oob_score = True),param_grid)
        print("Time taken : "+str(time.time()-start_time))
        printAccuracies(best_clf)
        print('OOB score : '+str(best_clf.oob_score_))
    elif qPart=="e":
        param_grid = {'n_estimators' : [50,100,150,200,250,300,350,400,450],
              'subsample' : [0.4,0.5,0.6,0.7,0.8],
              'max_depth' : [40,50,60,70]}

        start_time = time.time()
        clf = doGridSearch(xgb.XGBClassifier(),param_grid)
        print("Time taken : "+str(time.time()-start_time))
    elif qPart=="f":
        pass
    elif qPart=="g":
        pass
    else:
        print("Doesn't support this question part")

if __name__=="__main__":

    SEED = 50661
    DATA_IMPUTE = "mean"
    STOP_WORDS = stopwords.words('english')
    VISUALIZE = False
    MIN_DF = 2
    SAMPLE_SIZE = None

    trainFile = sys.argv[1]
    valFile = sys.argv[2]
    testFile = sys.argv[3]
    OUTPUT_DIR = sys.argv[4]
    qPart = sys.argv[5]

    DATASET_IND = "2_"
    vectorizer = TfidfVectorizer(stop_words = STOP_WORDS,min_df = MIN_DF)

    X_train,Y_train = returnDataText(trainFile,True)
    X_test,Y_test = returnDataText(testFile)
    X_val,Y_val = returnDataText(valFile)

    handlePart(X_train,X_test,X_val,Y_train,Y_test,Y_val,qPart)

