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
import os

def returnAgg(vector):
    if DATA_IMPUTE=="mean":
        return np.mean(vector)
    return np.median(vector)

def returnData(filename):
    df = pd.read_csv(filename)
    
    if DATA_IMPUTE!=None:
        
        ageRep = returnAgg(np.array(df[df.Age!='?']["Age"],dtype=float))
        shapeRep = returnAgg(np.array(df[df.Shape!='?']["Shape"],dtype=float))
        marginRep = returnAgg(np.array(df[df.Margin!='?']["Margin"],dtype=float))
        densityRep = returnAgg(np.array(df[df.Density!='?']["Density"],dtype=float))
        
        df.Age.replace('?',str(ageRep),inplace = True)
        df.Shape.replace('?',str(shapeRep),inplace = True)
        df.Margin.replace('?',str(marginRep),inplace = True)
        df.Density.replace('?',str(densityRep),inplace = True)
    
    df.dropna()
    df = df[(df.Age!='?') & (df.Shape!='?') & (df.Margin!='?') & (df.Density!='?')]
    
    X = np.array([df["Age"],df["Shape"],df["Margin"],df["Density"]],dtype = float).T
    Y = np.array(df["Severity"],dtype = float)
    
    #X = prepro.normalize(X)
    
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
    path = os.path.join(path,'1.1.c.'+name+'.png')
    plt.savefig(path)
    plt.figure()

def handlePart(X_train,X_test,X_val,Y_train,Y_test,Y_val,qPart):

    print("Runing for part : "+qPart)
    if qPart=="a":
        clf = DecisionTreeClassifier(random_state = SEED)
        clf.fit(X_train,Y_train)
        printAccuracies(clf)
        tree.plot_tree(clf)

        path = OUTPUT_DIR
        path = os.path.join(path,"1.1.a.png")
        plt.savefig(path)
        plt.figure()
        return clf

    elif qPart=="b":
        param_grid = {'criterion' :['gini', 'entropy'],
             'max_depth' : range(1,11),
             'min_samples_split': range(2,11),
             'min_samples_leaf': range(1,11)}

        clf = doGridSearch(DecisionTreeClassifier(random_state = SEED),param_grid)
        tree.plot_tree(clf)

        path = OUTPUT_DIR
        path = os.path.join(path,"1.1.b.png")
        plt.savefig(path)
        plt.figure()
        return clf

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

        plotGraph(pruneDict,numNodesList,'num_nodes')
        plotGraph(pruneDict,depthList,'max_depth')
        plotGraph(pruneDict,trainList,'train_accu')
        plotGraph(pruneDict,testList,'test_accu')
        plotGraph(pruneDict,valList,'val_accu')

        #print(best_val_acc)
        printAccuracies(best_clf)
        tree.plot_tree(best_clf)

        path = OUTPUT_DIR
        path = os.path.join(path,"1.1.c.png")
        plt.savefig(path)
        plt.figure()
        return best_clf

    elif qPart=="d":
        param_grid = {'criterion' :['gini', 'entropy'],
              'n_estimators' : [50,100,150,200],
              'max_features' : ["sqrt", "log2", None],
              'min_samples_split': [2,4,6,8,10]}

        best_clf = doGridSearch(RandomForestClassifier(random_state = SEED,oob_score = True),param_grid)
        print('OOB score : '+str(best_clf.oob_score_))
        return best_clf

    elif qPart=="e":
        pass 
    elif qPart=="f":
        param_grid = {'n_estimators' : [10,20,30,40,50],
              'subsample' : [0.1,0.2,0.3,0.4,0.5],
              'max_depth' : [4,5,6,7,8,9,10]}

        clf = doGridSearch(xgb.XGBClassifier(),param_grid)
        return clf
    else:
        print("Unkown question part entered\n")
        exit()

def printFileOutput(clf,qPart,X_test):
    filename = DATASET_IND+qPart+".txt"
    path = OUTPUT_DIR

    path = os.path.join(path,filename)
    thisFile = open(path,"w")

    Y_test_pred = clf.predict(X_test)
    for out in Y_test_pred:
        thisFile.write(str(out)+"\n")

    thisFile.close()
    return 

if __name__=="__main__":

    SEED = 50661

    trainFile = sys.argv[1]
    valFile = sys.argv[2]
    testFile = sys.argv[3]
    OUTPUT_DIR = sys.argv[4]
    qPart = sys.argv[5]

    DATASET_IND = "1_"

    if qPart!="e":
        
        DATA_IMPUTE = None
        
        X_train,Y_train = returnData(trainFile)
        X_test,Y_test = returnData(testFile)
        X_val,Y_val = returnData(valFile)

        clf = handlePart(X_train,X_test,X_val,Y_train,Y_test,Y_val,qPart)
        printFileOutput(clf,qPart,X_test)
    
    else:
        DATA_IMPUTE = "mean"
        print("\nFor data imputing with "+DATA_IMPUTE+"\n")
        X_train,Y_train = returnData(trainFile)
        X_test,Y_test = returnData(testFile)
        X_val,Y_val = returnData(valFile)

        for qPart in ['a','b','c','d']:
            clf = handlePart(X_train,X_test,X_val,Y_train,Y_test,Y_val,qPart)
        
        DATA_IMPUTE = "median"
        print("\nFor data imputing with "+DATA_IMPUTE+"\n")
        X_train,Y_train = returnData(trainFile)
        X_test,Y_test = returnData(testFile)
        X_val,Y_val = returnData(valFile)

        for qPart in ['a','b','c','d']:
            clf = handlePart(X_train,X_test,X_val,Y_train,Y_test,Y_val,qPart)
        
        printFileOutput(clf,"e",X_test)



