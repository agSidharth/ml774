# Importing libraries..
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd


# Reading data
df = pd.read_csv("/Users/sidharthagarwal/Desktop/assignments/ml774/data/q3/logisticX.csv")
X  = df.to_numpy()

# adding intercept in X in the last dimension...
X = np.append(X,np.ones((X.shape[0],1)),axis = 1)

dfY = pd.read_csv("/Users/sidharthagarwal/Desktop/assignments/ml774/data/q3/logisticY.csv")
Y = dfY.to_numpy()

# Normalizing the data

mean = np.mean(X[:,0:2],axis = 0)
std  = np.std(X[:,0:2],axis = 0)

X[:,0:2] = X[:,0:2]-mean
X[:,0:2] = X[:,0:2]/std

#print(mean)
#print(std)

# initialization of parameters
theta = np.zeros((X.shape[1],1))
epsilon = 0.0001

#print(theta)

# helper functions to calculate the hypothesis, hessian, gradients and training algorithm

# finding the hypothesis
def sigmoid(X,theta):
    y_temp = (1/(1+np.exp(-1*np.dot(X,theta))))
    return y_temp

# finding the hessian using the formula XT.diag.X
def returnHes(X,Y,theta):
    y_hyp = sigmoid(X,theta)
    diag = np.identity(X.shape[0])*np.dot(y_hyp.transpose(),(1-y_hyp))
    return np.dot(X.transpose(),np.dot(diag,X))
    
# finding the gradient..
def returnGrad(X,Y,theta):
    return np.dot(X.transpose(),sigmoid(X,theta)-Y)

# converging the algorithm..
def newton_method(X,Y,theta):
    diff = np.inf
    iterations = 0
              
    while diff>epsilon:
              
        hessian = returnHes(X,Y,theta)
        hessian_inv = np.linalg.inv(hessian)
        gradient = returnGrad(X,Y,theta)
        
        new_theta = theta - np.dot(hessian_inv,gradient)
        diff = abs(np.linalg.norm(new_theta - theta))
        theta = new_theta
        
        print("Iteration : "+str(iterations)+" ==> "+str(diff))
        iterations += 1
    
    print("Number of iterations taken: "+str(iterations))
    print("Final theta : "+str(theta))
    return theta

# running the algorithm
theta = newton_method(X,Y,theta)

# plotting the graph..
y0 = []
x10 = []
x20 = []

y1 = []
x11 = []
x21 = []

for idx in range(X.shape[0]):
    if Y[idx][0] == 0:
        y0.append(idx)
        x10.append(X[idx][0])
        x20.append(X[idx][1])
    else:
        y1.append(idx)
        x11.append(X[idx][0])
        x21.append(X[idx][1])

plt.scatter(x10,x20,color = "blue",marker = "x",label = '0')
plt.scatter(x11,x21,color = "orange" ,marker = "o",label = '1')

# the equation of line in logistic regression ins thetaT.x = 0
y_line = (-1)*X[:,0]*theta[0][0]/theta[1][0] + (-1)*theta[2][0]/theta[1][0]
plt.plot(X[:,0],y_line)

plt.legend()

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Logistic Regression")
plt.savefig("3.png")
#plt.show()