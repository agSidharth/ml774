# Importing libraries..
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import time


# sampling points
num_samples = 1000000
sample_theta_array = [[3],[1],[2]]
sample_theta = np.array(sample_theta_array)

x0 = np.ones((num_samples,1))
x1 = np.random.normal(3,2,(num_samples,1))
x2 = np.random.normal(-1,2,(num_samples,1))
noise = np.random.normal(0,np.sqrt(2),(num_samples,1))

Y = x0*sample_theta[0][0] + x1*sample_theta[1][0] + x2*sample_theta[2][0] + noise
X = np.append(x0,x1,axis = 1)
X = np.append(X,x2,axis = 1)


# -------------- CHANGE PARAMETERS-------------
#initializing parameters

alpha = 0.001
epsilon = 0.001

theta = np.zeros((sample_theta.size,1))
batch_size = 1000000
check_size = 1000
min_iter = (X.shape[0]/batch_size)

cost_list = []
theta_list = [theta]
# --------------xxxxxxxx-------------

# helper functions to compute cost, calculate the gradient and train the model..

def cost(X,Y,theta):
    temp = np.dot(X,theta)
    temp = (Y - temp)**2
    this_cost = np.sum(temp)/(2*X.shape[0])
    return this_cost

def cost_grad_update(X,Y,theta,alpha):
    temp = np.dot(X,theta)
    temp = Y - temp
    theta = theta + (alpha*np.dot(X.transpose(),temp))/(X.shape[0])
    return theta

def train(X,Y,theta):
    ax = plt.axes(projection='3d')
    iteration = 0
    finished = False
    curr_cost = 0
    old_cost = 0
    
    while(not finished):
        for idx in range(0,num_samples,batch_size):
            
            if(finished):
                break
                
            x_curr = X[idx:idx+batch_size,:]
            y_curr = Y[idx:idx+batch_size,:]
            
            curr_cost += cost(x_curr,y_curr,theta)
            theta = cost_grad_update(x_curr,y_curr,theta,alpha)
            
            if(iteration%check_size==0):
                curr_cost /= check_size
                cost_list.append(curr_cost)
                theta_list.append(theta)
                print("Iteration: "+str(iteration)+" ==> "+ str(abs(curr_cost-old_cost)))
                if(abs(curr_cost-old_cost)<epsilon and iteration>=min_iter):
                    finished = True
                
                old_cost = curr_cost
                curr_cost = 0
            
            if(iteration%(check_size/20)==0):
                ax.scatter3D(theta[0],theta[1],theta[2])
                
            iteration += 1
    
    plt.savefig('2_d_'+str(batch_size)+'.png')
    #plt.show()
    print("Training loss: "+str(cost(X,Y,theta)))
    print("Learning rate: "+str(alpha))
    print("Batch size used: "+str(batch_size))
    print("Iterations taken: "+str(iteration))
    print("Epsilon for cost used: "+str(epsilon))
    print("Final Parameters: "+str(theta))
    return theta

# training of model

START_TIME = time.time()
theta = train(X,Y,theta)
print("Time taken : "+str(time.time()-START_TIME))
# testing on given input

df = pd.read_csv("data/q2/q2test.csv")
x0 = np.ones((df['X_1'].shape[0],1))
x1 = np.array(df['X_1']).reshape(-1,1)
x2 = np.array(df['X_2']).reshape(-1,1)

Y_test = np.array(df['Y']).reshape(-1,1)
X_test_temp = np.append(x0,x1,axis = 1)
X_test = np.append(X_test_temp,x2,axis = 1)

error_learned = cost(X_test,Y_test,theta)
error_original = cost(X_test,Y_test,sample_theta)

print("Test Error for learned model with batch_size = "+str(batch_size)+" : "+str(error_learned))
print("Test Error for original hypthesis : "+str(error_original))

