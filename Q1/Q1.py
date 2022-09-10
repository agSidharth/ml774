#importing necessary libraries..
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits import mplot3d
import sys

#loading training data..
training_dir = str(sys.argv[1])
testing_dir  = str(sys.argv[2])

dfX = pd.read_csv(training_dir+"/X.csv")
dfY = pd.read_csv(training_dir+"/Y.csv")

X = dfX.to_numpy()
Y = dfY.to_numpy()

# adding intercept in X
X = np.append(X,np.ones(X.shape),axis = 1)

# Note: X is (num_examples*num_features)

# normalization of input
mean = np.mean(X[:,0])
std  = np.std(X[:,0])

X[:,0] = X[:,0]-mean
X[:,0] = X[:,0]/std

print("Mean: " + str(mean))
print("STD: "+str(std))


# -------------- CHANGE PARAMETERS-------------
# learning parameter and some more parameters initialization
alpha = 0.005
epsilon = 0.00000001
max_iter = 10000
theta = np.zeros((X.shape[1],1))

cost_list = []
theta_list = [theta]

draw_3d = False   # otherwise plot the scatter plot..

# --------------xxxxxxxxxxx-------------

if draw_3d:
    print("Creating the 3d plot")
else:
    print("Creating the contour plot")

# helper functions to compute cost, calculate the gradient and train the model..
# note that np.dot gives amtrix multiplication for 2d arrays..

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
    iteration = 0
    finished = False

    while(not finished):
        curr_cost = cost(X,Y,theta)
        theta = cost_grad_update(X,Y,theta,alpha)

        if(iteration>0):
            finished = finished | (abs(curr_cost - cost_list[-1])<epsilon)

        cost_list.append(curr_cost)
        theta_list.append(theta)

        if iteration%10==0 and draw_3d:
            ax.scatter3D(theta[0],theta[1],curr_cost)
        elif (iteration%10==0):
            ax.scatter(theta[0],theta[1],marker='x')
        plt.pause(0.2)
        iteration += 1
    
    if draw_3d:
        plt.savefig('1_c.png')
    else:
        plt.savefig('1_d_'+str(alpha)+'.png',bbox_inches='tight')
    
    print("Learning rate: "+str(alpha))
    print("Iterations taken: "+str(iteration))
    print("Epsilon for cost used: "+str(epsilon))
    print("Final Parameters: "+str(theta))
    print("Final loss: "+str(curr_cost))
    #plt.show()
    return theta

# Initialization for contour and 3d plots

# initalize x1 and x2 based on the range of theta's finally calculated

if not draw_3d:
    x1 = np.linspace(-1.5, 1.5, 200)
    y1 = np.linspace(-1.5, 2.5, 200)
else:
    x1 = np.linspace(-0.02, 0.02, 100)
    y1 = np.linspace(-0.75, 1.5, 100)
    

Xg,Yg = np.meshgrid(x1,y1)

# Zg contains the value of cost for theta = (Xg[i][j],Yg[i][j])
Zg = []

for i in range(Xg.shape[0]):
    Zg.append([])
    for j in range(Xg.shape[1]):
        Zg[i].append(cost(X,Y,np.array([[Xg[i][j]],[Yg[i][j]]])))

Zg = np.array(Zg)

if not draw_3d:
    ax = plt.figure(2).add_axes([0.2,0.2,1,1])
    cp = ax.contour(Xg,Yg,Zg)
    plt.clabel(cp,inline=1,fontsize = 10)
else:
    ax = plt.axes(projection='3d')
    ax.view_init(30, 60)
    ax.plot_wireframe(Xg,Yg,Zg,color='blue',linewidths=0.3)
    ax.set_zlabel('Cost')
    

ax.set_xlabel('Theta[0]')
ax.set_ylabel('Theta[1]')

# run the algorithm...
theta = train(X,Y,theta)

# 2d plot for data and hypothesis
plt.figure()
predictions = np.dot(X,theta)
plt.scatter(X[:,0],Y,label = "Ground Truth")
plt.plot(X[:,0],predictions,label = "Predictions",color = 'red')
plt.title("Data and hypothesis plot")
plt.xlabel("Wine Acidity")
plt.ylabel("Wine Density")
plt.savefig('1_b.png')
#plt.show()

# demo running to check the dimensionality and correctness of various operations in numpy.

#check1 = np.array([[1,2],[2,3]])
#check2 = np.array([[0],[-1]])
#print(check1.shape)
#print(check2.shape)
#print(np.dot(check1,check2))

# Testing on the test data
print("Testing on the test data")
dfX_test = pd.read_csv(testing_dir+"/X.csv")

X_test = dfX_test.to_numpy()

# adding intercept in X
X_test = np.append(X_test,np.ones(X_test.shape),axis = 1)

X_test[:,0] = X_test[:,0]-mean
X_test[:,0] = X_test[:,0]/std

test_predictions = np.dot(X_test,theta)
file = open("result_1.txt","w")

for idx in range(test_predictions.shape[0]):
    file.write(str(test_predictions[idx][0])+str("\n"))

file.close()
