#!/usr/bin/env python
# coding: utf-8

# In[46]:


import numpy as np
import sys
from sklearn.utils import shuffle
from time import time
from scipy import special
import matplotlib.pyplot as plt


# In[2]:


np.random.seed(seed=0)


# ## Loading Data

# In[3]:

train_file = sys.argv[1]
test_file = sys.argv[2]

# Train data
data_train = np.loadtxt(train_file, delimiter=',')
x_train = data_train[:, 0: -1]
y_train = data_train[:, -1]


# In[4]:


# Test data
data_test = np.loadtxt(test_file, delimiter=',')
x_test = data_test[:, 0: -1]
y_test = data_test[:, -1]


# ### Part A

# In[5]:


class NeuralNetwork:
    
    def __init__(self, batch_size, 
                 num_features, 
                 hidden_layers, 
                 target_classes, 
                 learning_rate, 
                 activation = 'sigmoid', 
                 random = False, 
                 max_epochs = 1000,
                 learning_rate_type='constant',
                 n_iter_no_change = 10,
                 EPSILON = 1e-4,
                 verbose = True):
        
        # Initialisation
        self.batch_size = batch_size
        self.num_features = num_features
        self.target_classes = target_classes
        self._activation = activation
        self._learning_rate = learning_rate
        self._base_learning_rate = learning_rate
        self._learning_rate_type = learning_rate_type
        self.max_epochs = max_epochs
        self.n_iter_no_change = n_iter_no_change
        self.EPSILON = EPSILON
        self.verbose = verbose
        
        # Layers is dictionary containing different values
        self.layers = {}
        self.layers['output'] = []
        self.layers['params'] = []
        self.layers['delta'] = []
        # Output before activation is applied        
        self.layers['preactivation'] = []
        
        if random:
            # If random initialisation is true
            self._rand_init(hidden_layers)
        else: 
            self._zero_init(hidden_layers)
        
    def __range(self, x):
        return np.sqrt(2 / x);
    
    # Initialise layers randomly
    def _rand_init(self, hidden_layers):
                
        # Initialising biases and params as randaom
        curr_layer = np.random.randn(hidden_layers[0], self.num_features + 1) * self.__range(self.num_features + 1)
        self.layers['params'] = [curr_layer]
        
        for i in range(1, len(hidden_layers)):
            
            # Values for a layer
            num_perceptrons = hidden_layers[i]
            num_params = hidden_layers[i - 1]
            
            curr_layer = np.random.randn(num_perceptrons, num_params + 1) * self.__range(num_params + 1)            
            self.layers['params'].append(curr_layer)
            
        # Output layer
        curr_layer = np.random.randn(self.target_classes, hidden_layers[len(hidden_layers) - 1] + 1) * self.__range(hidden_layers[len(hidden_layers) - 1] + 1)
        self.layers['params'].append(curr_layer) 
        
    
    # Initialise layers with 0
    def _zero_init(self, hidden_layers):
        
        
        self.layers['params'] = [np.zeros((hidden_layers[0], self.num_features + 1))]
        for i in range(1, len(hidden_layers)):
            
            # Values for a layer
            num_perceptrons = hidden_layers[i]
            num_params = hidden_layers[i - 1] + 1
            
            self.layers['params'].append(np.zeros((num_perceptrons, num_params)))
        
        # Output layer
        self.layers['params'].append(np.zeros((self.target_classes, hidden_layers[len(hidden_layers) - 1] + 1))) 
    
    # Activation function
    def _activate(self, x, is_hidden):
        
        # If it is the hidden layer
        if is_hidden:
            if self._activation == 'sigmoid':
                return self._sigmoid(x)
            else:
                return self._relu(x)
        else:
            # Output layer has sigmoid activation
            return self._sigmoid(x)
        
    # Derivative of the acivation function
    def _der_activation(self, x, is_hidden):
        # If it is the hidden layer
        if is_hidden:
            if self._activation == 'sigmoid':
                return self._der_sigmoid(x)
            else:
                return self._der_relu(x)
        else:
            # Output layer has sigmoid activation
            return self._der_sigmoid(x)
    
    # Relu
    def _relu(self, x):
        return np.maximum(0, x)
    
    # Derivative of Relu
    def _der_relu(self, x):
        return (np.ones_like(x) * (x >= 0))
        
    # Sigmoid
    def _sigmoid(self, x):
        return special.expit(x)
    
    # Derivative of sigmoid
    def _der_sigmoid(self, x):
        s = self._sigmoid(x)
        return (s * (1 - s))
    
    # Returning the predictions
    def predict(self, x, debug = False):
        
        pred = self.forward(x, debug)
        pred = np.argmax(pred, axis=0)    
        return pred
    
    # Passing data in the forward direction
    def forward(self, x, debug = False):
        
        layer_input = np.array(x)
        self.layers['preactivation'] = []
        self.layers['output'] = []
        
        #If layer is hidden layer
        is_hidden = True
        
        for (i, layer) in enumerate(self.layers['params']):
            
            if i == len(self.layers['params']) - 1:
                is_hidden = False

            # Appending one for bias term
            layer_input = np.append(layer_input, np.ones((1, layer_input.shape[1])), axis = 0)

            # Output before activation
            z = layer @ layer_input
            self.layers['preactivation'].append(z)

            if debug:
                print(z)

            layer_input = self._activate(z, is_hidden)

            # Output of this layer is input of next
            self.layers['output'].append(layer_input)
        
        return layer_input
        
        
    # Back propogating derivatives
    def back_propogate(self, y):
        
        # y is such that ith column contains the ith sample
        num_layers = len(self.layers['params'])
        
        # Initialising delta list
        self.layers['delta'] = []
        
        # To check if we are on a hidden layer
        is_hidden = False
        
        # Running the loop in reverse 
        for i in range(num_layers):
            
            # Output of current layer
            layer_output = self.layers['output'][num_layers - i - 1]
            layer_output_preactivation = self.layers['preactivation'][num_layers - i - 1]
            
            if not is_hidden:
                
                # Output Layer
                temp = (y - layer_output) * self._der_activation(layer_output_preactivation, is_hidden=is_hidden)
                
                self.layers['delta'].append(temp)
                                       
                # switching is hidden
                is_hidden = True
                
            else:
                
                # delta of next layer
                del_next = self.layers['delta'][i - 1]
                
                # Theta or params of next layer without bias
                theta_next = self.layers['params'][num_layers - i][:, 0 : -1]
                del_theta = theta_next.T @ del_next
                temp = del_theta * self._der_activation(layer_output_preactivation, is_hidden=is_hidden)
                
                self.layers['delta'].append(temp)
                
        self.layers['delta'].reverse()
    
    def _cost(self, x, y):
        
        pred = self.forward(x)
        cost = ((y - pred) ** 2)
        
        cost = np.sum(cost)
        cost /= (2 * y.shape[-1])
        
        return cost
        
    
    
    # Train the model
    def train(self, x, y):
        
        total_samples = y.shape[-1]
        
        diff = np.inf
        old_cost = 0
        new_cost = 0
        num_change = 0
        
        count = 0
        
        while True:
            if count == self.max_epochs:
                pred = self.predict(x)
                y_temp = np.argmax(y, axis=0) 
                acc = np.sum(self.predict(x) == y_temp) / y_temp.shape[0]
                print("Max iterations reached | Training accuracy => {}".format(acc))
                return
            
            # Old cost
            old_cost = new_cost
            
            if self.verbose:
                print("#Epoch => {} | Cost: {} | Difference in cost => {}".format(count, old_cost, diff))
            
            count += 1
            
            for i in range(0, total_samples, self.batch_size):
                
                x_curr = np.array(x[:, i : i + self.batch_size])
                y_curr = np.array(y[:, i : i + self.batch_size])
                
                gradient = []
                
                # Initialising the gradient shape
                for layer in self.layers['params']:
                    gradient.append(np.zeros_like(layer))
                  
                  
                # Assigning values to outputs and deltas
                self.forward(x_curr)
                self.back_propogate(y_curr)
                
                # Computing gradient layer wise
                for i in range(len(self.layers['params'])):

                    # Updating for each layer
                    if i == 0:
                        # If its the first layer
                        x_i_k = np.array(x_curr)
                    else:    
                        x_i_k = np.array(self.layers['output'][i - 1])

                    # 1 for bias
                    x_i_k = np.append(x_i_k, np.ones((1, x_i_k.shape[1])), axis = 0)
                    delta = np.array(self.layers['delta'][i])

                    gradient[i] = ((-delta) @ x_i_k.T) / self.batch_size
                
                # If learning rate is adaptive
                if self._learning_rate_type == 'adaptive':
                    self._learning_rate = self._base_learning_rate / np.sqrt(count)
                
                # Updating parameters layer wise
                for (i, layer) in enumerate(self.layers['params']):  
                    self.layers['params'][i] = self.layers['params'][i] - self._learning_rate * gradient[i]
                
    
            # New cost
            new_cost = self._cost(x, y)
            diff = old_cost - new_cost
            
            # Difference in cost
            if diff < self.EPSILON:
                num_change += 1
            else:
                num_change = 0
            
            # If change less than EPSILON for n_iter_no_change consecutive iterations
            if num_change == self.n_iter_no_change:
                
                pred = self.predict(x)
                y_temp = np.argmax(y, axis=0)
                pred = self.predict(x)
                y_temp = np.argmax(y, axis=0) 
                acc = np.sum(self.predict(x) == y_temp) / y_temp.shape[0]

                print("Training complete | Epochs => {} | Training accuracy => {}".format(count, acc))

                return
                    
        


# In[6]:


def get_one_hot_encoding(y, max_range):
    
    ans = np.zeros((max_range, y.shape[0]))
    for (i, val) in enumerate(y):
        ans[int(val), i] = 1
        
    return ans                     


# ### Part B

# In[7]:


# Modifying training data to send to model and shuffling
x_train_, y_train_ = shuffle(x_train, y_train)
x_train_new = np.array(x_train_.T) / 255
y_train_new = get_one_hot_encoding(y_train_, 26)
x_test_new = np.array(x_test.T) / 255


# In[8]:


hidden_layer = [1, 5, 10, 50, 100]


# In[9]:


# To get the accuracies and training time for the hidden layer sizes
def get_accuracies(hidden_layer, EPSILON = 1e-4, learning_rate_type = 'constant', learning_rate = 0.1):
    
    # Initialising lists
    training_time = []
    acc_train = []
    acc_test = []
    
    for h in hidden_layer:
        print("Training model: ", h)
        model = NeuralNetwork(batch_size=100, 
                              num_features=x_train.shape[-1], 
                              hidden_layers = [h], 
                              target_classes= 26, 
                              learning_rate=learning_rate, 
                              learning_rate_type=learning_rate_type,
                              random=True,
                              EPSILON=EPSILON,
                              verbose=True)
        t0 = time()
        model.train(x_train_new, y_train_new)

        # Training time
        training_time.append(time() - t0)

        # Predicting
        pred_train = model.predict(x_train_new)
        pred_test = model.predict(x_test_new)

        # Accuracy over different data sets
        train_acc = np.sum(pred_train == y_train_) / y_train_.shape[0]
        test_acc = np.sum(pred_test == y_test) / y_test.shape[0]

        acc_train.append(train_acc)
        acc_test.append(test_acc)
    return acc_train, acc_test, training_time


# In[10]:


acc_train, acc_test, training_time = get_accuracies(hidden_layer=hidden_layer)


# In[11]:


# Plotting the accuracies
def plot_accuracies(acc_train, acc_test, hidden_layer):
    plt.plot(hidden_layer, acc_train, label = "train", color='red')
    plt.plot(hidden_layer, acc_test, label = "test", color='blue')
    plt.scatter(hidden_layer, acc_train, color = 'red')
    plt.scatter(hidden_layer, acc_test, color='blue')
    plt.xlabel("Number of units in hidden layer")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Number of units in hidden layer")
    plt.legend()
    plt.show()
plot_accuracies(acc_train, acc_test, hidden_layer)


# In[12]:


# Plotting training time
def plot_training_time(training_time, hidden_layer):
    plt.plot(hidden_layer, training_time, color='red')
    plt.scatter(hidden_layer, training_time, color='red')
    plt.xlabel("Number of units in hidden layer")
    plt.ylabel("Training time(in s)")
    plt.title("Training time vs Number of units in hidden layer")
    plt.show()
plot_training_time(training_time, hidden_layer)


# In[13]:


print(acc_train)
print(acc_test)
print(training_time)


# ### Part C

# In[14]:


acc_train_b, acc_test_b, training_time_b = get_accuracies(hidden_layer=hidden_layer,
                                                          EPSILON=1e-4,
                                                          learning_rate_type='adaptive',
                                                          learning_rate=0.5)


# In[15]:


plot_accuracies(acc_train_b, acc_test_b, hidden_layer)


# In[16]:


plot_training_time(training_time_b, hidden_layer)


# In[17]:


print(acc_train_b)
print(acc_test_b)
print(training_time_b)


# ### Part D

# In[18]:


# Sigmoid
model_sig = NeuralNetwork(batch_size=100, 
                              num_features=x_train.shape[-1], 
                              hidden_layers = [100, 100], 
                              target_classes= 26, 
                              learning_rate=0.5,
                              random=True,
                              learning_rate_type='adaptive',
                              EPSILON=1e-4)


# In[19]:


get_ipython().run_cell_magic('time', '', 'model_sig.train(x_train_new, y_train_new)')


# In[20]:


pred_train = model_sig.predict(x_train_new)
pred_test = model_sig.predict(x_test_new)

# Accuracy over different data sets
train_acc = np.sum(pred_train == y_train_) / y_train_.shape[0]
test_acc = np.sum(pred_test == y_test) / y_test.shape[0]

print("Training accuracy: ", train_acc)
print("Test accuracy: ", test_acc)


# In[37]:


model_rel = NeuralNetwork(batch_size=100, 
                              num_features=x_train.shape[-1], 
                              hidden_layers = [100, 100], 
                              target_classes= 26, 
                              learning_rate=0.5, 
                              random=True,
                              activation = 'relu',
                              learning_rate_type='adaptive',
                              EPSILON=1e-4)


# In[38]:


get_ipython().run_cell_magic('time', '', 'model_rel.train(x_train_new, y_train_new)')


# In[39]:


pred_train = model_rel.predict(x_train_new)
pred_test = model_rel.predict(x_test_new)

# Accuracy over different data sets
train_acc = np.sum(pred_train == y_train_) / y_train_.shape[0]
test_acc = np.sum(pred_test == y_test) / y_test.shape[0]

print("Training accuracy: ", train_acc)
print("Test accuracy: ", test_acc)


# ### Part E

# In[66]:


from sklearn.neural_network import MLPClassifier


# In[67]:


clf = MLPClassifier(solver='sgd', 
                    batch_size=100,
                    learning_rate= 'invscaling',
                    learning_rate_init=0.5,
                    hidden_layer_sizes=(100, 100),
                    activation='relu',
                    power_t=0.5,
                    verbose=True,
                    max_iter=5000,
                    momentum=0,
                    nesterovs_momentum=False,
                    alpha=0,
                    shuffle=False,
                    random_state=0,
                    )


# In[68]:


get_ipython().run_cell_magic('time', '', 'clf.fit(x_train_new.T, y_train_new.T)')


# In[69]:


pred_train = clf.predict_proba(x_train_new.T)
pred_test = clf.predict_proba(x_test_new.T)

# Converting to 
pred_train = np.argmax(pred_train, axis=1)
pred_test = np.argmax(pred_test, axis=1)

# Accuracy over different data sets
train_acc = np.sum(pred_train == y_train_) / y_train_.shape[0]
test_acc = np.sum(pred_test == y_test) / y_test.shape[0]

print("Training accuracy: ", train_acc)
print("Test accuracy: ", test_acc)


# In[73]:


# Constant learning rate
clf2 = MLPClassifier(solver='sgd', 
                    batch_size=100,
                    learning_rate= 'constant',
                    learning_rate_init=0.1,
                    hidden_layer_sizes=(100, 100),
                    activation='relu',
                    power_t=0.5,
                    verbose=True,
                    max_iter=5000,
                    momentum=0,
                    nesterovs_momentum=False,
                    alpha=0,
                    shuffle=False, 
                    random_state=0,
                    )


# In[74]:


get_ipython().run_cell_magic('time', '', 'clf2.fit(x_train_new.T, y_train_new.T)')


# In[75]:


pred_train = clf2.predict_proba(x_train_new.T)
pred_test = clf2.predict_proba(x_test_new.T)

# Converting to prediction
pred_train = np.argmax(pred_train, axis=1)
pred_test = np.argmax(pred_test, axis=1)

# Accuracy over different data sets
train_acc = np.sum(pred_train == y_train_) / y_train_.shape[0]
test_acc = np.sum(pred_test == y_test) / y_test.shape[0]

print("Training accuracy: ", train_acc)
print("Test accuracy: ", test_acc)


# In[ ]:



