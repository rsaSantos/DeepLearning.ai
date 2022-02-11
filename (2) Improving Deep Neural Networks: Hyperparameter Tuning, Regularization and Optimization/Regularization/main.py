# Objectives
#   Understand the different regularization methods that could help your model.
#   Implement dropout and see it work on data.
#   Recognize that a model without regularization gives you a better accuracy on 
#       the training set but nor necessarily on the test set.
#   Understand that you could use both dropout and regularization on your model.

# import packages
import numpy as np
import matplotlib.pyplot as plt
from reg_utils import sigmoid, relu, plot_decision_boundary, initialize_parameters, load_2D_dataset, predict_dec
from reg_utils import compute_cost, predict, forward_propagation, backward_propagation, update_parameters
import sklearn
import sklearn.datasets
import scipy.io
from testCases import *

plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


# Problem Statement: 
#   You have just been hired as an AI expert by the French Football Corporation. 
#   They would like you to recommend positions where France's goal keeper should kick the 
#       ball so that the French team's players can then hit it with their head.


# Each dot represents a position on the football field where a football player has hit the 
#   ball with his/her head after the French goal keeper has shot the ball from the left side 
#   of the football field.
#
# If the dot is blue, it means the French player managed to hit the ball with his/her head
# If the dot is red, it means the other team's player hit the ball with their head

train_X, train_Y, test_X, test_Y = load_2D_dataset()
plt.show()

# Goal:
# Use a deep learning model to find the positions on the field where the goalkeeper should 
#   kick the ball.

# Analysis of the dataset: 
#   This dataset is a little noisy, but it looks like a diagonal line separating the upper 
#   left half (blue) from the lower right half (red) would work well.
#
# First let's try a non-regularized model. Then you'll learn how to regularize it and decide 
#   which model you will choose to solve the French Football Corporation's problem.

# The model can be used
#   in regularization mode -- by setting the lambd input to a non-zero value. 
#       We use "lambd" instead of "lambda" because "lambda" is a reserved keyword in Python.
#   in dropout mode -- by setting the keep_prob to a value less than one
#
# First without regularization, then
#   L2 regularization -- functions: "compute_cost_with_regularization()" and 
#   "backward_propagation_with_regularization()"
#   
#   Dropout -- functions: "forward_propagation_with_dropout()" and 
#       "backward_propagation_with_dropout()"
#

# model
def model(X, Y, learning_rate = 0.3, num_iterations = 30000, print_cost = True, lambd = 0, keep_prob = 1):
    """
    Implements a three-layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (output size, number of examples)
    learning_rate -- learning rate of the optimization
    num_iterations -- number of iterations of the optimization loop
    print_cost -- If True, print the cost every 10000 iterations
    lambd -- regularization hyperparameter, scalar
    keep_prob - probability of keeping a neuron active during drop-out, scalar.
    
    Returns:
    parameters -- parameters learned by the model. They can then be used to predict.
    """
        
    grads = {}
    costs = []                            # to keep track of the cost
    m = X.shape[1]                        # number of examples
    layers_dims = [X.shape[0], 20, 3, 1]
    
    # Initialize parameters dictionary.
    parameters = initialize_parameters(layers_dims)

    # Loop (gradient descent)

    for i in range(0, num_iterations):

        # Forward propagation: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
        if keep_prob == 1:
            a3, cache = forward_propagation(X, parameters)
        elif keep_prob < 1:
            a3, cache = forward_propagation_with_dropout(X, parameters, keep_prob)
        
        # Cost function
        if lambd == 0:
            cost = compute_cost(a3, Y)
        else:
            cost = compute_cost_with_regularization(a3, Y, parameters, lambd)
            
        # Backward propagation.
        assert(lambd==0 or keep_prob==1)    # it is possible to use both L2 regularization and dropout, 
                                            # but this assignment will only explore one at a time
        if lambd == 0 and keep_prob == 1:
            grads = backward_propagation(X, Y, cache)
        elif lambd != 0:
            grads = backward_propagation_with_regularization(X, Y, cache, lambd)
        elif keep_prob < 1:
            grads = backward_propagation_with_dropout(X, Y, cache, keep_prob)
        
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # Print the loss every 10000 iterations
        if print_cost and i % 10000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
        if print_cost and i % 1000 == 0:
            costs.append(cost)
    
    # plot the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (x1,000)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters


# Treinar o modelo sem regularização
parameters = model(train_X, train_Y)
print ("On the training set:")
predictions_train = predict(train_X, train_Y, parameters)
print ("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)

# The train accuracy is 94.8% while the test accuracy is 91.5%. 
# This is the baseline model (you will observe the impact of regularization on this model). 
plt.title("Model without regularization")
axes = plt.gca()
axes.set_xlim([-0.75,0.40])
axes.set_ylim([-0.75,0.65])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)

# The non-regularized model is obviously overfitting the training set. 
# It is fitting the noisy points! Lets now look at two techniques to reduce overfitting.


# The standard way to avoid overfitting is called L2 regularization
#   Consiste em acrescentar um parâmetro à função de custo -> 1/m * lambda/2 * weights
#
# Implement compute_cost_with_regularization()
def compute_cost_with_regularization(A3, Y, parameters, lambd):
    """
    Implement the cost function with L2 regularization. See formula (2) above.
    
    Arguments:
    A3 -- post-activation, output of forward propagation, of shape (output size, number of examples)
    Y -- "true" labels vector, of shape (output size, number of examples)
    parameters -- python dictionary containing parameters of the model
    
    Returns:
    cost - value of the regularized loss function (formula (2))
    """
    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    
    cross_entropy_cost = compute_cost(A3, Y) # This gives you the cross-entropy part of the cost
    
    L2_regularization_cost = (1/m) * (lambd/2) * ( np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)) )
    
    cost = cross_entropy_cost + L2_regularization_cost
    
    return cost



# Of course, because you changed the cost, you have to change backward propagation as well! 
# All the gradients have to be computed with respect to this new cost. 

def backward_propagation_with_regularization(X, Y, cache, lambd):
    """
    Implements the backward propagation of our baseline model to which we added an L2 regularization.
    
    Arguments:
    X -- input dataset, of shape (input size, number of examples)
    Y -- "true" labels vector, of shape (output size, number of examples)
    cache -- cache output from forward_propagation()
    lambd -- regularization hyperparameter, scalar
    
    Returns:
    gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
    """
    
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache
    
    dZ3 = A3 - Y
    
    dW3 = 1./m * np.dot(dZ3, A2.T) + (lambd/m) * W3
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)
    
    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1./m * np.dot(dZ2, A1.T) + (lambd/m) * W2
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)
    
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1./m * np.dot(dZ1, X.T) + (lambd/m) * W1
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)
    
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}
    
    return gradients


# Let's run the model with L2 regularization (lambda = 0.7)
parameters = model(train_X, train_Y, lambd = 0.7)
print ("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print ("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)

# Plot the data - see that there is no overfitting
plt.title("Model with L2-regularization")
axes = plt.gca()
axes.set_xlim([-0.75,0.40])
axes.set_ylim([-0.75,0.65])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)

# Observations
#   The value of λ is a hyperparameter that you can tune using a dev set.
#   L2 regularization makes your decision boundary smoother. If λλ is too large, it is 
#       also possible to "oversmooth", resulting in a model with high bias.

# What is L2-regularization actually doing?:
# 
# L2-regularization relies on the assumption that a model with small weights is simpler than 
#   a model with large weights. Thus, by penalizing the square values of the weights in the 
#   cost function you drive all the weights to smaller values. It becomes too costly for the 
#   cost to have large weights! This leads to a smoother model in which the output changes 
#   more slowly as the input changes. 

#  What you should remember -- the implications of L2-regularization on:
#
#   The cost computation:
#       A regularization term is added to the cost
#   The backpropagation function:
#       There are extra terms in the gradients with respect to weight matrices
#   Weights end up smaller ("weight decay"):
#       Weights are pushed to smaller values.




# Dropout
#   It randomly shuts down some neurons in each iteration
#   When you shut some neurons down, you actually modify your model. 
#   The idea behind drop-out is that at each iteration, you train a different model that 
#       uses only a subset of your neurons. With dropout, your neurons thus become less sensitive 
#       to the activation of one other specific neuron, because that other neuron might be shut 
#       down at any time. 


# Implement forward propagation with dropout. We will add dropout to the first and second
#   hidden layers. We will not apply dropout to the input layer or output layer. 
#
# 



def forward_propagation_with_dropout(X, parameters, keep_prob = 0.5):
    """
    Implements the forward propagation: LINEAR -> RELU + DROPOUT -> LINEAR -> RELU + DROPOUT -> LINEAR -> SIGMOID.
    
    Arguments:
    X -- input dataset, of shape (2, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                    W1 -- weight matrix of shape (20, 2)
                    b1 -- bias vector of shape (20, 1)
                    W2 -- weight matrix of shape (3, 20)
                    b2 -- bias vector of shape (3, 1)
                    W3 -- weight matrix of shape (1, 3)
                    b3 -- bias vector of shape (1, 1)
    keep_prob - probability of keeping a neuron active during drop-out, scalar
    
    Returns:
    A3 -- last activation value, output of the forward propagation, of shape (1,1)
    cache -- tuple, information stored for computing the backward propagation
    """
    
    np.random.seed(1)
    
    # retrieve parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    
    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    
    # Steps 1-4 below correspond to the Steps 1-4 described above. 
    D1 = np.random.rand(len(A1), len(A1[0]))    # Step 1: initialize matrix D1 = np.random.rand(..., ...)
    D1 = (D1 < keep_prob).astype(int)           # Step 2: convert entries of D1 to 0 or 1 (using keep_prob as the threshold)
    A1 = np.multiply(A1, D1)                    # Step 3: shut down some neurons of A1
    A1 = A1 / keep_prob                         # Step 4: scale the value of neurons that haven't been shut down
    
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    
    D2 = np.random.rand(len(A2), len(A2[0]))    # Step 1: initialize matrix D2 = np.random.rand(..., ...)
    D2 = (D2 < keep_prob).astype(int)           # Step 2: convert entries of D2 to 0 or 1 (using keep_prob as the threshold)
    A2 = np.multiply(A2, D2)                    # Step 3: shut down some neurons of A2
    A2 = A2 / keep_prob                         # Step 4: scale the value of neurons that haven't been shut down
    
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)
    
    cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)
    
    return A3, cache



# Back-Prop with dropout - only in first and second hidden layers
#   You had previously shut down some neurons during forward propagation, by applying a 
#   mask D[1] to A1. In backpropagation, you will have to shut down the same neurons, by 
#   reapplying the same mask D[1] to dA1.
# 
# During forward propagation, you had divided A1 by keep_prob. In backpropagation, you'll 
#   therefore have to divide dA1 by keep_prob again (the calculus interpretation is that if 
#   A[1] is scaled by keep_prob, then its derivative dA[1]dA[1] is also scaled by the same 
#   keep_prob).

def backward_propagation_with_dropout(X, Y, cache, keep_prob):
    """
    Implements the backward propagation of our baseline model to which we added dropout.
    
    Arguments:
    X -- input dataset, of shape (2, number of examples)
    Y -- "true" labels vector, of shape (output size, number of examples)
    cache -- cache output from forward_propagation_with_dropout()
    keep_prob - probability of keeping a neuron active during drop-out, scalar
    
    Returns:
    gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
    """
    
    m = X.shape[1]
    (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache
    
    dZ3 = A3 - Y
    dW3 = 1./m * np.dot(dZ3, A2.T)
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)
    dA2 = np.dot(W3.T, dZ3)

    dA2 = np.multiply(dA2, D2)  # Step 1: Apply mask D2 to shut down the same neurons as during the forward propagation
    dA2 = dA2 / keep_prob       # Step 2: Scale the value of neurons that haven't been shut down
    
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1./m * np.dot(dZ2, A1.T)
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)
    
    dA1 = np.dot(W2.T, dZ2)

    dA1 = np.multiply(dA1, D1)              # Step 1: Apply mask D1 to shut down the same neurons as during the forward propagation
    dA1 = dA1 / keep_prob              # Step 2: Scale the value of neurons that haven't been shut down

    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1./m * np.dot(dZ1, X.T)
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)
    
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}
    
    return gradients




# Let's run the model with dropout (keep_prob = 0.86)
parameters = model(train_X, train_Y, keep_prob = 0.86, learning_rate = 0.3)

print ("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print ("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)

plt.title("Model with dropout")
axes = plt.gca()
axes.set_xlim([-0.75,0.40])
axes.set_ylim([-0.75,0.65])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)


# Dropout works great! The test accuracy has increased again (to 95%)! 
# Your model is not overfitting the training set and does a great job on the test set. 
# The French football team will be forever grateful to you! 


# NOTES
#   A common mistake when using dropout is to use it both in training and testing. 
#   You should use dropout (randomly eliminate nodes) only in training. 
#
#   Deep learning frameworks like tensorflow, PaddlePaddle, keras or caffe come with a 
#       dropout layer implementation. 
#   Don't stress - you will soon learn some of these frameworks.
#
# What to remember about dropout?
#   Dropout is a regularization technique.
#   You only use dropout during training. Don't use dropout (randomly eliminate nodes) during test time.
#   Apply dropout both during forward and backward propagation.
#   During training time, divide each dropout layer by keep_prob to keep the same expected value for the activations. For example, if keep_prob is 0.5, then we will on average shut down half the nodes, so the output will be scaled by 0.5 since only the remaining half are contributing to the solution. Dividing by 0.5 is equivalent to multiplying by 2. Hence, the output now has the same expected value. 
#       You can check that this works even when keep_prob is other values than 0.5.
#

# Conclusions
#   model 	                          | train accuracy | test accuracy
#   -------------------------------------------------------------------------
#   3-layer NN without regularization |	95% 	       | 91.5%
#   3-layer NN with L2-regularization |	94% 	       | 93%
#   3-layer NN with dropout 	      | 93% 	       | 95% 
#
# Note that regularization hurts training set performance! 
# This is because it limits the ability of the network to overfit to the training set. 
# But since it ultimately gives better test accuracy, it is helping your system.

#  What we want you to remember from this notebook:
#     Regularization will help you reduce overfitting.
#     Regularization will drive your weights to lower values.
#     L2 regularization and Dropout are two very effective regularization techniques.
#

