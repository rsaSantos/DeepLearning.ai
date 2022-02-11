# Welcome to your third programming exercise of the deep learning specialization. 
# You will implement all the building blocks of a neural network and use these building blocks in the 
#   next assignment to build a neural network of any architecture you want. 
# 
# By completing this assignment you will:
#   - Develop an intuition of the over all structure of a neural network.
#   - Write functions (e.g. forward propagation, backward propagation, logistic loss, etc...) that would 
#           help you decompose your code and ease the process of building a neural network.
#   - Initialize/update parameters according to your desired structure. 


# Import packages

import numpy as np
import h5py
import matplotlib.pyplot as plt
from testCases_v4a import *
from dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


np.random.seed(1) # In order to get similar results with the grader.

# Overview of the assignment
#
# 1. Initialize the parameters for a two-layer network and for an L-layer neural network.
# 2. Implement the forward propagation module.
#       - Complete the LINEAR part of a layer's forward propagation step (resulting in Z[l]).
#       - We give you the ACTIVATION function (relu/sigmoid).
#       - Combine the previous two steps into a new [LINEAR->ACTIVATION] forward function.
#       - Stack the [LINEAR->RELU] forward function L-1 time (for layers 1 through L-1) and add a 
#           [LINEAR->SIGMOID] at the end (for the final layer LL). This gives you a new L_model_forward function.
# 3. Compute the Loss
# 4. Implement the backward propagation module
#       - Complete the LINEAR part of a layer's backward propagation step.
#       - We give you the gradient of the ACTIVATE function (relu_backward/sigmoid_backward)
#       - Combine the previous two steps into a new [LINEAR->ACTIVATION] backward function.
#       - Stack [LINEAR->RELU] backward L-1 times and add [LINEAR->SIGMOID] backward in a new L_model_backward function
# 5. Update the parameters.
#
# Nota: for every forward function, there is a corresponding backward function. That is why at every 
#       step of your forward module you will be storing some values in a cache. The cached values are 
#       useful for computing gradients. In the backpropagation module you will then use the cache to 
#       calculate the gradients. This assignment will show you exactly how to carry out each of these steps. 
#
#
# Check image -> final_outline.png



# 1. Inicializar os parâmetros de uma rede neuronal L = 2.
def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    parameters -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    
    np.random.seed(1)
    
    W1 = np.random.randn(n_h, n_x) * 0.01   # Guarantee lower random initial values for faster and diverse learning.
    b1 = np.zeros((n_h,1))                  # Random numbers not needed.
    W2 = np.random.randn(n_y, n_h) * 0.01   # Guarantee lower random initial values for faster and diverse learning.
    b2 = np.zeros((n_y,1))                  # Random numbers not needed.
    
    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters  


# 1. Inicializar os parâmetros para L layers.
#
# Instructions
#   - The model's structure is [LINEAR -> RELU] ×× (L-1) -> LINEAR -> SIGMOID. I.e., it has L−1L−1 layers 
#       using a ReLU activation function followed by an output layer with a sigmoid activation function.
#   - Use random initialization for the weight matrices. Use np.random.randn(shape) * 0.01.
#   - Use zeros initialization for the biases. Use np.zeros(shape).
#   - We will store n[l], the number of units in different layers, in a variable layer_dims. For example, 
#       the layer_dims for the "Planar Data classification model" from last week would have been [2,4,1]: 
#       There were two inputs, one hidden layer with 4 hidden units, and an output layer with 1 output unit. 
#       This means W1's shape was (4,2), b1 was (4,1), W2 was (1,4) and b2 was (1,1). Now you will generalize this to LL layers!
#   - Here is the implementation for L=1 (one layer neural network). It should inspire you to implement the general case (L-layer neural network).
#       if L == 1:
#           parameters["W" + str(L)] = np.random.randn(layer_dims[1], layer_dims[0]) * 0.01
#           parameters["b" + str(L)] = np.zeros((layer_dims[1], 1))


#
#
def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    
    np.random.seed(3)
    parameters = {}                # Dictionary with parameters
    L = len(layer_dims)            # number of layers in the network

    # Iterate through the L layers, initializing parameters W randomly and b with zeros.
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        
    return parameters



# 2. Forward Propagation module (in 3 steps/functions).
#
#       - LINEAR
#       - LINEAR -> ACTIVATION where ACTIVATION is ReLU or Sigmoid
#       - [LINEAR -> ReLU] × LINEAR -> Sigmoid (whole model)

# 1st function of forward propagation module -> Z[l] = W[l]A[l-1] + b[l]
def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    
    Z = np.dot(W,A) + b
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache


# 2nd function of forward propagation module -> A[l] = g(Z[l]) = g(W[l]A[l-1] + b[l])
def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python tuple containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
    
    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    
    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache


# 3rd function of forward propagation module -> A[L] = sigmoid(Z[L]) = sigmoid(W[L]A[L-1] + b[L])
def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    
    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A, parameters["W" + str(l)], parameters["b" + str(l)], "relu")
        caches.append(cache)
    
    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid")
    caches.append(cache)
    
    assert(AL.shape == (1,X.shape[1]))
            
    return AL, caches

# Forward propagation module is now complete.


# 3. Compute the Loss
def compute_cost(AL, Y):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    
    m = Y.shape[1]

    # Compute loss from aL and y.
    cost = -1/m * np.sum( np.multiply(np.log(AL),Y) + np.multiply(np.log(1 - AL), (1-Y)) )
    
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())
    
    return cost


# Backward Propagation module (check image backprop)
# We are going to build back prop in three steps.
#   - LINEAR backward
#   - LINEAR -> ACTIVATION backward where ACTIVATION computes the derivative of either the ReLU or sigmoid activation
#   - [LINEAR -> RELU] × (L-1) -> LINEAR -> SIGMOID backward (whole model)

# 1st function of backward propagation module -> Z[l] = W[l]A[l-1] + b[l] (forward)
#                   (we need to compute the integrals of dW[l], db[l] and dA[l-1])
def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1/m * np.dot(dZ, A_prev.T)
    db = 1/m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db


# 2nd function of backward propagation module -> dZ = dA[l] * g'(Z[l]) = (relu/sigmoid)_backward(dA, cache)
def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db


# 3rd function of backward propagation module ->
# A[L] = sigmoid(Z[L]), we have to compute dA[L]...
#   Use... dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) # derivative of cost with respect to AL
    
    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid")
    
    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+1)], current_cache, "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


# 5. Update the parameters => W[l+1] = W[l] - alfa * dW[l], ...
def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
    return parameters


# Conclusion
#  The functions are implemented. Now let's build the models to run this thing.