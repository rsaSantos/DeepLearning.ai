# import packages

import numpy as np                  # fundamental package for scientific computing with Python.
import h5py                         # package to interact with a dataset that is stored on an H5 file.
import scipy                        
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
import print_stuff as brief_overview


# Problem Statement: You are given a dataset ("data.h5") containing:
#   a training set of m_train images labeled as cat (y=1) or non-cat (y=0)
#   a test set of m_test images labeled as cat or non-cat
#   each image is of shape (num_px, num_px, 3) where 3 is for the 3 channels (RGB). Thus, each image is square (height = num_px) and (width = num_px)

# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# Find the values of 
#   m_train -> number of training examples
#   m_test  -> number of test examples
#   num_px  -> number of pixels in height/width (it's a square)

# train_set_x_orig is an numpy-array of shape (m_train, num_px, num_px, 3)
# test_set_x_orig is an numpy-array of shape (m_test, num_px, num_px, 3)
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

# A brief overview
brief_overview.ov_1(m_train, m_test, num_px, train_set_x_orig, test_set_x_orig, train_set_y, test_set_y)


# Reshape the training and test data sets so that images of size (num_px, num_px, 3) are flattened
#   into single vectors of shape (num_px ∗ num_px ∗ 3, 1).
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

# A brief overview
brief_overview.ov_2(train_set_x_flatten, train_set_y, test_set_x_flatten, test_set_y)


# At this point we have read the training and testing datasets
# We have reshaped this datasets into this format (num_px ∗ num_px ∗ 3, 1)

# To represent color images, the red, green and blue channels (RGB) must be specified for each pixel, 
#   and so the pixel value is actually a vector of three numbers ranging from 0 to 255.
# One common preprocessing step in machine learning is to center and standardize your dataset, meaning that 
#   you subtract the mean of the whole numpy array from each example, and then divide each example by the 
#   standard deviation of the whole numpy array. 
# But for picture datasets, it is simpler and more convenient and works almost as well to just divide 
#   every row of the dataset by 255 (the maximum value of a pixel channel).

#Let's standardize our dataset.
train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.



# !!!!!!!!!!!!! KEY - WHAT TO REMEMBER - KEY !!!!!!!!!!!!!
# Common steps for pre-processing a new dataset are:
#    Figure out the dimensions and shapes of the problem (m_train, m_test, num_px, ...)
#    Reshape the datasets such that each example is now a vector of size (num_px * num_px * 3, 1)
#    "Standardize" the data



# THE ALGORITHM
#   1. Define the model structure (such as number of input features)
#   2. Initialize the model's parameters
#   3. Loop:
#       Calculate current loss (forward propagation)
#       Calculate current gradient (backward propagation)
#       Update parameters (gradient descent)





# sigmoid function
def sigmoid(x):
    s = 1 / (1 + np.exp(-x))    
    return s


# Initializing parameters -> b as zero and w as a matrix of shape (dim,1) full of zeros.
def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0
    ### END CODE HERE ###

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    
    return w, b


# Now, let's start forward and backward propagation steps for learning the parameters.
# Implement a function propagate() that computes the cost function and its gradient

# Forward Propagation
def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    
    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    """
    
    m = X.shape[1]
    
    # FORWARD PROPAGATION (FROM X TO COST)
    A = sigmoid(np.dot(w.T, X)+ b)                                  # compute activation
    cost = - (1/m) * np.sum(Y * np.log(A) + (1-Y) * np.log(1 - A))  # compute cost
    

    # BACKWARD PROPAGATION (TO FIND GRAD)
    dw = (1/m) * np.dot(X, (A - Y).T) # derivative of J in order to w 
    db = (1/m) * np.sum(A - Y)        # derivative of J in order to b

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost




# Now, we've initialized the parameters, computed the cost function and it's gradient, and now
#   we will be updating the parameters using gradient descent


# Write down the optimization function. The goal is to learn w and b by minimizing the cost function J. 
# For a parameter θ, the update rule is θ = θ − α dθ, where α is the learning rate.
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    """
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps
    
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    
    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """
    
    costs = []
    
    for i in range(num_iterations):
        
        
        # Cost and gradient calculation
        grads, cost = propagate(w, b, X, Y)
        
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        # update rule
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        
        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs




# The optimize() function will output the learned w and b.
#   With those results, let's build the function predict() to label de training examples.
# Calculate Ŷ = A = sigmoid(w.T * X + b)
# Convert the entries of A to 0 (if activation < 0.5) or to 1 (if activation > 0.5)
def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    
    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    A = sigmoid(np.dot(w.T, X)+ b)
    
    # Este for-loop pode ser vetorizado
    for i in range(A.shape[1]):
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        if(A[0, i] < 0.5):
            Y_prediction[0, i] = 0
        else:
            Y_prediction[0, i] = 1
    
    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction


# !!!!!!!!!!!!! KEY - WHAT TO REMEMBER - KEY !!!!!!!!!!!!!
#  What to remember: You've implemented several functions that:
#   1. Initialize (w,b)
#   2. Optimize the loss iteratively to learn parameters (w,b):
#        computing the cost and its gradient
#        updating the parameters using gradient descent
#   3. Use the learned (w,b) to predict the labels for a given set of examples



# Final Step - Merge all functions into a model
# Implement the model function. Use the following notation:
#    Y_prediction_test for your predictions on the test set
#    Y_prediction_train for your predictions on the train set
#    w, costs, grads for the outputs of optimize()

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    """
    Builds the logistic regression model by calling the function you've implemented previously
    
    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations
    
    Returns:
    d -- dictionary containing information about the model.
    """
        
    # initialize parameters with zeros
    w, b = initialize_with_zeros(X_train.shape[0])

    # Gradient descent
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    
    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]
    
    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)


    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d


# Run the model
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)


# Final comment
# Training accuracy is close to 100%. This is a good sanity check: your model is working and has high 
#   enough capacity to fit the training data. Test accuracy is 68%. It is actually not bad for this 
#   simple model, given the small dataset we used and that logistic regression is a linear classifier. 
# But no worries, you'll build an even better classifier next week!

# Also, you see that the model is clearly overfitting the training data. 
# Later in this specialization you will learn how to reduce overfitting, for example by using 
#   regularization.

# WHAT THE HELL IS OVERFITTING?
#   If we increase the number of iterations the training set accuracy goes up BUT the test set
#       accuracy goes down.



# Further analysis

# In order for Gradient Descent to work you must choose the learning rate wisely.
# The learning rate α determines how rapidly we update the parameters.
# If the learning rate is too large we may "overshoot" the optimal value.
# Similarly, if it is too small we will need too many iterations to converge to the best values. 
# That's why it is crucial to use a well-tuned learning rate.


# Let's observe the learning curve for different α values.

# learning rate is: 0.01
# train accuracy: 99.52153110047847 %
# test accuracy: 68.0 %

# -------------------------------------------------------

# learning rate is: 0.001
# train accuracy: 88.99521531100478 %
# test accuracy: 64.0 %

# -------------------------------------------------------

# learning rate is: 0.0001
# train accuracy: 68.42105263157895 %
# test accuracy: 36.0 %

# -------------------------------------------------------

# Different learning rates give different costs and thus different predictions results.
# If the learning rate is too large (0.01), the cost may oscillate up and down. 
#   It may even diverge (though in this example, using 0.01 still eventually ends up at a good value 
#   for the cost).
# A lower cost doesn't mean a better model. You have to check if there is possibly overfitting.
#   It happens when the training accuracy is a lot higher than the test accuracy.
# In deep learning, we usually recommend that you:
#       Choose the learning rate that better minimizes the cost function.
#       If your model overfits, use other techniques to reduce overfitting. 
#           (We'll talk about this in later videos.)


#  What to remember from this assignment:
#    1. Preprocessing the dataset is important.
#    2. You implemented each function separately: initialize(), propagate(), optimize(). 
#           Then you built a model().
#    3. Tuning the learning rate (which is an example of a "hyperparameter") can make a big difference 
#           to the algorithm. You will see more examples of this later in this course!


# Things to play with! (except the little thing)
#   - Play with the learning rate and the number of iterations
#   - Try different initialization methods and compare the results
#   - Test other preprocessings (center the data, or divide each row by its standard deviation)