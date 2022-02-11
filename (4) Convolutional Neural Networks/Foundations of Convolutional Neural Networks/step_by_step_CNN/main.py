# Welcome to Course 4's first assignment! 
# In this assignment, you will implement 
#   convolutional (CONV) and pooling (POOL) layers in numpy, 
#   including both forward propagation and (optionally) backward propagation. 
#
# By the end of this notebook, you'll be able to:
#   Explain the convolution operation
#   Apply two different types of pooling operation
#   Identify the components used in a convolutional neural network (padding, stride, filter, ...) and their purpose
#   Build a convolutional neural network
#


# Packages
import numpy as np  # numpy is the fundamental package for scientific computing with Python.
import h5py
import matplotlib.pyplot as plt # matplotlib is a library to plot graphs in Python.
from public_tests import *

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1) # np.random.seed(1) is used to keep all the random function calls consistent. This helps to grade your work.


# Outline of the assignment
#   You will be implementing the building blocks of a convolutional neural network!
#       Convolution functions, including:
#           Zero Padding
#           Convolve window
#           Convolution forward
#           Convolution backward (optional)
#       Pooling functions, including:
#           Pooling forward
#           Create mask
#           Distribute value
#           Pooling backward (optional)
#
# Note: For every forward function, there is a corresponding backward equivalent. 
# Hence, at every step of your forward module you will store some parameters in a cache. 
# These parameters are used to compute gradients during backpropagation. 
#




# Zero padding - adding zeros to the border of an image.
#   Advantages
#       It allows you to use a CONV layer without necessarily shrinking the height and width of the volumes. 
#       This is important for building deeper networks, since otherwise the height/width would shrink as you go to deeper layers. 
#       An important special case is the "same" convolution, in which the height/width is exactly preserved after one layer.
#       
#       It helps us keep more of the information at the border of an image. 
#       Without padding, very few values at the next layer would be affected by pixels at the edges of an image.
#
def zero_pad(X, pad):
    """
    Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image, 
    as illustrated in Figure 1.
    
    Argument:
    X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions
    
    Returns:
    X_pad -- padded image of shape (m, n_H + 2 * pad, n_W + 2 * pad, n_C)
    """
    
    X_pad = np.pad(X, ((0,0), (pad,pad), (pad,pad), (0,0)), mode='constant', constant_values = (0,0))

    return X_pad



# Single step of convolution
#   Takes an input volume
#   Applies a filter at every position of the input
#   Outputs another volume (usually of different size)
#
#   Note: The variable b will be passed in as a numpy array. 
#   If you add a scalar (a float or integer) to a numpy array, the result is a numpy array. 
#   In the special case of a numpy array containing a single value, you can cast it as a float to convert it to a scalar.
#
def conv_single_step(a_slice_prev, W, b):
    """
    Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation 
    of the previous layer.
    
    Arguments:
    a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
    W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
    b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)
    
    Returns:
    Z -- a scalar value, the result of convolving the sliding window (W, b) on a slice x of the input data
    """

    # Element-wise product between a_slice_prev and W. Do not add the bias yet.
    s = np.multiply(a_slice_prev, W)
    # Sum over all entries of the volume s.
    Z = np.sum(s)
    # Add bias b to Z. Cast b to a float() so that Z results in a scalar value.
    Z = Z + float(b)

    return Z


# Implement the function below to convolve the filters W on an input activation A_prev.
# This function takes the following inputs:
#   A_prev, the activations output by the previous layer (for a batch of m inputs);
#   Weights are denoted by W. The filter window size is f by f.
#   The bias vector is b, where each filter has its own (single) bias.
#
# You also have access to the hyperparameters dictionary, which contains the stride and the padding. 
#
# Hint:
#  To select a 2x2 slice at the upper left corner of a matrix "a_prev" (shape (5,5,3)), you would do:
#   a_slice_prev = a_prev[0:2,0:2,:]
#  Notice how this gives a 3D slice that has height 2, width 2, and depth 3. Depth is the number of channels.
#  This will be useful when you will define a_slice_prev below, using the start/end indexes you will define.
#
#  To define a_slice you will need to first define its corners vert_start, vert_end, horiz_start and horiz_end. 
#  This figure may be helpful for you to find out how each of the corners can be defined using h, w, f and s in the code below.
#
# The formulas relating the output shape of the convolution to the input shape are:
#  𝑛𝐻=⌊𝑛𝐻𝑝𝑟𝑒𝑣−𝑓+2×𝑝𝑎𝑑𝑠𝑡𝑟𝑖𝑑𝑒⌋+1
#  𝑛𝑊=⌊𝑛𝑊𝑝𝑟𝑒𝑣−𝑓+2×𝑝𝑎𝑑𝑠𝑡𝑟𝑖𝑑𝑒⌋+1
#  𝑛𝐶=number of filters used in the convolution
#
# Additional Hints (if you're stuck):
#   Use array slicing (e.g.varname[0:1,:,3:5]) for the following variables:
#    a_prev_pad ,W, b
#        Copy the starter code of the function and run it outside of the defined function, in separate cells.
#        Check that the subset of each array is the size and dimension that you're expecting.
#   To decide how to get the vert_start, vert_end, horiz_start, horiz_end, remember that these are indices of the previous layer.
#        Draw an example of a previous padded layer (8 x 8, for instance), and the current (output layer) (2 x 2, for instance).
#        The output layer's indices are denoted by h and w.
#   Make sure that a_slice_prev has a height, width and depth.
#   Remember that a_prev_pad is a subset of A_prev_pad.
#        Think about which one should be used within the for loops.
#
# For this exercise, don't worry about vectorization! Just implement everything with for-loops.
#
def conv_forward(A_prev, W, b, hparameters):
    """
    Implements the forward propagation for a convolution function
    
    Arguments:
    A_prev -- output activations of the previous layer, 
        numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
    b -- Biases, numpy array of shape (1, 1, 1, n_C)
    hparameters -- python dictionary containing "stride" and "pad"
        
    Returns:
    Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward() function
    """
    
    # Retrieve dimensions from A_prev's shape (≈1 line)  
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    # Retrieve dimensions from W's shape (≈1 line)
    (f, f, n_C_prev, n_C) = W.shape
    
    # Retrieve information from "hparameters" (≈2 lines)
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    
    # Compute the dimensions of the CONV output volume using the formula given above. 
    # Hint: use int() to apply the 'floor' operation. (≈2 lines)
    # Using // works as floor division
    n_H = int((n_H_prev - f + 2 * pad)//stride) + 1
    n_W = int((n_W_prev - f + 2 * pad)//stride) + 1
    
    # Initialize the output volume Z with zeros. (≈1 line)
    Z = np.zeros((m, n_H, n_W, n_C))
    
    # Create A_prev_pad by padding A_prev
    A_prev_pad = zero_pad(A_prev, pad)
    
    for i in range(m):                      # loop over the batch of training examples
        a_prev_pad = A_prev_pad[i]          # Select ith training example's padded activation
        for h in range(n_H):                # loop over vertical axis of the output volume
            # Find the vertical start and end of the current "slice" (≈2 lines)
            vert_start = stride * h
            vert_end = vert_start + f
            
            for w in range(n_W):            # loop over horizontal axis of the output volume
                # Find the horizontal start and end of the current "slice"
                horiz_start = stride * w
                horiz_end = horiz_start + f
                
                for c in range(n_C):        # loop over channels (= #filters) of the output volume
                                        
                    # Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell).
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end,:]
                    
                    # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron.
                    weights = W[:, :, :, c]
                    biases = b[:, :, :, c]
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, weights, biases)
    
    # Making sure your output shape is correct
    assert(Z.shape == (m, n_H, n_W, n_C))

    # Save information in "cache" for the backprop
    cache = (A_prev, W, b, hparameters)
    
    return Z, cache

# Finally, a CONV layer should also contain an activation, in which case you would add the following line of code:
#
# Convolve the window to get back one output neuron
#   Z[i, h, w, c] = ...
# Apply activation
#   A[i, h, w, c] = activation(Z[i, h, w, c])
#
# You don't need to do it here, however.
#




# Pooling layer
#   The pooling (POOL) layer reduces the height and width of the input. 
#   It helps reduce computation, as well as helps make feature detectors more invariant to its position in the input. 
#   The two types of pooling layers are:
#       Max-pooling layer: slides an (𝑓,𝑓) window over the input and stores the max value of the window in the output.
#       Average-pooling layer: slides an (𝑓,𝑓) window over the input and stores the average value of the window in the output.
#
# These pooling layers have no parameters for backpropagation to train. However, they have hyperparameters such as the window size 𝑓. 
# This specifies the height and width of the 𝑓×𝑓 window you would compute a max or average over.
#
# Reminder: As there's no padding, the formulas binding the output shape of the pooling to the input shape is:
#   𝑛𝐻=⌊𝑛𝐻𝑝𝑟𝑒𝑣−𝑓𝑠𝑡𝑟𝑖𝑑𝑒⌋+1
#   𝑛𝑊=⌊𝑛𝑊𝑝𝑟𝑒𝑣−𝑓𝑠𝑡𝑟𝑖𝑑𝑒⌋+1
#   𝑛𝐶=𝑛𝐶𝑝𝑟𝑒𝑣
#
def pool_forward(A_prev, hparameters, mode = "max"):
    """
    Implements the forward pass of the pooling layer
    
    Arguments:
    A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    hparameters -- python dictionary containing "f" and "stride"
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
    
    Returns:
    A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters 
    """
    
    # Retrieve dimensions from the input shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    # Retrieve hyperparameters from "hparameters"
    f = hparameters["f"]
    stride = hparameters["stride"]
    
    # Define the dimensions of the output
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev
    
    # Initialize output matrix A
    A = np.zeros((m, n_H, n_W, n_C))              
    
    for i in range(m):                         # loop over the training examples
        for h in range(n_H):                     # loop on the vertical axis of the output volume
            # Find the vertical start and end of the current "slice" (≈2 lines)
            vert_start = stride * h
            vert_end = vert_start + f
            
            for w in range(n_W):                 # loop on the horizontal axis of the output volume
                # Find the vertical start and end of the current "slice" (≈2 lines)
                horiz_start = stride * w
                horiz_end = horiz_start + f
                
                for c in range (n_C):            # loop over the channels of the output volume
                    
                    # Use the corners to define the current slice on the ith training example of A_prev, channel c. (≈1 line)
                    a_slice_prev = A_prev[i][vert_start:vert_end, horiz_start:horiz_end, c]
                    
                    # Compute the pooling operation on the slice. 
                    # Use an if statement to differentiate the modes. 
                    # Use np.max and np.mean.
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_slice_prev)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_slice_prev)
       
    # Store the input and hparameters in "cache" for pool_backward()
    cache = (A_prev, hparameters)
    
    # Making sure your output shape is correct
    assert(A.shape == (m, n_H, n_W, n_C))
    
    return A, cache





# What you should remember:
#
#    A convolution extracts features from an input image by taking the dot product between the input data and a 2D array of weights (the filter).
#    The 2D output of the convolution is called the feature map.
#    A convolution layer is where the filter slides over the image and computes the dot product.
#        This transforms the input volume into an output volume of different size.
#    Zero padding helps keep more information at the image borders, and is helpful for building deeper 
#       networks, because you can build a CONV layer without shrinking the height and width of the volumes.
#    Pooling layers gradually reduce the height and width of the input by sliding a 2D window over each 
#       specified region, then summarizing the features in that region.
#



#################################
# Extra
#################################
# 
# Backpropagation in Convolutional Neural Networks
#   In modern deep learning frameworks, you only have to implement the forward pass, and the framework takes 
#   care of the backward pass, so most deep learning engineers don't need to bother with the details of the 
#   backward pass. The backward pass for convolutional networks is complicated. 
#   If you wish, you can work through this optional portion of the notebook to get a sense of what backprop in a convolutional network looks like.
#
# When in an earlier course you implemented a simple (fully connected) neural network, you used backpropagation 
#   to compute the derivatives with respect to the cost to update the parameters. 
#   Similarly, in convolutional neural networks you can calculate the derivatives with respect to the cost 
#   in order to update the parameters. The backprop equations are not trivial and were not derived in lecture, 
#   but are briefly presented below.
#

# Let's start by implementing the backward pass for a CONV layer. 
#
#
# Computing dA
#
# This is the formula for computing 𝑑𝐴 with respect to the cost for a certain filter 𝑊𝑐 and a given training example:
#   𝑑𝐴 += ∑(ℎ=0, 𝑛𝐻) ( ∑(𝑤=0, 𝑛𝑊) (𝑊𝑐 × 𝑑𝑍ℎ𝑤) )
#
# Where 𝑊𝑐 is a filter and 𝑑𝑍ℎ𝑤 is a scalar corresponding to the gradient of the cost with respect to the output 
#   of the conv layer Z at the hth row and wth column (corresponding to the dot product taken at the ith 
#   stride left and jth stride down). 
# Note that at each time, you multiply the the same filter 𝑊𝑐 by a different dZ when updating dA. 
# We do so mainly because when computing the forward propagation, each filter is dotted and summed by a different a_slice. 
# Therefore when computing the backprop for dA, you are just adding the gradients of all the a_slices.
#
# In code, inside the appropriate for-loops, this formula translates into:
#   da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c]
#
#
#
# Computing dW
#
# This is the formula for computing 𝑑𝑊𝑐 (𝑑𝑊𝑐 is the derivative of one filter) with respect to the loss:
#   𝑑𝑊𝑐 += (∑(ℎ=0, 𝑛𝐻) ( ∑(𝑤=0, 𝑛𝑊) 𝑎𝑠𝑙𝑖𝑐𝑒×𝑑𝑍ℎ𝑤) )
#
# Where 𝑎𝑠𝑙𝑖𝑐𝑒 corresponds to the slice which was used to generate the activation 𝑍𝑖𝑗. 
# Hence, this ends up giving us the gradient for 𝑊 with respect to that slice. 
# Since it is the same 𝑊, we will just add up all such gradients to get 𝑑𝑊.
#
# In code, inside the appropriate for-loops, this formula translates into:
#   dW[:,:,:,c] += a_slice * dZ[i, h, w, c]
#
#
#
# Computing db
#
# This is the formula for computing 𝑑𝑏 with respect to the cost for a certain filter 𝑊𝑐:
#   𝑑𝑏 = ∑ℎ ∑𝑤 𝑑𝑍ℎ𝑤
#
# As you have previously seen in basic neural networks, db is computed by summing 𝑑𝑍. 
# In this case, you are just summing over all the gradients of the conv output (Z) with respect to the cost.
#
# In code, inside the appropriate for-loops, this formula translates into:
# db[:,:,:,c] += dZ[i, h, w, c]
#
def conv_backward(dZ, cache):
    """
    Implement the backward propagation for a convolution function
    
    Arguments:
    dZ -- gradient of the cost with respect to the output of the conv layer (Z), numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward(), output of conv_forward()
    
    Returns:
    dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
               numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    dW -- gradient of the cost with respect to the weights of the conv layer (W)
          numpy array of shape (f, f, n_C_prev, n_C)
    db -- gradient of the cost with respect to the biases of the conv layer (b)
          numpy array of shape (1, 1, 1, n_C)
    """    
    
        
    # Retrieve information from "cache"
    (A_prev, W, b, hparameters) = cache
    # Retrieve dimensions from A_prev's shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    # Retrieve dimensions from W's shape
    (f, f, n_C_prev, n_C) = W.shape
    
    # Retrieve information from "hparameters"
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    
    # Retrieve dimensions from dZ's shape
    (m, n_H, n_W, n_C) = dZ.shape
    
    # Initialize dA_prev, dW, db with the correct shapes
    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))                      
    dW = np.zeros((f, f, n_C_prev, n_C))
    db = np.zeros((1, 1, 1, n_C))
    
    # Pad A_prev and dA_prev
    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)
    
    for i in range(m):                       # loop over the training examples
        
        # select ith training example from A_prev_pad and dA_prev_pad
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        
        for h in range(n_H):                   # loop over vertical axis of the output volume
            for w in range(n_W):               # loop over horizontal axis of the output volume
                for c in range(n_C):           # loop over the channels of the output volume
                    
                    # Find the corners of the current "slice"
                    vert_start = stride * h
                    vert_end = vert_start + f
                    horiz_start = stride * w
                    horiz_end = horiz_start + f

                    # Use the corners to define the slice from a_prev_pad
                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end,:]

                    # Update gradients for the window and the filter's parameters using the code formulas given above
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c]
                    dW[:,:,:,c] += a_slice * dZ[i, h, w, c]
                    db[:,:,:,c] += dZ[i, h, w, c]
                    
        # Set the ith training example's dA_prev to the unpadded da_prev_pad (Hint: use X[pad:-pad, pad:-pad, :])
        dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]
    
    # Making sure your output shape is correct
    assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))
    
    return dA_prev, dW, db





# Pooling layer - Backward Pass
#   Next, let's implement the backward pass for the pooling layer, starting with the MAX-POOL layer.
#   Even though a pooling layer has no parameters for backprop to update, you still need to backpropagate 
#       the gradient through the pooling layer in order to compute gradients for layers that came before the 
#       pooling layer. 
#

# Implement create_mask_from_window(). This function will be helpful for pooling backward. Hints:
#
#    np.max() may be helpful. It computes the maximum of an array.
#    If you have a matrix X and a scalar x: A = (X == x) will return a matrix A of the same size as X such that:
#
#    A[i,j] = True if X[i,j] = x
#    A[i,j] = False if X[i,j] != x
#
#    Here, you don't need to consider cases where there are several maxima in a matrix.
#
def create_mask_from_window(x):
    """
    Creates a mask from an input matrix x, to identify the max entry of x.
    
    Arguments:
    x -- Array of shape (f, f)
    
    Returns:
    mask -- Array of the same shape as window, contains a True at the position corresponding to the max entry of x.
    """    
    mask = (x == np.max(x))
    
    return mask


# Implement the function below to equally distribute a value dz through a matrix of dimension shape. 
#
def distribute_value(dz, shape):
    """
    Distributes the input value in the matrix of dimension shape
    
    Arguments:
    dz -- input scalar
    shape -- the shape (n_H, n_W) of the output matrix for which we want to distribute the value of dz
    
    Returns:
    a -- Array of size (n_H, n_W) for which we distributed the value of dz
    """    
    # Retrieve dimensions from shape (≈1 line)
    (n_H, n_W) = shape
    
    # Compute the value to distribute on the matrix (≈1 line)
    average = average = dz / (n_H * n_W)
    
    # Create a matrix where every entry is the "average" value (≈1 line)
    a = np.ones(shape) * average

    return a



# Putting things together.
#
# Implement the pool_backward function in both modes ("max" and "average"). 
# You will once again use 4 for-loops (iterating over training examples, height, width, and channels). 
# You should use an if/elif statement to see if the mode is equal to 'max' or 'average'. 
# If it is equal to 'average' you should use the distribute_value() function you implemented above to create a matrix of the same shape as a_slice. 
# Otherwise, the mode is equal to 'max', and you will create a mask with create_mask_from_window() and multiply it by the corresponding value of dA.
#
def pool_backward(dA, cache, mode = "max"):
    """
    Implements the backward pass of the pooling layer
    
    Arguments:
    dA -- gradient of cost with respect to the output of the pooling layer, same shape as A
    cache -- cache output from the forward pass of the pooling layer, contains the layer's input and hparameters 
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
    
    Returns:
    dA_prev -- gradient of cost with respect to the input of the pooling layer, same shape as A_prev
    """
        
    # Retrieve information from cache (≈1 line)
    (A_prev, hparameters) = cache
    
    # Retrieve hyperparameters from "hparameters" (≈2 lines)
    stride = hparameters["stride"]
    f = hparameters["f"]
    
    # Retrieve dimensions from A_prev's shape and dA's shape (≈2 lines)
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    m, n_H, n_W, n_C = dA.shape
    
    # Initialize dA_prev with zeros (≈1 line)
    dA_prev = np.zeros(A_prev.shape)
    
    for i in range(m):                       # loop over the training examples
        # select training example from A_prev (≈1 line)
        a_prev = A_prev[i]
        for h in range(n_H):                   # loop on the vertical axis
            for w in range(n_W):               # loop on the horizontal axis
                for c in range(n_C):           # loop over the channels (depth)
                    # Find the corners of the current "slice" (≈4 lines)
                    vert_start = h
                    vert_end = vert_start + f
                    horiz_start = w
                    horiz_end = horiz_start + f
                    
                    # Compute the backward propagation in both modes.
                    if mode == "max":
                        # Use the corners and "c" to define the current slice from a_prev (≈1 line)
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                        # Create the mask from a_prev_slice (≈1 line)
                        mask = create_mask_from_window(a_prev_slice)
                        # Set dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA) (≈1 line)
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += np.multiply(mask, dA[i, h, w, c])
                        
                    elif mode == "average":
                        # Get the value a from dA (≈1 line)
                        da = dA[i, h, w, c]
                        # Define the shape of the filter as fxf (≈1 line)
                        shape = (f, f)
                        # Distribute it to get the correct slice of dA_prev. i.e. Add the distributed value of da. (≈1 line)
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += distribute_value(da, shape)
                            
    # Making sure your output shape is correct
    assert(dA_prev.shape == A_prev.shape)
    
    return dA_prev

# Congratulations! You've completed the assignment and its optional portion. 
# You now understand how convolutional neural networks work, and have implemented all the building blocks of a neural network. 
# In the next assignment you will implement a ConvNet using TensorFlow. 
# Nicely done! See you there.
#