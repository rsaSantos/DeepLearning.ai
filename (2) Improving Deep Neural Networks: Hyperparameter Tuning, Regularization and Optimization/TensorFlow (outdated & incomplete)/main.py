# Until now, you've always used numpy to build neural networks. 
# Now we will step you through a deep learning framework that will allow you to build 
#   neural networks more easily. 
# Machine learning frameworks like TensorFlow, PaddlePaddle, Torch, Caffe, Keras, and 
#   many others can speed up your machine learning development significantly. 
# All of these frameworks also have a lot of documentation, which you should feel free to read. 
# In this assignment, you will learn to do the following in TensorFlow:
#
#   Initialize variables
#   Start your own session
#   Train algorithms
#   Implement a Neural Network
#
# Programing frameworks can not only shorten your coding time, but sometimes also perform 
#   optimizations that speed up your code. 


#imports
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict

np.random.seed(1)


# Compute loss of a single training example -> loss = L(ŷ ,y) = (ŷ − y)^2
y_hat = tf.constant(36, name='y_hat')            # Define y_hat constant. Set to 36.
y = tf.constant(39, name='y')                    # Define y. Set to 39

loss = tf.compat.v1.Variable((y - y_hat)**2, name='loss')  # Create a variable for the loss

init = tf.compat.v1.global_variables_initializer()         # When init is run later (session.run(init)),
                                                 # the loss variable will be initialized and ready to be computed
with tf.compat.v1.Session() as session:                    # Create a session and print the output
    session.run(init)                            # Initializes the variables
    print(session.run(loss))                     # Prints the loss
