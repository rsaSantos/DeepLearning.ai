# Package imports
import numpy as np                  # fundamental package for scientific computing with Python.
import matplotlib.pyplot as plt
from testCases_v2 import *
import sklearn.datasets
import sklearn.linear_model
from planar_utils import load_extra_datasets
# Extra: Let's test things with some different data sets
#
# Datasets

def xtra_data():

    noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()
    datasets = {"noisy_circles": noisy_circles,
                "noisy_moons": noisy_moons,
                "blobs": blobs,
                "gaussian_quantiles": gaussian_quantiles}

    # Choose dataset
    dataset = "noisy_moons"

    # Load datasets into matrixes X (input) and Y (results)
    X, Y = datasets[dataset]
    X, Y = X.T, Y.reshape(1, Y.shape[0])

    # make blobs binary
    if dataset == "noisy_circles":
        Y = Y%2
    
    return (X, Y)
