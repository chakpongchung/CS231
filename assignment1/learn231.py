# Run some setup code for this notebook.

import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt



# Load the raw CIFAR-10 data.
cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# As a sanity check, we print out the size of the training and test data.
print 'Training data shape: ', X_train.shape
print 'Training labels shape: ', y_train.shape
print 'Test data shape: ', X_test.shape
print 'Test labels shape: ', y_test.shape


print y_train[:2]

from cs231n.classifiers import KNearestNeighbor

# # Subsample the data for more efficient code execution in this exercise
# num_training = 5000
# mask = range(num_training)
# X_train = X_train[mask]
# y_train = y_train[mask]

# num_test = 500
# mask = range(num_test)
# X_test = X_test[mask]
# y_test = y_test[mask]


# # Reshape the image data into rows
# X_train = np.reshape(X_train, (X_train.shape[0], -1))
# X_test = np.reshape(X_test, (X_test.shape[0], -1))
# # print X_train.shape, X_test.shape
# # Create a kNN classifier instance. 
# # Remember that training a kNN classifier is a noop: 
# # the Classifier simply remembers the data and does no further processing 
# classifier = KNearestNeighbor()
# classifier.train(X_train, y_train)


# Test your implementation:
# dists = classifier.compute_distances_two_loops(X_test)
# print dists.shape

# dists_two = classifier.compute_distances_no_loops(X_test)
# print dists_two.shape

# Let's compare how fast the implementations are
def time_function(f, *args):
  """
  Call a function f with args and return the time (in seconds) that it took to execute.
  """
  import time
  tic = time.time()
  f(*args)
  toc = time.time()
  return toc - tic

# no_loop_time = time_function(classifier.compute_distances_no_loops, X_test)
# print 'No loop version took %f seconds' % no_loop_time


# Split the data into train, val, and test sets. In addition we will
# create a small development set as a subset of the training data;
# we can use this for development so our code runs faster.
num_training = 49000
num_validation = 1000
num_test = 1000
num_dev = 500

# Our validation set will be num_validation points from the original
# training set.
mask = range(num_training, num_training + num_validation)
X_val = X_train[mask]
y_val = y_train[mask]

# Our training set will be the first num_train points from the original
# training set.
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]

# We will also make a development set, which is a small subset of
# the training set.
mask = np.random.choice(num_training, num_dev, replace=False)
X_dev = X_train[mask]
y_dev = y_train[mask]

# We use the first num_test points of the original test set as our
# test set.
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]

print 'Train data shape: ', X_train.shape
print 'Train labels shape: ', y_train.shape
print 'Validation data shape: ', X_val.shape
print 'Validation labels shape: ', y_val.shape
print 'Test data shape: ', X_test.shape
print 'Test labels shape: ', y_test.shape