import numpy as np
import itertools
np.random.seed(42)


def preprocess(X, y):
    """
    Perform mean normalization on the features and the labels
 
    Input:
    - X: Inputs  (n features over m instances).
    - y: True labels.

    Returns a two vales:
    - X: The mean normalized inputs.
    - y: The mean normalized labels.
    """
    X = (X - X.mean(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    y = (y - y.mean()) / (y.max() - y.min())

    return X, y


def compute_cost(X, y, theta):
    """
    Computes the average squared difference between an observation’s actual and
    predicted values for linear regression.

    Input:
    - X: Inputs  (n features over m instances).
    - y: True labels (1 value over m instances).
    - theta: The parameters (weights) of the model being learned.

    Returns a single value:
    - J: the cost associated with the current set of parameters (single number).
    """
    m = len(X) # m is the number of instances
    hypothesis = np.dot(X, theta)
    square_error = (hypothesis - y) ** 2

    J = np.sum(square_error) / (2 * m)

    return J

