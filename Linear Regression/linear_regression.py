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



def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of the model using gradient descent. Gradient descent
    is an optimization algorithm used to minimize some (loss) function by
    iteratively moving in the direction of steepest descent as defined by the
    negative of the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Inputs  (n features over m instances).
    - y: True labels (1 value over m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns two values:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """

    J_history = []  # Use a python list to save cost in every iteration
    learning_rate = alpha / len(X)

    for i in range(num_iters):
        hypothesis = np.dot(X, theta)
        error = hypothesis - y

        theta = theta - learning_rate * np.dot(error, X)

        J_history.append(compute_cost(X, y, theta))

    return theta, J_history