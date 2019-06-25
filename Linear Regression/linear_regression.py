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


def pseudo_inverse(X):
    """
    Matrices that aren't full rank (i.e. not all the columns are linearly independent)
    are not invertible. So we can create a good approximation of the inverse of the matrix
    using its pseudo-inverse defined as pinv(X) = (X_transpose * X)^-1 * X_transpose.
    Note if the matrix is already full rank then the pseudo-inverse of the matrix is just its inverse.

    This method of calculating the pseudo-inverse of a matrix can still fail if X_transpose * X
    is still singular (i.e. not invertible). In which case you must use np.linalg.pinv(X)

    :param X: The matrix
    :return: The pseudo-inverse matrix of X
    """
    # X_transpose * X gives a square matrix
    square = np.matmul(X.transpose(), X)
    square_inverse = np.linalg.inv(square)
    pinv_X = np.matmul(square_inverse, X.transpose())
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pinv_X

def optimal_theta(X, y):
    """
    Calculate the optimal values of the parameters using the pseudo-inverse
    approach as you saw in class.

    Input:
    - X: Inputs  (n features over m instances).
    - y: True labels (1 value over m instances).

    Returns:
    - theta: The optimal parameters of your model.

    ########## DO NOT USE numpy.pinv ##############
    """
    pinv_X = pseudo_inverse(X)
    pinv_theta = np.dot(pinv_X, y)

    return pinv_theta


def efficient_gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of your model, but stop the learning process once
    the improvement of the loss value and the loss value itself is smaller than 1e-8.
    This function is very similar to the gradient descent function you already implemented.

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
    J_history.append(compute_cost(X, y, theta))

    for i in range(num_iters):
        hypothesis = np.dot(X, theta)
        error = hypothesis - y

        theta = theta - learning_rate * np.dot(error, X)
        J_history.append(compute_cost(X, y, theta))

        if J_history[-1] < 1e-8 and abs(J_history[-1] - J_history[-2]) < 1e-8:
            break

    return theta, J_history


def find_best_alpha(X, y, iterations):
    """
    Iterate over provided values of alpha and maintain a python dictionary
    with alpha as the key and the final loss as the value.

    Input:
    - X: a dataframe that contains all relevant features.

    Returns:
    - alpha_dict: A python dictionary that hold the loss value after training
    for every value of alpha.
    """

    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    alpha_dict = {}

    for alpha in alphas:
        loops = 0
        theta = np.random.random(size=2)
        last_loss = np.inf

        while loops < iterations:
            theta, loss = efficient_gradient_descent(X, y, theta, alpha, 100)

            if loops > 0 and loss[-1] > last_loss:
                break
            loops += 100
            last_loss = loss[-1]
        alpha_dict[alpha] = last_loss

    return alpha_dict


def generate_triplets(X):
    """
    generate all possible sets of three features out of all relevant features
    available from the given dataset X. You might want to use the itertools
    python library.

    Input:
    - X: a dataframe that contains all relevant features.

    Returns:
    - A python list containing all feature triplets.
    """
    triplets = itertools.combinations(X, 3)

    return list(triplets)