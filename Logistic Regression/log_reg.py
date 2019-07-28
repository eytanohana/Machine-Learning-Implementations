import numpy as np
import itertools

np.random.seed(42)


def min_max_scale(X):
    """
    Perform min max scaling on the features

    Input:
    - X: Inputs  (n features over m instances).

    Returns a two vales:
    - X: The mean normalized inputs.
    """
    return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

def standaradize(X):
    """
    Perform mean normalization on the features of a dataset.
    Center the features around mean = 0 with a standard deviation = 1
    """
    return (X - X.mean(axis=0)) / X.std(axis=0)
                                   

def sigmoid(x):
    """
    Computes the value of the sigmoid function S(x) = 1 / (1 + e^-x)
    """
    return 1 / (1 + np.exp(-x))



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
    m = len(X)  # m is the number of instances
    hypothesis = sigmoid(X @ theta)
    J = -np.sum(y * np.log(hypothesis) + (1 - y) * np.log(1 - hypothesis)) / m
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
        error = sigmoid(X @ theta) - y
        theta = theta - learning_rate * (error @ X)
        J_history.append(compute_cost(X, y, theta))

    return theta, J_history

def predict(x, theta):
    """
    Predicts the class of a single instance or set of instances.
    :param x: Can be a single instance vector or a matrix of instances
    :param theta: The parameters learned during training.
    :return: The class.
    """
    return sigmoid(x @ theta) > .5



def compute_accuracy(X, y, theta):
    """
    Computes the accuracy on a given testset.

    Input
        - X: The dataset for which to compute the accuracy (Numpy array).
        - y: The labels of each instance in the dataset.
        - theta: The parameters learned in training.
    Ouput
        - Accuracy = #Correctly Classified / #testset size
    """
    prediction = predict(X, theta)
    correct = (prediction == y).sum()
    return round(correct * 100.0 / len(X), ndigits=2)



################ Need to modify from linear regression ###########################

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

    J_history = []  
    learning_rate = alpha / len(X)
    J_history.append(compute_cost(X, y, theta))

    for i in range(num_iters):
        hypothesis = X @ theta
        error = hypothesis - y

        theta = theta - learning_rate * (error @ X)
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


def find_best_triplet(df, triplets, alpha, num_iter):
    """
    Iterate over all possible triplets and find the triplet that best
    minimizes the cost function. For better performance, you should use the
    efficient implementation of gradient descent. You should first preprocess
    the data and obtain an array containing the columns corresponding to the
    triplet. Don't forget the bias trick.

    Input:
    - df: A dataframe that contains the data
    - triplets: a list of three strings representing three features in X.
    - alpha: The value of the best alpha previously found.
    - num_iters: The number of updates performed.

    Returns:
    - The best triplet as a python list holding the best triplet as strings.
    """
    best_triplet = None
    min_cost = np.inf
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    y = np.array(df['price'])
    theta = np.ones(4)
    for t in triplets:
        X = np.array(df[list(t)])
#         X, y = preprocess(X, y)

        X = np.column_stack((np.ones(X.shape[0]), X))

        _, J_history = efficient_gradient_descent(X, y, theta, alpha, num_iter)

        if J_history[-1] < min_cost:
            best_triplet = t
            min_cost = J_history[-1]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return best_triplet

