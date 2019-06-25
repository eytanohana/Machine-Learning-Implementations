import numpy as np
np.random.seed(42)

chi_table = {0.01  : 6.635,
             0.005 : 7.879,
             0.001 : 10.828,
             0.0005 : 12.116,
             0.0001 : 15.140,
             0.00001: 19.511}

def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns the gini impurity of the dataset.
    """
    size = len(data)
    classes = data[:,-1]
    _, class_counts = np.unique(classes, return_counts=True)
    gini = 1 - np.sum((class_counts / size) ** 2)

    return gini


def calc_entropy(data):
    """
    Calculates the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns the entropy of the dataset.
    """
    size = len(data)
    classes = data[:,-1]
    _, class_counts = np.unique(classes, return_counts=True)

    entropy = (class_counts / size) * np.log2(class_counts / size)
    entropy = -np.sum(entropy)

    return entropy


def best_threshold(data, feature, impurity):
    """
    Calculates the best_threshold in a given feature to split the dataset by.

    Input:
    - data: any dataset where the last column holds the labels.
    - feature: the column index of the feature to split by.
    - impurity: the impurity measure to use (either calc_gini or calc_entropy).

    Returns the best threshold from the given feature to split the data by.
    """
    values = np.unique(data[:,feature])
    max_gain = -np.inf
    best_thresh = None

    for i in range(len(values) - 1):
        thresh = (values[i] + values[i+1]) / 2

        gain = info_gain(data, feature, thresh, impurity)

        if gain > max_gain:
            max_gain = gain
            best_thresh = thresh

    return best_thresh



def info_gain(data, feature, threshold, impurity):
    """
    Calculates the information gain of a split by the given feature.

    Input:
    - data: any dataset where the last column holds the labels.
    - feature: the column index of the feature to split by.
    - impurity: the impurity measure to use (either calc_gini or calc_entropy).

    Returns the information gain of a split by the given feature.
    """
    size = len(data)
    left = data[data[:,feature] < threshold]
    right = data[data[:,feature] > threshold]

    parent_entropy = impurity(data)

    left_entropy =  impurity(left)
    right_entropy = impurity(right)

    gain = parent_entropy - (len(left) / size) * left_entropy - (len(right) / size) * right_entropy

    return gain