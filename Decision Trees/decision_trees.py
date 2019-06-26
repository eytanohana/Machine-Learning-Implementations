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


def best_feature_threshold(data, impurity):
    """
    Calculates the best feature and its corresponding threshold to split the data.

    Input:
    - data: any dataset where the last column holds the labels.
    - impurity: the impurity measure to use (either calc_gini or calc_entropy).
    """
    max_gain = -np.inf
    best_feat = None
    best_thresh = None

    for feat in range(data.shape[1] - 1):
        thresh = best_threshold(data, feat, impurity)
        gain = info_gain(data, feat, thresh, impurity)

        if gain > max_gain:
            max_gain = gain
            best_feat = feat
            best_thresh = thresh

    return best_feat, best_thresh


class DecisionNode:

    # This class will hold everything you require to construct a decision tree.
    # The structure of this class is up to you. However, you need to support basic
    # functionality as described in the notebook. It is highly recommended that you
    # first read and understand the entire exercise before diving into this class.

    def __init__(self, data, feature, threshold):
        self.data = data
        self.feature = feature  # column index of feature that best splits the data
        self.threshold = threshold  # the best threshold of the feature
        self.children = []

    def __str__(self):
        if self.is_leaf():
            string = f"leaf: [{{{self.data[0,-1]}: {len(self.data)}}}]"
        else:
            string = f"X{self.feature} < {self.threshold}"

        return string

    def add_child(self, node):
        self.children.append(node)

    def is_leaf(self):
        return len(np.unique(self.data[:, -1])) <= 1

def build_tree(data, impurity):
    """
    Build a tree using the given impurity measure and training dataset.
    You are required to fully grow the tree until all leaves are pure.

    Input:
    - data: the training dataset.
    - impurity: the chosen impurity measure. Notice that you can send a function
                as an argument in python.

    Output: the root node of the tree.
    """
    feat, thresh = best_feature_threshold(data, impurity)
    root = DecisionNode(data, feat, thresh)

    if root.is_leaf():
        return root

    left_data = data[data[:,feat] < thresh]
    right_data = data[data[:,feat] > thresh]

    root.children.append(build_tree(left_data, impurity))
    root.children.append(build_tree(right_data, impurity))

    return root

