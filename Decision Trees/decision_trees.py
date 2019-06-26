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
    best_thresh = -1

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
    best_feat = -1
    best_thresh = -1

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

        labels, labels_count = np.unique(data[:,-1], return_counts=True)
        num_x_label = -np.inf
        for i in range(len(labels_count)):
            if (labels_count[i] > num_x_label):
                num_x_label = labels_count[i]
                self.majority = labels[i]


    def __str__(self):
        if self.is_leaf():
            string = "leaf: ["
            vals, counts = np.unique(self.data[:,-1], return_counts=True)
            for i in range(len(vals)):
                string += f"{{{vals[i]}: {counts[i]}}},"
            string = string[:-1] + "]"

        else:
            string = f"[X{self.feature} <= {self.threshold:.3f}]"

        return string

    def add_child(self, node):
        self.children.append(node)
        node.parent = self


    def is_leaf(self):
        return len(self.children) == 0

    def is_pure(self):
        return len(np.unique(self.data[:, -1])) <= 1

def build_tree(data, impurity, chi_value=1):
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

    if root.is_pure():
        return root

    left_data = data[data[:,feat] < thresh]
    right_data = data[data[:,feat] > thresh]

    if chi_value != 1:
        chi_square = prune(data, left_data, right_data)
        if chi_square <= chi_table[chi_value]:
            return root

    root.add_child(build_tree(left_data, impurity, chi_value))
    root.add_child(build_tree(right_data, impurity, chi_value))

    return root

def predict(root: DecisionNode, instance):
    """
    Predict a given instance using the decision tree

    Input:
    - root: the root of the decision tree.
    - instance: a row vector from the dataset. Note that the last element
                of this vector is the label of the instance.

    Output: the prediction of the instance.
    """
    node = root

    while not node.is_leaf():
        node = node.children[0] if instance[node.feature] < node.threshold else node.children[1]

    return node.majority


def calc_accuracy(root: DecisionNode, dataset):
    """
    calculate the accuracy starting from the root of the decision tree using
    the given dataset.

    Input:
    - root: the root of the decision tree.
    - dataset: the dataset on which the accuracy is evaluated

    Output: the accuracy of the decision tree on the given dataset (%).
    """
    accuracy = 0.0

    for instance in dataset:
        if predict(root, instance) == instance[-1]:
            accuracy += 1

    accuracy /= len(dataset)

    return accuracy * 100


def prune(data, left_data, right_data):
    """
    Determines if a nodes children have predictive power using
    the chi square test.

    Input:
    - node: a DecisionNode.
    - left: the dataset of the node's left child.
    - right: the dataset of the node's right child.

    Output: the chi square value.
    """
    chi_square = 0
    # counts[0] is # of instances Y = 0, counts[1] is for Y=1
    values, counts = np.unique(data[:, -1], return_counts=True)
    total = len(data)  # total # of instances

    probability_y0 = counts[0] / total  # P(Y = 0)
    probability_y1 = counts[1] / total  # P(Y = 1)

    # D = # of instances where feature <= thresh and feature > thresh
    D = np.array([len(left_data), len(right_data)])

    p = np.array([0, 0])
    n = np.array([0, 0])
    # p = # of instances where feature <= thresh and y = 0 in left and right nodes
    p[0] = len(left_data[left_data[:, -1] == 0])
    p[1] = len(right_data[right_data[:, -1] == 0])
    # n = # of instances xj > thresh and y = 1 in left and right nodes
    n[0] = len(left_data[left_data[:, -1] == 1])
    n[1] = len(right_data[right_data[:, -1] == 1])

    for f in values:
        E0 = D[int(f)] * probability_y0
        E1 = D[int(f)] * probability_y1
        chi_square += (np.square((p[int(f)] - E0)) / E0) + (np.square(n[int(f)] - E1) / E1)

    return chi_square

def num_internal(node: DecisionNode):
    """
    Calaculates the number of internal nodes in a tree. i.e. the number of non-leaf nodes.
    :param node: The root of the tree
    :return:  the number of internal nodes.
    """
    if node.is_leaf():
        return 0
    else:
        return num_internal(node.children[0]) + 1 + num_internal(node.children[1])

def list_leaves(root: DecisionNode):
    """
    Creates a list of all the leaves in a tree.

    Input:
    - node: a DecisionNode

    Output: A list of all the leaves in the tree.
    """
    ################# helper function ##################
    def list_leaves(root: DecisionNode, leaves):
        if root.is_leaf():
            leaves.append(root)
        else:
            list_leaves(root.children[0], leaves)
            list_leaves(root.children[1], leaves)
        return leaves
    ###################################################

    return list_leaves(root, [])


def post_prune_predict(root, instance, stop):
    """
    Predicts the class of the instance by traversing the tree
    until it reaches either a leaf or the stop node.

    :param root: The root of the decision tree.
    :param instance: The instance to classify.
    :param stop: The node to stop at.
    :return: The prediction of the instance's class.
    """
    node = root

    while not node.is_leaf():
        if node == stop:
            return node.majority
        
        node = node.children[0] if instance[node.feature] < node.threshold else node.children[1]

    return node.majority
#
# def post_prune_predict(node: DecisionNode, instance, stop: DecisionNode):
#     """
#     Predicts the class of an instance in the tree pruned at the given stop node.
#
#     Input:
#     - node: the root of the tree.
#     - instance: the instance we want to predict.
#     - stop: the node that was pruned due to the chi square test
#
#     Output: the prediction.
#     """
#     while (len(node.children) > 0):
#         if (stop == node):
#             return node.majority
#
#         if (not node.is_leaf()):
#             if (instance[node.feature] <= node.threshold):
#                 node = node.children[0]
#             else:
#                 node = node.children[1]
#     return node.majority


def post_prune_accuracy(root: DecisionNode, dataset, stop: DecisionNode):
    """
    Calculates the accuracy of the tree on the dataset without going past the stop node.

    Input:
    - root: the root of the Decision Tree
    - data_set: the set to compute the accuracy on
    - stop: The node to stop at.

    Output: The accuracy of the tree without the children of the stop node.
    """
    accuracy = 0

    for instance in dataset:
        prediction = post_prune_predict(root, instance, stop)

        if prediction == instance[-1]:
            accuracy += 1

    accuracy /= len(dataset)

    return accuracy * 100


def post_prune(root: DecisionNode, train_set, test_set):
    """
    For each leaf in the tree, calculate the test accuracy of the tree
    assuming no split occurred on the parent of that leaf and find the best
    such parent (in the sense that not splitting on that parent results in
    the best testing accuracy among possible parents). Make that parent into
    a leaf and repeat this process until you are left with just the root.

    Input:
    - root: the root of the Decision tree
    - train_set: the training set
    - test_set: the testing set

    Output:
    (1) - The list of training accuracies
    (2) - the list of testing accuracies
    (3) - the list of number of nodes
    """
    training_accuracy = []
    testing_accuracy = []
    num_nodes = []

    num_nodes.append(num_internal(root))
    training_accuracy.append(calc_accuracy(root, train_set))
    testing_accuracy.append((calc_accuracy(root, test_set)))

    while len(root.children) != 0:
        leaves = list_leaves(root)
        max_test_acc = np.NINF
        best_parent = None

        for leaf in leaves:
            stop = leaf.parent
            test_acc = post_prune_accuracy(root, test_set, stop)

            if test_acc > max_test_acc:
                max_test_acc = test_acc
                best_parent = stop

        best_parent.children = []
        train_acc = calc_accuracy(root, train_set)

        training_accuracy.append(train_acc)
        testing_accuracy.append(max_test_acc)
        num_nodes.append(num_internal(root))

    return training_accuracy, testing_accuracy, num_nodes




def print_tree(node):
    """
    prints the tree according to the example in the notebook

	Input:
	- node: a node in the decision tree

	This function has no return value
	"""
    ################ helper function ##############
    def print_tree(node, depth):
        print('   ' * depth, end='')
        print(node)

        if node.is_leaf():
            return

        depth += 1

        print_tree(node.children[0], depth)
        print_tree(node.children[1], depth)
    #################################################

    print_tree(node, 0)
