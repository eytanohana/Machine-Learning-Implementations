from numpy import count_nonzero, logical_and, logical_or, concatenate, mean, array_split, poly1d, polyfit, array
from numpy.random import permutation
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt


SVM_DEFAULT_DEGREE = 3
SVM_DEFAULT_GAMMA = 'auto'
SVM_DEFAULT_C = 1.0
ALPHA = 1.5


def prepare_data(data, labels, max_count=None, train_ratio=0.8):
    """
    :param data: a numpy array with the features dataset
    :param labels:  a numpy array with the labels
    :param max_count: max amount of samples to work on. can be used for testing
    :param train_ratio: ratio of samples used for train
    :return: train_data: a numpy array with the features dataset - for train
             train_labels: a numpy array with the labels - for train
             test_data: a numpy array with the features dataset - for test
             test_labels: a numpy array with the features dataset - for test
    """
    if max_count:
        data = data[:max_count]
        labels = labels[:max_count]


    dataset = concatenate((data, labels.reshape((-1,1))), axis=1)
    dataset = permutation(dataset)

    slice_size = int(train_ratio * dataset.shape[0])

    train_data = dataset[:slice_size, :-1]
    train_labels = dataset[:slice_size, -1]

    test_data = dataset[slice_size:, :-1]
    test_labels = dataset[slice_size:, -1]

    return train_data, train_labels, test_data, test_labels


def get_stats(prediction, labels):
    """
    :param prediction: a numpy array with the prediction of the model
    :param labels: a numpy array with the target values (labels)
    :return: tpr: true positive rate = #tp / #p
             fpr: false positive rate = #fp / #n
             accuracy: accuracy of the model given the predictions
    """

    tpr = 0.0
    fpr = 0.0
    accuracy = (prediction == labels).sum()
    pos = count_nonzero(labels)

    for pred, lab in zip(prediction, labels):
        if pred == 1:
            if lab == 1:
                tpr += 1
            else:
                fpr += 1

    tpr /= pos
    fpr /= (len(prediction) - pos)
    accuracy /= len(prediction)

    return tpr, fpr, accuracy



def get_k_fold_stats(folds_array, labels_array, clf):
    """
    :param folds_array: a k-folds arrays based on a dataset with M features and N samples
    :param labels_array: a k-folds labels array based on the same dataset
    :param clf: the configured SVC learner
    :return: mean(tpr), mean(fpr), mean(accuracy) - means across all folds
    """
    tpr = []
    fpr = []
    accuracy = []

    for fold in range(len(folds_array)):
        x = concatenate((folds_array[:fold] + folds_array[fold+1:]), axis=0)
        y = concatenate((labels_array[:fold] + labels_array[fold+1:]))

        clf.fit(x, y)
        predictions = clf.predict(folds_array[fold])
        temp_tpr, temp_fpr, temp_accu = get_stats(predictions, labels_array[fold])

        tpr.append(temp_tpr)
        fpr.append(temp_fpr)
        accuracy.append(temp_accu)

    return mean(tpr), mean(fpr), mean(accuracy)


def compare_svms(data_array,
                 labels_array,
                 folds_count,
                 kernels_list=('poly', 'poly', 'poly', 'rbf', 'rbf', 'rbf',),
                 kernel_params=({'degree': 2}, {'degree': 3}, {'degree': 4}, {'gamma': 0.005}, {'gamma': 0.05}, {'gamma': 0.5},)):
    """
    :param data_array: a numpy array with the features dataset
    :param labels_array: a numpy array with the labels
    :param folds_count: number of cross-validation folds
    :param kernels_list: a list of strings defining the SVM kernels
    :param kernel_params: a dictionary with kernel parameters - degree, gamma, c
    :return: svm_df: a dataframe containing the results as described below
    """
    svm_df = pd.DataFrame()
    svm_df['kernel'] = kernels_list
    svm_df['kernel_params'] = kernel_params
    svm_df['tpr'] = None
    svm_df['fpr'] = None
    svm_df['accuracy'] = None



    return svm_df


def get_most_accurate_kernel():
    """
    :return: integer representing the row number of the most accurate kernel
    """
    best_kernel = 0
    return best_kernel


def get_kernel_with_highest_score():
    """
    :return: integer representing the row number of the kernel with the highest score
    """
    best_kernel = 0
    return best_kernel


def plot_roc_curve_with_score(df, alpha_slope=1.5):
    """
    :param df: a dataframe containing the results of compare_svms
    :param alpha_slope: alpha parameter for plotting the linear score line
    :return:
    """
    x = df.fpr.tolist()
    y = df.tpr.tolist()

    ###########################################################################
    # TODO: Implement the function                                            #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################


def evaluate_c_param(data_array, labels_array, folds_count):
    """
    :param data_array: a numpy array with the features dataset
    :param labels_array: a numpy array with the labels
    :param folds_count: number of cross-validation folds
    :return: res: a dataframe containing the results for the different c values. columns similar to `compare_svms`
    """

    res = pd.DataFrame()
    ###########################################################################
    # TODO: Implement the function                                            #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return res


def get_test_set_performance(train_data, train_labels, test_data, test_labels):
    """
    :param train_data: a numpy array with the features dataset - train
    :param train_labels: a numpy array with the labels - train

    :param test_data: a numpy array with the features dataset - test
    :param test_labels: a numpy array with the labels - test
    :return: kernel_type: the chosen kernel type (either 'poly' or 'rbf')
             kernel_params: a dictionary with the chosen kernel's parameters - c value, gamma or degree
             clf: the SVM leaner that was built from the parameters
             tpr: tpr on the test dataset
             fpr: fpr on the test dataset
             accuracy: accuracy of the model on the test dataset
    """

    kernel_type = ''
    kernel_params = None
    clf = SVC(class_weight='balanced')  # TODO: set the right kernel
    tpr = 0.0
    fpr = 0.0
    accuracy = 0.0

    ###########################################################################
    # TODO: Implement the function                                            #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return kernel_type, kernel_params, clf, tpr, fpr, accuracy
