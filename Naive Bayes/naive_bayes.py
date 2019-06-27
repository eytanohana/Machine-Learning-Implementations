import numpy as np
np.random.seed(42)

####################################################################################################
#                                            Part A
####################################################################################################

class NaiveNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, std) for a class conditional normal distribution.
        The mean and std are computed from a given data set.
        
        Input
        - dataset: The dataset from which to compute the mean and mu (Numpy Array).
        - class_value : The class to calculate the mean and mu for.
        """
        self.dataset = dataset
        self.class_dataset = dataset[dataset[:,-1] == class_value]
        self.class_value = class_value
        self.mean = np.mean(self.class_dataset[:, :-1], axis=0)
        self.std = np.std(self.class_dataset[:,:-1], axis=0)
    
    def get_prior(self):
        """
        Returns the prior probability of the class according to the dataset distribution.
        P(A)
        """
        return len(self.class_dataset) / len(self.dataset)
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood probability of the instance under the class according to the dataset distribution.
        P(x | A)
        """
        return normal_pdf(x[0], self.mean[0], self.std[0]) * normal_pdf(x[1], self.mean[1], self.std[1])
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        P(A|x)
        * Ignoring p(x)
        """
        return self.get_instance_likelihood(x) * self.get_prior()
    
class MultiNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, cov matrix) for a class conditional multi normal distribution.
        The mean and cov matrix (You can use np.cov for this!) will be computed from a given data set.
        
        Input
        - dataset: The dataset from which to compute the mean and mu (Numpy Array).
        - class_value : The class to calculate the mean and mu for.
        """
        self.dataset = dataset
        self.class_dataset = dataset[dataset[:,-1] == class_value]
        self.class_value = class_value
        self.mean = np.mean(self.class_dataset[:,:-1], axis=0)
        self.cov = np.cov(self.class_dataset[:,:-1], rowvar=False)
        
    def get_prior(self):
        """
        Returns the prior probability of the class according to the dataset distribution.
        P(A)
        """
        return len(self.class_dataset) / len(self.dataset)
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod probability of the instance under the class according to the dataset distribution.
        P(x|A)
        """
        return multi_normal_pdf(x, self.mean, self.cov)
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        P(A|x) = P(x|A) * P(A)
        * Ignoring p(x)
        """
        return self.get_instance_likelihood(x) * self.get_prior()
    
    

def normal_pdf(x, mean, std):
    """
    Calculate normal density function for a given x, mean and standard deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and standard deviation for the given x.
    """
    fraction = 1 / np.sqrt(2 * np.pi * std**2)
    exponent = -(x - mean)**2 / (2 * std**2)

    return fraction * np.exp(exponent)


def multi_normal_pdf(x, mean, cov):
    """
    Calculate multi variant normal density function for a given x, mean and covariance matrix.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - cov: The covariance matrix of the distribution
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """
    fraction = 1 / (2 * np.pi * np.sqrt(np.linalg.det(cov)))
    # @ denotes matrix multiplication in numpy
    exponent = -.5 * (x[:-1] - mean).transpose() @ np.linalg.inv(cov) @ (x[:-1] - mean)

    return fraction  * np.exp(exponent)


####################################################################################################
#                                            Part B
####################################################################################################
EPSILLON = 1e-6 # == 0.000001 It could happen that a certain value will only occur in the test set.
                # In case such a thing occur the probability for that value will EPSILLON.

class DiscreteNBClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which computes and encapsulates the relevant probabilities for a discrete naive bayes
        distribution for a specific class. The probabilities are computed with la place smoothing.
        
        Input
        - dataset: The dataset from which to compute the probabilities (Numpy Array).
        - class_value : Compute the relevant parameters only for instances from the given class.
        """
        self.dataset = dataset
        self.class_dataset = dataset[dataset[:,-1] == class_value]
        self.class_value = class_value
    
    def get_prior(self):
        """
        Returns the prior probability of the class according to the dataset distribution.
        P(A)
        """
        return len(self.class_dataset) / len(self.dataset)
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood probability of the instance under the class according to the dataset distribution.
        P(x|A)
        """
        return 1
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior probability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        return 1

    
####################################################################################################
#                                            General
####################################################################################################            
class MAPClassifier():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a posteriori classifier.
        This class will hold 2 class distribution, one for class 0 and one for class 1, and will predict the
         class of an instance by the class that outputs the highest posterior probability for the given instance.
    
        Input
            - ccd0 : An object containing the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object containing the relevant parameters and methods for the distribution of class 1.
        """
        self.clf0 = ccd0
        self.clf1 = ccd1
    
    def predict(self, x):
        """
        Predicts the instance's class using the 2 distribution objects given in the object constructor.
        
        Input
            - An instance to predict.
            
        Output
            - 0 if the posterior probability of class 0 is higher than 1 otherwise.
        """
        posterior_0 = self.clf0.get_instance_posterior(x)
        posterior_1 = self.clf1.get_instance_posterior(x)

        if posterior_0 > posterior_1:
            return self.clf0.class_value
        else:
            return self.clf1.class_value


def compute_accuracy(testset, map_classifier):
    """
    Computes the accuracy of a given testset using a map classifier object.
    
    Input
        - testset: The test for which to compute the accuracy (Numpy array).
        - map_classifier : A MAPClassifier object capable of predicting the class for each instance in the testset.
        
    Ouput
        - Accuracy = #Correctly Classified / #testset size
    """
    accuracy = 0
    for instance in testset:
        if map_classifier.predict(instance) == instance[-1]:
            accuracy += 1

    accuracy /= len(testset)

    return accuracy * 100
    
            
            
            
            
            
            
            
            
            
    