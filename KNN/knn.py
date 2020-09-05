import numpy as np
import pandas as pd
from scipy import stats

class KNNClassifier:
    
    def __init__(self, n_neighbors=5, p=2):
        self.n_neighbors = n_neighbors
        self.p = p
        
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        
    def predict(self, X_test):
        dists = (np.abs(self.X_train - X_test[:,np.newaxis,:]) ** self.p).sum(axis=2)
        sorted_indices = dists.argsort()
        closest_k = self.y_train[sorted_indices][:,:self.n_neighbors]
        preds = stats.mode(closest_k, axis=1).mode
        return preds
