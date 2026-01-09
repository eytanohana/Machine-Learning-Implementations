import numpy as np


class KNNClassifier:
    def __init__(self, n_neighbors=5, p=2):
        self.n_neighbors = n_neighbors
        self.p = p

    def fit(self, X_train, y_train):
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)

        if X_train.ndim != 2:
            raise ValueError(f'X_train must be 2D, got {X_train.ndim}D')
        if y_train.ndim != 1:
            raise ValueError(f'y_train must be 1D, got {y_train.ndim}D')
        if len(X_train) != len(y_train):
            raise ValueError(f'X_train and y_train must have same length, got {len(X_train)} and {len(y_train)}')
        if self.n_neighbors > len(X_train):
            raise ValueError(
                f'n_neighbors ({self.n_neighbors}) cannot be greater than number of samples ({len(X_train)})'
            )

        self.X_train = X_train
        self.y_train = y_train

    def _mode(self, arr):
        """Find the mode of an array, handling string arrays."""
        unique_vals, counts = np.unique(arr, return_counts=True)
        return unique_vals[np.argmax(counts)]

    def predict(self, X_test):
        X_test = np.asarray(X_test)
        if X_test.ndim != 2:
            raise ValueError(f'X_test must be 2D, got {X_test.ndim}D')
        if X_test.shape[1] != self.X_train.shape[1]:
            raise ValueError(f'X_test must have {self.X_train.shape[1]} features, got {X_test.shape[1]}')

        dists = (np.abs(self.X_train - X_test[:, np.newaxis, :]) ** self.p).sum(axis=2) ** (1 / self.p)
        sorted_indices = dists.argsort()
        closest_k = self.y_train[sorted_indices[:, : self.n_neighbors]]
        preds = np.array([self._mode(row) for row in closest_k])
        return preds


class KNNRegressor:
    def __init__(self, n_neighbors=5, p=2):
        self.n_neighbors = n_neighbors
        self.p = p

    def fit(self, X_train, y_train):
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)

        if X_train.ndim != 2:
            raise ValueError(f'X_train must be 2D, got {X_train.ndim}D')
        if y_train.ndim != 1:
            raise ValueError(f'y_train must be 1D, got {y_train.ndim}D')
        if len(X_train) != len(y_train):
            raise ValueError(f'X_train and y_train must have same length, got {len(X_train)} and {len(y_train)}')
        if self.n_neighbors > len(X_train):
            raise ValueError(
                f'n_neighbors ({self.n_neighbors}) cannot be greater than number of samples ({len(X_train)})'
            )

        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        X_test = np.asarray(X_test)
        if X_test.ndim != 2:
            raise ValueError(f'X_test must be 2D, got {X_test.ndim}D')
        if X_test.shape[1] != self.X_train.shape[1]:
            raise ValueError(f'X_test must have {self.X_train.shape[1]} features, got {X_test.shape[1]}')

        dists = (np.abs(self.X_train - X_test[:, np.newaxis, :]) ** self.p).sum(axis=2) ** (1 / self.p)
        sorted_indices = dists.argsort()
        closest_k = self.y_train[sorted_indices[:, : self.n_neighbors]]
        preds = closest_k.mean(axis=1)
        return preds
