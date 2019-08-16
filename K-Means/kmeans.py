import numpy as np
from skimage import io

def get_random_centroids(X, k):
    centroids = X[np.random.randint(len(X), size=k)]
    return centroids


def lp_distance(X, centroids, p=2):
    distances = np.abs(X.astype(float) - centroids[:, np.newaxis, :]) ** p
    distances = distances.sum(axis=2)
    distances = distances ** (1 / p)

    return distances


def kmeans(X, k, p, max_iter=100):
    """
    Inputs:
    - X: a single image of shape (num_features, 3).
    - k: number of centroids (i.e. colors).
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.
    Outputs:
    - The calculated centroids
    - The final assignment of all RGB points to the closest centroids
    """
    classes = []
    centroids = get_random_centroids(X, k)

    for i in range(max_iter):
        print('iteration:', i + 1)
        old_classes = classes.copy()

        distances = lp_distance(X, centroids, p)
        classes = np.argmin(distances, axis=0)

        if np.array_equal(old_classes, classes):
            print('done')
            return centroids, classes

        for j in range(k):
            centroids[j] = X[classes == j].mean(axis=0)

    return centroids, classes

def display_image(centroids, classes, img_shape):
    classes = classes.reshape(img_shape)
    compressed_image = np.zeros((classes.shape[0],classes.shape[1],3),dtype=np.uint8 )
    for i in range(classes.shape[0]):
        for j in range(classes.shape[1]):
                compressed_image[i,j,:] = centroids[classes[i,j],:]
    io.imshow(compressed_image)
    io.show()