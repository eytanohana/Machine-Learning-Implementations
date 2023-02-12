import streamlit as st
from skimage import io
from .src.kmeans import kmeans, display_image


def run():
    st.markdown('''
    # K-Means
    K-Means is an algorithm for grouping similar data into K predefined groups. In this app we'll use
    K-Means to compress an image using color quantization, the process of compressing an image by representing
    it using less colors.
    ''')
    image = st.file_uploader('Choose an image', accept_multiple_files=False)
    if not image:
        st.stop()
    st.image(image)
    image = io.imread(image)
    original_shape = image.shape
    with st.expander('Explanation'):
        st.write(f'''
        The shape of the image is: {image.shape}
        
        The first dimension, {image.shape[0]}, represents the height of the image, while the second, {image.shape[1]},
        represents the width of the image, both being in pixels. The third dimension, {image.shape[2]}, represents the
        different color channels of the image. Most traditionally, rgb, representing the red/green/blue intensities of
        each channel for each pixel. The intensities ranging from 0 - 255.
        ''')

        image = image.reshape(image.shape[0] * image.shape[1], image.shape[2])
        st.write(f'''
        For the K-means algorithm, we need to reshape the data into two dimensions. The number of rows corresponding to
        the number of pixels in the image and the number of columns representing the different color 
        channels: {image.shape} - {len(image):,} pixels
        ''')

        st.markdown(r'''
        ## The Algorithm
        1. We start by choosing k random points, called the `centroids`.
        1. We then assign every point in the dataset to the nearest centroid.
            * All points belonging to the same centroid belong to the same "group".
        1. We then calculate the mean point for each group and assign the means as the new centroids. 
        1. We then repeat steps 2 and 3 until we don't see a change in the means or 
        we reach a predetermined maximum number of iterations.
        
        ### The distance metric
        To calculate the distance between two points, we use the Minkowski distance metric.
        The Minkowski distance of order $p$ between two points: 
        $\vec{x}=(x_1, ..., x_n)$ and $\vec{y}=(y_1, ..., y_n)$ is:
        $$
        D(\vec{x},\vec{y}) = (\sum_{i=1}^n \mid x_i - y_i \mid ^p)^{\frac{1}{p}}
        $$
        The Minkowski distance is a generalization of the Euclidean ($p=2$) and Manhattan ($p=1$) distances.
        ''')
    a, b = st.columns(2)
    k = a.number_input('Number of centroids', 2, 20)
    p = b.number_input('Distance metric', 2, 100)
    with st.spinner():
        centroids, classes = kmeans(image, k, p)
        compressed_img = display_image(centroids, classes, original_shape)
        st.image(compressed_img)

