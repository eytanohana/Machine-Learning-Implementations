
import io
import time

import numpy as np
import streamlit as st
from PIL import Image

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
    
    image = Image.open(image)
    image.thumbnail(size=(500, 500))
    st.image(image)
    
    # Calculate original image size (as PNG)
    original_img_buffer = io.BytesIO()
    image.save(original_img_buffer, format='PNG')
    original_size_bytes = len(original_img_buffer.getvalue())

    image = np.asarray(image)
    original_shape = image.shape
    st.text(f'image compressed to {original_shape}, size = {original_size_bytes / 1024:.2f} KB')

    with (st.expander('Explanation')):
        num_channels = image.shape[2]
        channel_description = "RGB (red/green/blue)" if num_channels == 3 \
            else "RGBA (red/green/blue/alpha)" if num_channels == 4 \
            else f"{num_channels} channels"

        st.write(f'''
        The shape of the image is: {image.shape}

        The first dimension, {image.shape[0]}, represents the height of the image, while the second, {image.shape[1]},
        represents the width of the image, both being in pixels. The third dimension, {image.shape[2]}, represents the
        different color channels of the image. This image has {num_channels} channel{'s' if num_channels != 1 else ''},
        representing {channel_description} intensities for each pixel. The intensities range from 0 - 255.
        ''')

        image = image.reshape(image.shape[0] * image.shape[1], image.shape[2])
        st.write(f'''
        For the K-means algorithm, we need to reshape the data into two dimensions. The number of rows corresponding to
        the number of pixels in the image and the number of columns representing the different color 
        channels: {image.shape} - {len(image):,} pixels
        ''')

        st.markdown(r'''
        ## The Algorithm
        1. Start by choosing k random points (pixels), called `centroids`.
        2. Assign every point in the dataset to the nearest centroid.
            * All points belonging to the same centroid belong to the same "group".
        3. Calculate the mean point for each group and assign the means as the new centroids.
        4. Repeat steps 2 and 3 until we don't see a change in the means or we reach a predetermined maximum number of iterations.

        ### The distance metric
        To calculate the distance between two points, we use the Minkowski distance metric.
        The Minkowski distance of order $p$ between two points:
        $\vec{x}=(x_1, ..., x_n)$ and $\vec{y}=(y_1, ..., y_n)$ is:
        $$
        D(\vec{x},\vec{y}) = (\sum_{i=1}^n \mid x_i - y_i \mid ^p)^{\frac{1}{p}}
        $$
        The Minkowski distance is a generalization of the Euclidean distance $p=2$:

        $\sqrt{\sum_{i=1}^n (x_i - y_i)^2}$

        and Manhattan distance $p=1$:

        $\sum_{i=1}^n |x_i - y_i|$
        ''')
    a, b, c = st.columns(3)
    k = a.number_input('Number of centroids', 2, 100, help='the number of colors to use (1-100)')
    p = b.number_input('Distance metric', 1, 100, help='distance metric to use between each pixel color and the centroid color (1-100)')
    max_iter = c.number_input('Max Iterations', 10, 100, help='max iterations the algorithm can run, it can complete earlier (10-100)')

    progress_text = f'Running K-means with K = {k}'
    progress = st.progress(0., text=progress_text)
    image_space = st.empty()
    download_placeholder = st.empty()
    final_image = None

    # Run the simulation
    start = time.perf_counter()
    for i, (centroids, classes) in enumerate(kmeans(image, k, p, max_iter=max_iter), 1):
        progress.progress(i / max_iter, text=f'{i}: ' + progress_text + f', time: {time.perf_counter() - start:.2f}s')
        final_image = display_image(centroids, classes, original_shape)
        image_space.image(final_image)
    
    # Mark simulation as complete
    progress.progress(1., text=f'Finished K-means with K = {k} colors in {i} iterations,'
                               f' total time: {time.perf_counter() - start:.2f}s')
    
    # Add download button only after simulation completes
    if final_image is not None:
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(final_image)
        # Save to bytes buffer
        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        compressed_size_bytes = len(img_buffer.getvalue())
        
        # Calculate compression ratio
        compression_ratio = (1 - compressed_size_bytes / original_size_bytes) * 100
        
        # Display size comparison
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Original Size", f"{original_size_bytes / 1024:.2f} KB")
        with col2:
            st.metric("Compressed Size", f"{compressed_size_bytes / 1024:.2f} KB")
        with col3:
            st.metric("Size Reduction", f"{compression_ratio:.1f}%")
        
        # Place download button in placeholder only after simulation completes
        with download_placeholder.container():
            @st.fragment()
            def download_image():
                st.download_button(
                    label='Download Compressed Image',
                    data=img_buffer.getvalue(),
                    file_name=f'kmeans_k{k}_compressed.png',
                    mime='image/png'
                )
            download_image()

