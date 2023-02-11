import streamlit as st
from skimage import io


def run():
    st.header('K-Means', anchor='k_means')
    st.markdown('''
    K-Means is an algorithm for grouping similar data into K predefined groups. In this app we'll use
    K-Means to compress an image using color quantization, the process of compressing an image by representing
    it using less colors.
    ''')
    image = st.file_uploader('Choose an image', accept_multiple_files=False)
    if not image:
        st.stop()
    st.image(image)
    image = io.imread(image)
    st.write(f'''
    The shape of the image is: {image.shape}
    
    The first dimension, {image.shape[0]}, represents the height of the image, while the second, {image.shape[1]},
    represents the width of the image, both being in pixels. The third dimension, {image.shape[2]}, represents the
    different color channels of the image. Most traditionally, rgb, representing the red/green/blue intensities of
    each channel for each pixel. The intensities ranging from 0 - 255.
    ''')
