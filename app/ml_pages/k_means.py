import streamlit as st


def run():
    st.header('K-Means', anchor='k_means')
    st.markdown('''
    K-Means is an algorithm for grouping similar data into K predefined groups. In this app we'll use
    K-Means to compress an image using color quantization, the process of compressing an image by representing
    it using less colors.
    ''')
    image = st.file_uploader('Choose an image', accept_multiple_files=False)
    if image:
        st.image(image)