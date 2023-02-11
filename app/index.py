import streamlit as st
from ml_pages import k_means

st.set_page_config('ML Deep Dive')

pages = {
    'K Means': k_means
}

with st.sidebar:
    page = st.selectbox("Algorithms", pages.keys())

pages[page].run()
