import streamlit as st
import os
from ml_pages import k_means

st.set_page_config('ML Deep Dive')
logger = st.logger.get_logger(__name__)
logger.info(f'App running on version {os.environ.get("DOCKER_TAG")}')

pages = {
    'K Means': k_means
}

with st.sidebar:
    page = st.selectbox("Algorithms", pages.keys())

pages[page].run()
logger.info(f'Running page {page}')
