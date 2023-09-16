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

footer = f"""<style>
.footer {{
position: fixed;
left: 0;
bottom: 0;
width: 100%;
text-align: right;
padding-right: 10px;
}}
</style>
<div class="footer">
<p>{os.getenv('DOCKER_TAG', '')}</p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)
pages[page].run()
logger.info(f'Running page {page}')
