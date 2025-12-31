"""
Main entry point for the multi-page Streamlit app
"""

import streamlit as st

st.set_page_config(
    page_title="Classification XAI Demo",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

dataset_selection_page = st.Page("pages/dataset_selection.py", title="Dataset Selection", icon="🎯")
home_page = st.Page("pages/home.py", title="Home - Prediction", icon="🏠")
local_xai_page = st.Page("pages/local_xai.py", title="Local XAI", icon="🔍")
global_xai_page = st.Page("pages/global_xai.py", title="Global XAI", icon="🌍")

pg = st.navigation([dataset_selection_page, home_page, local_xai_page, global_xai_page])
pg.run()

