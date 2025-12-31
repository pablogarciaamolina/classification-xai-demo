"""
Dataset selection page
"""

import streamlit as st
import requests
from app.frontend.config import DATASET_SELECTION_ENDPOINT, DATASET_METADATA

st.title("🎯 Dataset Selection")
st.markdown("Select a dataset to begin classification and XAI analysis")

st.session_state.setdefault("selected_dataset", None)

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"### {DATASET_METADATA['stl10']['icon']} {DATASET_METADATA['stl10']['display_name']}")
    st.markdown(f"**Classes:** {DATASET_METADATA['stl10']['num_classes']}")
    st.markdown(f"**Description:** {DATASET_METADATA['stl10']['description']}")
    
    if st.button("Select STL10", key="select_stl10", use_container_width=True, type="primary"):
        with st.spinner("Loading STL10 dataset..."):
            try:
                response = requests.post(
                    DATASET_SELECTION_ENDPOINT,
                    json={"dataset_name": "stl10"},
                    timeout=30
                )
                
                if response.status_code == 200:
                    st.session_state.selected_dataset = "stl10"
                    st.session_state.dataset_info = response.json()
                    st.success("✅ STL10 dataset loaded successfully!")
                    st.balloons()
                    st.rerun()
                else:
                    st.error(f"❌ Error loading dataset: {response.json().get('error', 'Unknown error')}")
            except Exception as e:
                st.error(f"❌ Failed to connect to backend: {str(e)}")

with col2:
    st.markdown(f"### {DATASET_METADATA['big_cats']['icon']} {DATASET_METADATA['big_cats']['display_name']}")
    st.markdown(f"**Classes:** {DATASET_METADATA['big_cats']['num_classes']}")
    st.markdown(f"**Description:** {DATASET_METADATA['big_cats']['description']}")
    
    if st.button("Select Big Cats", key="select_big_cats", use_container_width=True, type="primary"):
        with st.spinner("Loading Big Cats dataset..."):
            try:
                response = requests.post(
                    DATASET_SELECTION_ENDPOINT,
                    json={"dataset_name": "big_cats"},
                    timeout=30
                )
                
                if response.status_code == 200:
                    st.session_state.selected_dataset = "big_cats"
                    st.session_state.dataset_info = response.json()
                    st.success("✅ Big Cats dataset loaded successfully!")
                    st.balloons()
                    st.rerun()
                else:
                    st.error(f"❌ Error loading dataset: {response.json().get('error', 'Unknown error')}")
            except Exception as e:
                st.error(f"❌ Failed to connect to backend: {str(e)}")

st.divider()

if st.session_state.selected_dataset:
    dataset_name = DATASET_METADATA[st.session_state.selected_dataset]["display_name"]
    st.info(f"ℹ️ Currently selected: **{dataset_name}**")
    st.markdown("Navigate to **Home - Prediction** to start making predictions!")
else:
    st.warning("⚠️ No dataset selected. Please select a dataset to continue.")
