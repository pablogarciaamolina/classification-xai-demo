"""
Global XAI page - Generate and display global explanations using Gradient Ascent
"""

import streamlit as st
import requests

from app.frontend.config import GLOBAL_XAI_ENDPOINT
from components.utils import get_big_cats_dataset
from components.saliency_viz import render_global_visualization


st.title("🌍 Global XAI - Class Visualizations")
st.markdown("Generate visualizations that show what the model has learned for each class using Gradient Ascent.")

# Load dataset to get class names
dataset = get_big_cats_dataset()
class_names = dataset.classes

# Initialize session state
if "global_xai_result" not in st.session_state:
    st.session_state.global_xai_result = None

# =================
# Class Selection
# =================
st.subheader("Available Classes")

# Display classes in a grid
cols = st.columns(5)
for i, class_name in enumerate(class_names):
    with cols[i % 5]:
        st.markdown(f"**{i}.** {class_name}")

st.markdown("---")

# Class selector
selected_class_idx = st.selectbox(
    "Select a class to visualize:",
    options=range(len(class_names)),
    format_func=lambda x: f"{x}. {class_names[x]}"
)

selected_class_name = class_names[selected_class_idx]

st.markdown(f"### Selected Class: **{selected_class_name}**")

# ======================
# Generate Explanation
# ======================
if st.button("🎨 Generate Global Explanation", type="primary"):
    with st.spinner(f"Generating visualization for {selected_class_name}... This may take a minute."):
        try:
            payload = {"target_class": selected_class_idx}
            response = requests.post(GLOBAL_XAI_ENDPOINT, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                st.session_state.global_xai_result = result
                st.success("✅ Visualization generated successfully!")
                st.rerun()
            else:
                st.error(f"Error {response.status_code}: {response.text}")
                
        except Exception as e:
            st.error(f"Request failed: {e}")

# Display result if available
if st.session_state.global_xai_result is not None:
    result = st.session_state.global_xai_result
    
    st.markdown("---")
    st.subheader("Generated Visualization")
    
    # Center the visualization
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        render_global_visualization(
            result["visualization"],
            result["class_name"]
        )
    
    st.markdown("""
    ### 
    This image was generated using **Gradient Ascent**
    
    The resulting image shows patterns, textures, and features that the model has learned to 
    associate with this particular class.
    """)
