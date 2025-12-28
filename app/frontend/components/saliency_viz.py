"""
Saliency map visualization component
"""

import streamlit as st
import base64
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt


def render_saliency_visualization(
    saliency_map_base64: str,
    method_name: str,
    class_name: str,
    faithfulness: float,
    robustness: float,
    deletion_curve: dict = None
) -> None:
    """
    Render a saliency map with its evaluation metrics and deletion curve
    
    Args:
        saliency_map_base64: Base64-encoded saliency map overlay
        method_name: Name of the XAI method (e.g., "GradCAM")
        class_name: Target class name
        faithfulness: ROAD faithfulness score
        robustness: Average Sensitivity robustness score
        deletion_curve: Dictionary containing 'percentiles' and 'scores' lists
    """
    st.markdown(f"### {method_name} - {class_name}")
    
    img_data = base64.b64decode(saliency_map_base64)
    img = Image.open(BytesIO(img_data))
    st.image(img, width="stretch")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label="📊 Faithfulness (AUC)",
            value=f"{faithfulness:.3f}",
            help="Higher is better. Measures how well the saliency map identifies important pixels."
        )
    with col2:
        st.metric(
            label="🛡️ Robustness (Avg. Sensitivity)",
            value=f"{robustness:.3f}",
            help="Lower is better. Measures stability of explanations under small input perturbations."
        )
        
    if deletion_curve:
        st.markdown("**Deletion Curve (ROAD)**")
        try:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(
                deletion_curve['percentiles'], 
                deletion_curve['scores'], 
                marker='o', 
                linestyle='-', 
                color='#ff4b4b',
                linewidth=2,
                markersize=4
            )
            ax.set_xlabel('Pixels Removed (%)')
            ax.set_ylabel('Model Confidence')
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, linestyle='--', alpha=0.5)
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            st.pyplot(fig, width="stretch")
            plt.close(fig)
        except Exception as e:
            st.warning(f"Could not plot deletion curve: {str(e)}")


def render_global_visualization(
    visualization_base64: str,
    class_name: str
) -> None:
    """
    Render a global explanation visualization
    
    Args:
        visualization_base64: Base64-encoded gradient ascent visualization
        class_name: Target class name
    """
    st.markdown(f"### Global Explanation for: **{class_name}**")
    st.markdown("This image was generated to maximize the activation of the target class.")
    
    img_data = base64.b64decode(visualization_base64)
    img = Image.open(BytesIO(img_data))
    st.image(img, width="stretch")
