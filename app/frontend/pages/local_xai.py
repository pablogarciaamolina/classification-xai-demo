"""
Local XAI page - Generate and display saliency maps with evaluations
"""

import streamlit as st
import requests

from app.frontend.config import IMAGES_PER_PAGE, MAX_GALLERY_COLUMNS, PREDICTION_BACKEND_ENDPOINT, LOCAL_XAI_ENDPOINT
from components.utils import get_big_cats_dataset, get_big_cats_dataloader, load_new_batch, tensor_to_display
from components.gallery import render_gallery
from components.saliency_viz import render_saliency_visualization


st.title("🔍 Local XAI - Saliency Maps")
st.markdown("Select an image to generate local explanations using GradCAM and Integrated Gradients.")

# Load data
dataset = get_big_cats_dataset()
dataloader = get_big_cats_dataloader(dataset, batch_size=IMAGES_PER_PAGE)

# Set state
for key in ["batch", "labels", "selected_index", "local_xai_results"]:
    st.session_state.setdefault(key, None)

# Set batch
if st.session_state.batch is None:
    images, labels = load_new_batch(dataloader)
    st.session_state.batch = images
    st.session_state.labels = labels

images_batch = st.session_state.batch
labels_batch = st.session_state.labels

# Set displayed data
displayable_images = [tensor_to_display(img) for img in images_batch]
textual_labels = [dataset.classes[idx] for idx in labels_batch]

# =================
# Selection gallery
# =================
st.subheader("Image Batch")
st.markdown("Click on an image to select it.")

def select_callback(idx):
    st.session_state.selected_index = idx
    st.session_state.local_xai_results = None  # Reset results when new image selected

render_gallery(
    displayable_images,
    labels=textual_labels,
    on_select=select_callback,
    max_num_cols=MAX_GALLERY_COLUMNS
)

if st.button("🔄 Load New Batch"):
    images, labels = load_new_batch(dataloader)
    st.session_state.batch = images
    st.session_state.labels = labels
    st.session_state.selected_index = None
    st.session_state.local_xai_results = None
    st.rerun()

# ======================
# Generate Explanations
# ======================
if st.session_state.selected_index is not None:
    idx = st.session_state.selected_index
    true_label_idx = labels_batch[idx].item()
    true_label = dataset.classes[true_label_idx]
    
    st.markdown("---")
    st.subheader("Selected Image")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(displayable_images[idx], width=350, clamp=True)
        st.write(f"**True Label:** {true_label}")
    
    with col2:
        if st.button("🚀 Generate Explanations", type="primary"):
            with st.spinner("Generating explanations... This may take a minute."):
                try:
                    # First, get prediction
                    pred_payload = {
                        "tensor": images_batch[idx].tolist(),
                        "true_label_idx": true_label_idx
                    }
                    pred_response = requests.post(PREDICTION_BACKEND_ENDPOINT, json=pred_payload)
                    
                    if pred_response.status_code == 200:
                        predicted_class_idx = pred_response.json()["predicted_class"]
                        predicted_label = dataset.classes[predicted_class_idx]
                        
                        # Generate explanation for true class
                        xai_payload = {
                            "tensor": images_batch[idx].tolist(),
                            "target_class": true_label_idx
                        }
                        xai_response = requests.post(LOCAL_XAI_ENDPOINT, json=xai_payload)
                        
                        if xai_response.status_code == 200:
                            results = {
                                "predicted_label": predicted_label,
                                "predicted_class_idx": predicted_class_idx,
                                "true_class_results": xai_response.json()
                            }
                            
                            # If prediction differs, get explanation for predicted class too
                            if predicted_class_idx != true_label_idx:
                                xai_payload_pred = {
                                    "tensor": images_batch[idx].tolist(),
                                    "target_class": predicted_class_idx
                                }
                                xai_response_pred = requests.post(LOCAL_XAI_ENDPOINT, json=xai_payload_pred)
                                if xai_response_pred.status_code == 200:
                                    results["pred_class_results"] = xai_response_pred.json()
                            
                            st.session_state.local_xai_results = results
                            st.success("✅ Explanations generated successfully!")
                            st.rerun()
                        else:
                            st.error(f"XAI Error {xai_response.status_code}: {xai_response.text}")
                    else:
                        st.error(f"Prediction Error {pred_response.status_code}: {pred_response.text}")
                        
                except Exception as e:
                    st.error(f"Request failed: {e}")
    
    # Display results if available
    if st.session_state.local_xai_results is not None:
        results = st.session_state.local_xai_results
        predicted_label = results["predicted_label"]
        
        st.markdown("---")
        st.subheader("Explanation Results")
        
        # Show prediction result
        if predicted_label == true_label:
            st.success(f"**Prediction:** {predicted_label} ✅ (Correct)")
        else:
            st.error(f"**Prediction:** {predicted_label} ❌ (Incorrect - True: {true_label})")
        
        st.markdown("### Saliency Maps")
        
        # Display explanations for true class
        st.markdown(f"#### Explanations for True Class: **{true_label}**")
        col1, col2 = st.columns(2)
        
        true_results = results["true_class_results"]
        
        with col1:
            render_saliency_visualization(
                true_results["gradcam_map"],
                "GradCAM",
                true_label,
                true_results["gradcam_faithfulness"],
                true_results["gradcam_robustness"],
                deletion_curve=true_results.get("gradcam_deletion_curve")
            )
        
        with col2:
            render_saliency_visualization(
                true_results["integrated_gradients_map"],
                "Integrated Gradients",
                true_label,
                true_results["ig_faithfulness"],
                true_results["ig_robustness"],
                deletion_curve=true_results.get("ig_deletion_curve")
            )
        
        # Display explanations for predicted class if different
        if "pred_class_results" in results:
            st.markdown(f"#### Explanations for Predicted Class: **{predicted_label}**")
            col3, col4 = st.columns(2)
            
            pred_results = results["pred_class_results"]
            
            with col3:
                render_saliency_visualization(
                    pred_results["gradcam_map"],
                    "GradCAM",
                    predicted_label,
                    pred_results["gradcam_faithfulness"],
                    pred_results["gradcam_robustness"],
                    deletion_curve=pred_results.get("gradcam_deletion_curve")
                )
            
            with col4:
                render_saliency_visualization(
                    pred_results["integrated_gradients_map"],
                    "Integrated Gradients",
                    predicted_label,
                    pred_results["ig_faithfulness"],
                    pred_results["ig_robustness"],
                    deletion_curve=pred_results.get("ig_deletion_curve")
                )
