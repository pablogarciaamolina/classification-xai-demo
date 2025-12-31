import requests

import numpy as np
import streamlit as st
import torch

def render_selected(
    image: np.ndarray,
    true_label_idx: int,
    tensor_for_backend: torch.Tensor,
    backend_url: str,
    dataset_classes: list[str]
) -> None:
    """
    Displays the selected image with the option to predict it.
    
    Args:
        image: The image array ready for display.
        true_label: The true label for the image.
        tensor_for_backend: The corresponding original tensor. Vital for preventing the underperformance of the model due to rounding errors.
        backend_url: Backend url for prediction.
        dataset_classes: The possible classes tags.
    """
    st.markdown("---")
    st.subheader("Selected Image for Prediction")

    colA, colB = st.columns([1, 2])
    true_label = dataset_classes[true_label_idx]

    with colA:
        st.image(image, width=350, clamp=True)

    with colB:
        st.write(f"True Label: **{true_label}**")

        if st.button("🚀 Predict Diagnosis", type="primary"):
            payload = {"tensor": tensor_for_backend.tolist(), "true_label_idx": true_label_idx}
            try:
                response = requests.post(backend_url, json=payload)
                if response.status_code == 200:
                    pred = response.json()["predicted_class"]
                    pred_label = dataset_classes[pred]
                    display_type = st.success if pred_label == true_label else st.error
                    display_type(f"Prediction: **{pred_label}**")
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
            except Exception as e:
                st.error(f"Prediction request failed: {e}")
