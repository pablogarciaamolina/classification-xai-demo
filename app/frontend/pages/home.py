"""
Home page for the frontend
"""

import streamlit as st

from app.frontend.config import IMAGES_PER_PAGE, MAX_GALLERY_COLUMNS, PREDICTION_BACKEND_ENDPOINT
from components.utils import get_stl10_dataset, get_stl10_dataloader, load_new_batch, tensor_to_display
from components.gallery import render_gallery
from components.selected_view import render_selected


st.title("STL10 predictor - Deployment demo")

# Load data
dataset = get_stl10_dataset()
dataloader = get_stl10_dataloader(dataset, batch_size=IMAGES_PER_PAGE)

# Set state
for key in ["batch", "labels", "selected_index"]:
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
    st.rerun()

# ======================
# Preview and prediction
# ======================
if st.session_state.selected_index is not None:
    idx = st.session_state.selected_index
    render_selected(
        image=displayable_images[idx],
        true_label_idx=labels_batch[idx].item(),
        tensor_for_backend=images_batch[idx],
        backend_url=PREDICTION_BACKEND_ENDPOINT,
        dataset_classes=dataset.classes
    )
