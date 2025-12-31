"""
Home page for the frontend
"""

import streamlit as st

from app.frontend.config import IMAGES_PER_PAGE, MAX_GALLERY_COLUMNS, PREDICTION_BACKEND_ENDPOINT, DATASET_METADATA
from components.utils import get_dataset, get_dataloader, load_new_batch, tensor_to_display
from components.gallery import render_gallery
from components.selected_view import render_selected

if "selected_dataset" not in st.session_state or st.session_state.selected_dataset is None:
    st.warning("⚠️ No dataset selected. Please select a dataset first.")
    st.page_link("pages/dataset_selection.py", label="Go to Dataset Selection", icon="🎯")
    st.stop()

dataset_name = st.session_state.selected_dataset
dataset_display_name = DATASET_METADATA[dataset_name]["display_name"]

st.title(f"{DATASET_METADATA[dataset_name]['icon']} {dataset_display_name} Predictor")

dataset = get_dataset(dataset_name)
dataloader = get_dataloader(dataset, batch_size=IMAGES_PER_PAGE)

for key in ["batch", "labels", "selected_index"]:
    st.session_state.setdefault(key, None)

if st.session_state.batch is None:
    images, labels = load_new_batch(dataloader)
    st.session_state.batch = images
    st.session_state.labels = labels

images_batch = st.session_state.batch
labels_batch = st.session_state.labels

displayable_images = [tensor_to_display(img) for img in images_batch]
textual_labels = [dataset.classes[idx] for idx in labels_batch]

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

if st.session_state.selected_index is not None:
    idx = st.session_state.selected_index
    render_selected(
        image=displayable_images[idx],
        true_label_idx=labels_batch[idx].item(),
        tensor_for_backend=images_batch[idx],
        backend_url=PREDICTION_BACKEND_ENDPOINT,
        dataset_classes=dataset.classes
    )
