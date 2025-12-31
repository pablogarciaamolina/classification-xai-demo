from typing import Callable

import streamlit as st

def render_gallery(images: list, labels: list[str] = None, on_select: Callable = None, max_num_cols: int = 5) -> None:
    """
    Renders a grid gallery of images with a centered button and centered label underneath.

    Args:
        images: A list of all the arrays streamlit-ready for display .
        labels: An optional list of the true labels for each image, to give more information to the user.
        on_select: Callback method for when pressing the image button.
        max_num_cols: Maximum numner of colums for the gallery display. Defaults to 5.
    """
    num_cols = min(max_num_cols, len(images))
    cols = st.columns(num_cols)

    for i, img in enumerate(images):
        col = cols[i % num_cols]
        with col:
            button_label = f"Nº {i + 1}"
            if st.button(button_label, key=f"btn_{i}"):
                if on_select:
                    on_select(i)

            st.image(img, clamp=True, width="stretch")

            # if labels is not None:
            #     st.markdown(f'<p style="text-align: center;"><b>{labels[i]}</b></p>', unsafe_allow_html=True)
