
import streamlit as st
import numpy as np
import mediapipe as mp
from PIL import Image, ImageFilter
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
from io import BytesIO

st.set_page_config(
    page_title="Croquis Studio",
    page_icon="✏️",
    layout="wide"
)

st.title("✏️ Croquis Studio")
st.write("Create a printable fashion drawing guide from a model photo.")

MODEL_PATH = "pose_landmarker.task"

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/1/"
    "pose_landmarker_lite.task"
)


@st.cache_resource
def load_detector():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    base_options = python.BaseOptions(
        model_asset_path=MODEL_PATH
    )

    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        output_segmentation_masks=True
    )

    return vision.PoseLandmarker.create_from_options(options)


uploaded = st.file_uploader(
    "Upload a full-body model photo",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded is not None:
    try:
        image = Image.open(uploaded).convert("RGB")

        st.subheader("Original photo")
        st.image(image, use_container_width=True)

        if st.button("Create fashion outline"):
            with st.spinner("Detecting the body outline..."):

                detector = load_detector()

                rgb = np.array(image, dtype=np.uint8)
                rgb = np.ascontiguousarray(rgb)

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                result = detector.detect(mp_image)

                if not result.pose_landmarks:
                    st.warning(
                        "No person detected. Try a clear, "
                        "full-body photo."
                    )

                elif not result.segmentation_masks:
                    st.warning(
                        "The body outline could not be generated. "
                        "Please try another photo."
                    )

                else:
                    # Get the person's segmentation mask.
                    mask_array = (
                        result.segmentation_masks[0].numpy_view()
                    )

                    mask_array = np.squeeze(mask_array)

                    # Convert the mask into a black-and-white image.
                    mask = Image.fromarray(
                        (mask_array > 0.5).astype(np.uint8) * 255
                    )

                    # Match the original photo dimensions.
                    mask = mask.resize(
                        image.size,
                        Image.Resampling.LANCZOS
                    )

                    # Clean the mask slightly.
                    mask = mask.filter(
                        ImageFilter.MedianFilter(size=3)
                    )

                    # Find the outer edges of the person.
                    edges = mask.filter(
                        ImageFilter.FIND_EDGES
                    )

                    # Make the outline easier to see.
                    edges = edges.filter(
                        ImageFilter.MaxFilter(size=3)
                    )

                    edges = edges.point(
                        lambda p: 255 if p > 20 else 0
                    )

                    # Create a white page with black contours.
                    outline = Image.new(
                        "RGB",
                        image.size,
                        "white"
                    )

                    outline.paste(
                        "black",
                        mask=edges
                    )

                    st.subheader("Your fashion outline")
                    st.image(
                        outline,
                        use_container_width=True
                    )

                    # Prepare a high-resolution PNG download.
                    output = BytesIO()

                    outline.save(
                        output,
                        format="PNG",
                        dpi=(300, 300)
                    )

                    st.download_button(
                        "Download fashion outline",
                        data=output.getvalue(),
                        file_name="croquis_outline.png",
                        mime="image/png"
                    )

                    st.success(
                        "Your outline is ready!"
                    )

                    st.caption(
                        "The outline follows the detected body "
                        "silhouette. Results depend on the photo "
                        "and may need some manual refinement."
                    )

    except Exception as e:
        st.error(f"Could not process the image: {e}")

else:
    st.info("Upload a full-body photo to get started.")
