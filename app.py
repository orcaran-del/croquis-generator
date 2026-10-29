
import os
import urllib.request
from io import BytesIO

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFilter
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

st.set_page_config(
    page_title="Croquis Studio",
    page_icon="✏️",
    layout="wide"
)

st.title("✏️ Croquis Studio")
st.write("Pose detection prototype for fashion design.")

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/1/"
    "pose_landmarker_full.task"
)
MODEL_PATH = "pose_landmarker_full.task"


@st.cache_resource
def load_pose_detector():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=MODEL_PATH
        ),
        running_mode=vision.RunningMode.IMAGE,
        output_segmentation_masks=True
    )

    return vision.PoseLandmarker.create_from_options(options)


uploaded = st.file_uploader(
    "Upload a full-body model photo",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.subheader("Original photo")
    st.image(image, use_container_width=True)

    if st.button("Detect Pose", type="primary"):
        try:
            with st.spinner("Detecting body pose..."):
                detector = load_pose_detector()

                rgb = np.array(image.convert("RGB"), dtype=np.uint8)
rgb = np.ascontiguousarray(rgb)

mp_image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=rgb
)

result = detector.detect(mp_image)

                if not result.pose_landmarks:
                    st.warning(
                        "No pose detected. Try a clear, "
                        "full-body photo with visible limbs."
                    )
                else:
                    points = result.pose_landmarks[0]
                    w, h = image.size

                    # Draw the estimated body skeleton.
                    skeleton = Image.new("RGB", (w, h), "white")
                    draw = ImageDraw.Draw(skeleton)

                    connections = [
                        (11, 12),
                        (11, 13), (13, 15),
                        (12, 14), (14, 16),
                        (11, 23), (12, 24),
                        (23, 24),
                        (23, 25), (25, 27),
                        (24, 26), (26, 28),
                        (27, 31), (28, 32)
                    ]

                    for a, b in connections:
                        p1, p2 = points[a], points[b]
                        if (p1.visibility > 0.45
                                and p2.visibility > 0.45):
                            draw.line(
                                (p1.x * w, p1.y * h,
                                 p2.x * w, p2.y * h),
                                fill="black",
                                width=max(2, w // 250)
                            )

                    for p in points:
                        if p.visibility > 0.45:
                            x, y = int(p.x * w), int(p.y * h)
                            r = max(3, w // 150)
                            draw.ellipse(
                                (x-r, y-r, x+r, y+r),
                                fill="black"
                            )

                    st.session_state["skeleton"] = skeleton

                    # Create an outer silhouette outline.
                    if result.segmentation_masks:
                        mask = result.segmentation_masks[0].numpy_view()
                        mask_image = Image.fromarray(
                            (mask * 255).clip(0, 255).astype("uint8")
                        ).resize((w, h))

                        mask_image = mask_image.filter(
                            ImageFilter.GaussianBlur(1)
                        )
                        mask_image = mask_image.point(
                            lambda p: 255 if p > 127 else 0
                        )

                        expanded = mask_image.filter(
                            ImageFilter.MaxFilter(5)
                        )
                        contracted = mask_image.filter(
                            ImageFilter.MinFilter(5)
                        )

                        outer = np.array(expanded, dtype=np.int16)
                        inner = np.array(contracted, dtype=np.int16)
                        outline = np.where(
                            outer > inner, 0, 255
                        ).astype("uint8")

                        st.session_state["silhouette"] = (
                            Image.fromarray(outline)
                        )

                    st.success("Pose detection complete!")

        except Exception as e:
            st.error(f"Could not process the image: {e}")

    if "skeleton" in st.session_state:
        st.subheader("Detected pose")
        st.image(
            st.session_state["skeleton"],
            caption="Estimated body joints and pose",
            use_container_width=True
        )

        output = BytesIO()
        st.session_state["skeleton"].save(output, format="PNG")
        st.download_button(
            "Download pose guide",
            output.getvalue(),
            "pose_guide.png",
            "image/png"
        )

    if "silhouette" in st.session_state:
        st.subheader("Person silhouette outline")
        st.image(
            st.session_state["silhouette"],
            caption="Outer silhouette, not yet a full fashion croquis",
            use_container_width=True
        )

        output = BytesIO()
        st.session_state["silhouette"].save(output, format="PNG")
        st.download_button(
            "Download silhouette",
            output.getvalue(),
            "silhouette_outline.png",
            "image/png"
        )
else:
    st.info("Upload a full-body photo to begin.")
