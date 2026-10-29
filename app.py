
import streamlit as st
import numpy as np
import mediapipe as mp
from PIL import Image
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os

st.set_page_config(
    page_title="Croquis Studio",
    page_icon="✏️",
    layout="wide"
)

st.title("✏️ Croquis Studio")
st.write("Turn a model photo into a pose guide for fashion drawing.")

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
        num_poses=1
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

        if st.button("Detect the pose"):
            with st.spinner("Detecting pose..."):
                detector = load_detector()

                rgb = np.array(image, dtype=np.uint8)
                rgb = np.ascontiguousarray(rgb)

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                result = detector.detect(mp_image)

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]

                # Draw the detected body landmarks on a white canvas.
                import matplotlib.pyplot as plt

                connections = (
                    vision.PoseLandmarksConnections.POSE_LANDMARKS
                    if hasattr(vision, "PoseLandmarksConnections")
                    else []
                )

                fig, ax = plt.subplots(figsize=(5, 8))
                ax.set_xlim(0, 1)
                ax.set_ylim(1, 0)
                ax.set_aspect("equal")
                ax.axis("off")
                fig.patch.set_facecolor("white")
                ax.set_facecolor("white")

                points = [
                    (lm.x, lm.y) for lm in landmarks
                ]

                # Connect body landmarks when the connection list exists.
                for connection in connections:
                    start = connection.start
                    end = connection.end

                    if start < len(points) and end < len(points):
                        x1, y1 = points[start]
                        x2, y2 = points[end]
                        ax.plot(
                            [x1, x2],
                            [y1, y2],
                            color="black",
                            linewidth=2
                        )

                # Show every detected landmark.
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                ax.scatter(xs, ys, color="black", s=12)

                st.subheader("Detected pose guide")
                st.pyplot(fig)

                from io import BytesIO

                output = BytesIO()
                fig.savefig(
                    output,
                    format="png",
                    dpi=300,
                    bbox_inches="tight",
                    facecolor="white"
                )
                plt.close(fig)
                output.seek(0)

                st.download_button(
                    "Download pose guide (PNG)",
                    data=output.getvalue(),
                    file_name="croquis_pose_guide.png",
                    mime="image/png"
                )

                st.info(
                    "This is a pose guide, not a finished fashion "
                    "croquis. You can use it as a reference for drawing."
                )
            else:
                st.warning(
                    "No pose detected. Try a clear, full-body photo "
                    "with the arms and legs visible."
                )

    except Exception as e:
        st.error(f"Could not process the image: {e}")
else:
    st.info("Upload a photo to get started.")
