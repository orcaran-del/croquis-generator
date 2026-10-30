
import streamlit as st
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw
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
st.write("Turn a model photo into a clean fashion croquis.")

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

    base = python.BaseOptions(model_asset_path=MODEL_PATH)

    options = vision.PoseLandmarkerOptions(
        base_options=base,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1
    )

    return vision.PoseLandmarker.create_from_options(options)


def midpoint(a, b):
    return (
        (a[0] + b[0]) / 2,
        (a[1] + b[1]) / 2
    )


def draw_croquis(landmarks, width, height):
    scale = 3
    canvas = Image.new(
        "RGB",
        (width * scale, height * scale),
        "white"
    )

    draw = ImageDraw.Draw(canvas)

    def point(index):
        lm = landmarks[index]
        return (
            int(lm.x * width * scale),
            int(lm.y * height * scale)
        )

    def line(points, thickness=5):
        draw.line(
            points,
            fill="black",
            width=thickness * scale,
            joint="curve"
        )

    def joint(p, radius=4):
        r = radius * scale
        draw.ellipse(
            (p[0] - r, p[1] - r,
             p[0] + r, p[1] + r),
            fill="black"
        )

    # Main body landmarks
    nose = point(0)
    left_shoulder = point(11)
    right_shoulder = point(12)
    left_elbow = point(13)
    right_elbow = point(14)
    left_wrist = point(15)
    right_wrist = point(16)

    left_hip = point(23)
    right_hip = point(24)
    left_knee = point(25)
    right_knee = point(26)
    left_ankle = point(27)
    right_ankle = point(28)
    left_foot = point(31)
    right_foot = point(32)

    shoulder_mid = midpoint(left_shoulder, right_shoulder)
    hip_mid = midpoint(left_hip, right_hip)

    # Head
    head_size = max(
        12 * scale,
        int(abs(shoulder_mid[1] - nose[1]) * 0.8)
    )

    head_box = (
        nose[0] - head_size // 2,
        nose[1] - head_size,
        nose[0] + head_size // 2,
        nose[1]
    )

    draw.ellipse(
        head_box,
        outline="black",
        width=5 * scale
    )

    # Neck
    neck_top = (
        shoulder_mid[0],
        shoulder_mid[1] - 3 * scale
    )

    neck_bottom = (
        shoulder_mid[0],
        shoulder_mid[1] + 8 * scale
    )

    line([neck_top, neck_bottom], 4)

    # Torso outline
    torso_top_left = (
        left_shoulder[0],
        left_shoulder[1]
    )
    torso_top_right = (
        right_shoulder[0],
        right_shoulder[1]
    )

    waist_left = (
        int(left_shoulder[0] * 0.35 + left_hip[0] * 0.65),
        int(left_shoulder[1] * 0.35 + left_hip[1] * 0.65)
    )

    waist_right = (
        int(right_shoulder[0] * 0.35 + right_hip[0] * 0.65),
        int(right_shoulder[1] * 0.35 + right_hip[1] * 0.65)
    )

    torso_points = [
        neck_bottom,
        torso_top_left,
        waist_left,
        left_hip,
        right_hip,
        waist_right,
        torso_top_right,
        neck_bottom
    ]

    line(torso_points, 5)

    # Arms
    for shoulder, elbow, wrist in [
        (left_shoulder, left_elbow, left_wrist),
        (right_shoulder, right_elbow, right_wrist)
    ]:
        line([shoulder, elbow, wrist], 5)
        joint(elbow, 3)
        joint(wrist, 3)

    # Legs
    for hip, knee, ankle, foot in [
        (left_hip, left_knee, left_ankle, left_foot),
        (right_hip, right_knee, right_ankle, right_foot)
    ]:
        line([hip, knee, ankle], 6)
        line([ankle, foot], 4)
        joint(knee, 3)
        joint(ankle, 3)

    # Small shoulder and hip details
    joint(left_shoulder, 3)
    joint(right_shoulder, 3)
    joint(left_hip, 3)
    joint(right_hip, 3)

    # Return to original image dimensions
    canvas = canvas.resize(
        (width, height),
        Image.Resampling.LANCZOS
    )

    return canvas


uploaded = st.file_uploader(
    "Upload a full-body model photo",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")

    st.subheader("Original photo")
    st.image(image, use_container_width=True)

    if st.button("Create fashion croquis"):
        with st.spinner("Creating your croquis..."):
            try:
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
                        "No body detected. Try a clear full-body photo."
                    )

                else:
                    landmarks = result.pose_landmarks[0]

                    croquis = draw_croquis(
                        landmarks,
                        image.width,
                        image.height
                    )

                    st.subheader("Your fashion croquis")
                    st.image(
                        croquis,
                        use_container_width=True
                    )

                    output = BytesIO()

                    croquis.save(
                        output,
                        format="PNG",
                        dpi=(300, 300)
                    )

                    st.download_button(
                        "Download croquis",
                        data=output.getvalue(),
                        file_name="fashion_croquis.png",
                        mime="image/png"
                    )

                    st.success("Your croquis is ready!")

                    st.caption(
                        "This is a simplified pose-based drawing. "
                        "Body contours are estimated, not reconstructed "
                        "from hidden clothing details."
                    )

            except Exception as e:
                st.error(f"Something went wrong: {e}")

else:
    st.info("Upload a full-body photo to get started.")
