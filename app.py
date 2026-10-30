
import streamlit as st
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
from io import BytesIO
import math

st.set_page_config(
    page_title="Croquis Studio",
    page_icon="✏️",
    layout="wide"
)

st.title("✏️ Croquis Studio")
st.write("Create a clean fashion croquis from a model photo.")

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


def create_croquis(landmarks, width, height):

    scale = 3
    W = width * scale
    H = height * scale

    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)

    def p(index):
        lm = landmarks[index]
        return (int(lm.x * W), int(lm.y * H))

    def midpoint(a, b):
        return (
            int((a[0] + b[0]) / 2),
            int((a[1] + b[1]) / 2)
        )

    def distance(a, b):
        return math.hypot(b[0] - a[0], b[1] - a[1])

    def segment(a, b, width_a, width_b=None):
        if width_b is None:
            width_b = width_a

        dx = b[0] - a[0]
        dy = b[1] - a[1]
        length = math.hypot(dx, dy)

        if length < 2:
            return

        nx = -dy / length
        ny = dx / length

        polygon = [
            (int(a[0] + nx * width_a),
             int(a[1] + ny * width_a)),
            (int(b[0] + nx * width_b),
             int(b[1] + ny * width_b)),
            (int(b[0] - nx * width_b),
             int(b[1] - ny * width_b)),
            (int(a[0] - nx * width_a),
             int(a[1] - ny * width_a))
        ]

        draw.polygon(
            polygon,
            fill="white",
            outline="black",
            width=4 * scale
        )

        # Rounded ends
        for center, radius in [
            (a, width_a),
            (b, width_b)
        ]:
            draw.ellipse(
                (
                    center[0] - radius,
                    center[1] - radius,
                    center[0] + radius,
                    center[1] + radius
                ),
                fill="white",
                outline="black",
                width=4 * scale
            )

    def joint(center, radius):
        draw.ellipse(
            (
                center[0] - radius,
                center[1] - radius,
                center[0] + radius,
                center[1] + radius
            ),
            fill="white",
            outline="black",
            width=4 * scale
        )

    # Main landmarks
    nose = p(0)

    ls, rs = p(11), p(12)
    le, re = p(13), p(14)
    lw, rw = p(15), p(16)

    lh, rh = p(23), p(24)
    lk, rk = p(25), p(26)
    la, ra = p(27), p(28)
    lf, rf = p(31), p(32)

    shoulder_mid = midpoint(ls, rs)
    hip_mid = midpoint(lh, rh)

    shoulder_width = max(distance(ls, rs), 15 * scale)
    hip_width = max(distance(lh, rh), 12 * scale)

    # Head
    head_radius_x = int(shoulder_width * 0.23)
    head_radius_y = int(head_radius_x * 1.25)

    head_center = (
        nose[0],
        int(nose[1] - head_radius_y * 0.45)
    )

    draw.ellipse(
        (
            head_center[0] - head_radius_x,
            head_center[1] - head_radius_y,
            head_center[0] + head_radius_x,
            head_center[1] + head_radius_y
        ),
        fill="white",
        outline="black",
        width=5 * scale
    )

    # Neck
    neck_width = int(shoulder_width * 0.10)

    neck_top = (
        shoulder_mid[0],
        head_center[1] + head_radius_y - 2 * scale
    )

    neck_bottom = (
        shoulder_mid[0],
        shoulder_mid[1]
    )

    segment(neck_top, neck_bottom, neck_width)

    # Torso shape
    torso_length = max(
        distance(shoulder_mid, hip_mid),
        20 * scale
    )

    waist_y = int(
        shoulder_mid[1] + torso_length * 0.55
    )

    waist_center = (
        int((shoulder_mid[0] + hip_mid[0]) / 2),
        waist_y
    )

    waist_width = int(
        min(shoulder_width * 0.29, hip_width * 0.42)
    )

    shoulder_left = ls
    shoulder_right = rs

    waist_left = (
        waist_center[0] - waist_width,
        waist_center[1]
    )

    waist_right = (
        waist_center[0] + waist_width,
        waist_center[1]
    )

    torso_points = [
        (shoulder_mid[0] - neck_width, neck_bottom[1]),
        shoulder_left,
        waist_left,
        lh,
        rh,
        waist_right,
        shoulder_right,
        (shoulder_mid[0] + neck_width, neck_bottom[1])
    ]

    draw.polygon(
        torso_points,
        fill="white",
        outline="black",
        width=5 * scale
    )

    # Arms
    arm_width = shoulder_width * 0.12
    forearm_width = shoulder_width * 0.085

    for shoulder, elbow, wrist in [
        (ls, le, lw),
        (rs, re, rw)
    ]:
        segment(
            shoulder,
            elbow,
            int(arm_width),
            int(arm_width * 0.8)
        )

        segment(
            elbow,
            wrist,
            int(forearm_width),
            int(forearm_width * 0.65)
        )

        joint(elbow, int(arm_width * 0.7))

    # Legs
    thigh_width = hip_width * 0.22
    calf_width = hip_width * 0.14

    for hip, knee, ankle, foot in [
        (lh, lk, la, lf),
        (rh, rk, ra, rf)
    ]:
        segment(
            hip,
            knee,
            int(thigh_width),
            int(thigh_width * 0.72)
        )

        segment(
            knee,
            ankle,
            int(calf_width),
            int(calf_width * 0.55)
        )

        joint(knee, int(calf_width * 0.65))

        # Feet
        foot_width = int(calf_width * 0.75)

        segment(
            ankle,
            foot,
            foot_width,
            int(foot_width * 0.65)
        )

    # Downsample for smoother lines
    canvas = canvas.resize(
        (width, height),
        Image.Resampling.LANCZOS
    )

    return canvas


uploaded = st.file_uploader(
    "Upload a full-body model photo",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded is not None:

    image = Image.open(uploaded).convert("RGB")

    st.subheader("Original photo")
    st.image(image, use_container_width=True)

    if st.button("Create fashion croquis"):

        with st.spinner("Drawing your fashion figure..."):

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
                        "No person detected. Try a clear full-body photo."
                    )

                else:
                    croquis = create_croquis(
                        result.pose_landmarks[0],
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
                        "Download fashion croquis",
                        data=output.getvalue(),
                        file_name="fashion_croquis.png",
                        mime="image/png"
                    )

                    st.success("Croquis created!")

                    st.caption(
                        "This is an estimated pose-based fashion figure. "
                        "Hidden body contours cannot be recovered exactly "
                        "from clothing."
                    )

            except Exception as e:
                st.error(f"Could not create croquis: {e}")

else:
    st.info("Upload a full-body photo to begin.")
