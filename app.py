\
import streamlit as st
from PIL import Image, ImageFilter, ImageOps
from io import BytesIO

st.set_page_config(
    page_title="Croquis Studio",
    page_icon="✏️",
    layout="wide"
)

st.title("✏️ Croquis Studio")
st.write("Create cleaner outlines for fashion design.")

uploaded_file = st.file_uploader(
    "Upload a full-body model photo",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    original = Image.open(uploaded_file).convert("RGB")

    st.subheader("Original Photo")
    st.image(original, use_container_width=True)

    threshold = st.slider(
        "Remove small details and noise",
        min_value=200,
        max_value=250,
        value=235,
        step=5,
        help="Higher values keep fewer faint edges."
    )

    if st.button("Generate Croquis", type="primary"):
        with st.spinner("Creating cleaner outlines..."):
            # Reduce fine texture and small dots
            gray = ImageOps.grayscale(original)
            gray = gray.filter(ImageFilter.MedianFilter(3))
            gray = gray.filter(ImageFilter.GaussianBlur(1.5))

            # Detect edges
            edges = gray.filter(ImageFilter.FIND_EDGES)
            edges = ImageOps.invert(edges)
            edges = ImageOps.autocontrast(edges)

            # White background and thin black lines
            croquis = edges.point(
                lambda p: 255 if p > threshold else 0
            )

            # Do not thicken the lines
            st.session_state["croquis_result"] = croquis
            st.session_state["croquis_source"] = uploaded_file.name

    if (
        "croquis_result" in st.session_state
        and st.session_state.get("croquis_source")
        == uploaded_file.name
    ):
        result = st.session_state["croquis_result"]

        st.subheader("Croquis Preview")
        st.image(result, use_container_width=True)

        output = BytesIO()
        result.save(output, format="PNG")

        st.download_button(
            "Download Croquis",
            data=output.getvalue(),
            file_name="my_croquis.png",
            mime="image/png"
        )
else:
    st.info("Upload a model photo to get started.")
