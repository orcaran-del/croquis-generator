
import streamlit as st
from PIL import Image, ImageFilter, ImageOps
from io import BytesIO

st.set_page_config(
    page_title="Croquis Studio",
    page_icon="✏️",
    layout="wide"
)

st.title("✏️ Croquis Studio")
st.write("Turn your model photos into sketch-style fashion references.")

st.divider()

uploaded_file = st.file_uploader(
    "Upload a full-body model photo",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    original = Image.open(uploaded_file).convert("RGB")

    st.subheader("Your model photo")
    st.image(original, use_container_width=True)

    if st.button("✏️ Generate Croquis", type="primary"):
        with st.spinner("Creating your outline..."):
            # Convert the photo into a black-and-white outline
            gray = ImageOps.grayscale(original)
            edges = gray.filter(ImageFilter.FIND_EDGES)
            edges = ImageOps.invert(edges)
            edges = ImageOps.autocontrast(edges)

            # Make the outline clearer
            threshold = 190
            croquis = edges.point(
                lambda pixel: 255 if pixel > threshold else 0
            )

            st.session_state["croquis_result"] = croquis

    if "croquis_result" in st.session_state:
        st.subheader("Your Croquis Preview")

        result = st.session_state["croquis_result"]
        st.image(
            result,
            caption="First outline prototype",
            use_container_width=True
        )

        image_bytes = BytesIO()
        result.save(image_bytes, format="PNG")

        st.download_button(
            label="⬇️ Download Croquis",
            data=image_bytes.getvalue(),
            file_name="my_croquis.png",
            mime="image/png"
        )

        st.caption(
            "This prototype creates an outline from your photo. "
            "AI-based fashion figure generation is a future step."
        )

else:
    st.info("Upload a model photo to get started.")
