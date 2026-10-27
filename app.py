
import streamlit as st
from PIL import Image, ImageFilter, ImageOps
from io import BytesIO

st.set_page_config(
    page_title="Croquis Studio",
    page_icon="✏️",
    layout="wide"
)

st.title("✏️ Croquis Studio")
st.write("Create clearer outlines for fashion design.")

uploaded_file = st.file_uploader(
    "Upload a full-body model photo",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    original = Image.open(uploaded_file).convert("RGB")

    st.subheader("Original Photo")
    st.image(original, use_container_width=True)

    thickness = st.select_slider(
        "Outline thickness",
        options=["Thin", "Medium", "Thick"],
        value="Medium"
    )

    filter_size = {
        "Thin": 1,
        "Medium": 3,
        "Thick": 5
    }[thickness]

    if st.button("✏️ Generate Croquis", type="primary"):
        with st.spinner("Improving your outlines..."):
            # Reduce tiny details and noise
            gray = ImageOps.grayscale(original)
            gray = gray.filter(ImageFilter.GaussianBlur(0.5))

            # Detect the edges
            edges = gray.filter(ImageFilter.FIND_EDGES)
            edges = ImageOps.invert(edges)
            edges = ImageOps.autocontrast(edges)

            # Keep more faint outline details
            croquis = edges.point(
                lambda p: 255 if p > 220 else 0
            )

            # Expand black lines to help close small gaps
            if filter_size > 1:
                croquis = croquis.filter(
                    ImageFilter.MinFilter(filter_size)
                )

            st.session_state["croquis_result"] = croquis
            st.session_state["croquis_source"] = uploaded_file.name

    if (
        "croquis_result" in st.session_state
        and st.session_state.get("croquis_source")
        == uploaded_file.name
    ):
        result = st.session_state["croquis_result"]

        st.subheader("Your Croquis Preview")
        st.image(
            result,
            caption="Black outlines on a white background",
            use_container_width=True
        )

        output = BytesIO()
        result.save(output, format="PNG")

        st.download_button(
            "⬇️ Download Croquis",
            data=output.getvalue(),
            file_name="my_croquis.png",
            mime="image/png"
        )

else:
    st.info("Upload a model photo to get started.")
