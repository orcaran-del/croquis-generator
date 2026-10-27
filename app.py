
import streamlit as st
from PIL import Image
from io import BytesIO

st.set_page_config(
    page_title="Croquis Studio",
    page_icon="✏️",
    layout="wide"
)

st.title("✏️ Croquis Studio")
st.write("Your fashion sketch workspace")
st.write("Upload a model photo to get started.")

uploaded_file = st.file_uploader(
    "Choose a model photo",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Your uploaded photo")
    st.image(image, use_container_width=True)

    image_bytes = BytesIO()
    image.save(image_bytes, format="PNG")

    st.download_button(
        label="Download image",
        data=image_bytes.getvalue(),
        file_name="my_model_photo.png",
        mime="image/png"
    )

    st.info(
        "Next, we'll work on converting "
        "your model photo into a fashion croquis."
    )
else:
    st.info("Your model photo will appear here.")
0--
