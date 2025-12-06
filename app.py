import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="Violence Detection", layout="centered")

st.title("🔍 Violence Detection App")
st.write("Upload an image and the YOLO model will classify it (Violence / NonViolence).")

@st.cache_resource
def load_model():
    # best.pt must be in the same folder as app.py
    return YOLO("best.pt")

model = load_model()

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Convert PIL -> numpy array for YOLO
    img_np = np.array(img)

    # Run prediction
    results = model.predict(img_np)

    # Render image with bounding boxes / labels
    annotated = results[0].plot()  # returns a NumPy array
    st.image(annotated, caption="Prediction", use_column_width=True)

    # Show class + confidence (for classification models)
    if hasattr(results[0], "probs") and results[0].probs is not None:
        probs = results[0].probs.data
        class_id = int(np.argmax(probs))
        conf = float(probs[class_id])
        label = results[0].names[class_id]
        st.subheader("Prediction")
        st.write(f"**{label}** ({conf:.2f} confidence)")

