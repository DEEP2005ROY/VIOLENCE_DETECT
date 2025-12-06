import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

st.set_page_config(page_title="Violence Detection", layout="centered")

st.title("🔍 Violence Detection App")
st.write("Upload an image and the YOLO model will classify it as Violence / NonViolence.")

@st.cache_resource
def load_model():
    model = YOLO("best.pt")    # Make sure best.pt is in same folder
    return model

model = load_model()

uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Convert PIL → numpy
    img_np = np.array(img)

    # Predict
    results = model.predict(img_np)

    # Render prediction
    pred_img = results[0].plot()  # YOLO returns numpy array with boxes drawn

    st.image(pred_img, caption="Prediction Result", use_column_width=True)

    # Text prediction
    probs = results[0].probs
    if probs is not None:
        st.subheader("Prediction:")
        class_id = int(np.argmax(probs.data))
        conf = float(probs.data[class_id])
        label = results[0].names[class_id]
        st.write(f"**{label}** ({conf:.2f} confidence)")

