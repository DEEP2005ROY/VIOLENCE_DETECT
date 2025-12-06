import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

st.title("Violence Detection App")

model = YOLO("best.pt")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Predict
    results = model.predict(img)

    # show annotated image
    res_img = results[0].plot()  # NumPy array
    st.image(res_img, caption="Prediction", use_column_width=True)
