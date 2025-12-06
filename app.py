import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="Violence Detection", layout="centered")

st.title("🔍 Violence Detection Classifier")
st.write("Upload an image and the model will predict whether it is **Violence** or **NonViolence**.")

@st.cache_resource
def load_model():
    # best.pt must be in the same folder as app.py
    model = YOLO("best.pt")
    # sanity print (2 classes)
    print("Class names:", model.names)
    return model

model = load_model()

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded is not None:
    # Show original image
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Convert PIL -> numpy
    img_np = np.array(img)

    # Run prediction (classification)
    results = model.predict(img_np, imgsz=224, verbose=False)
    r = results[0]

    # Get probabilities
    if hasattr(r, "probs") and r.probs is not None:
        probs = r.probs.data.cpu().numpy()  # shape: (num_classes,)
        class_names = r.names               # dict: {0: 'NonViolence', 1: 'Violence'}

        # assume {0: NonViolence, 1: Violence} like we saw in Colab
        non_violence_prob = float(probs[0])
        violence_prob = float(probs[1])

        # Decide label by threshold on violence probability
        threshold = 0.5
        if violence_prob >= threshold:
            final_label = "Violence"
        else:
            final_label = "NonViolence"

        # Show probabilities
        st.subheader("Prediction probabilities")
        st.write(f"**NonViolence:** {non_violence_prob:.3f}")
        st.write(f"**Violence:** {violence_prob:.3f}")
        st.write(f"**Decision (threshold = {threshold:.2f}):** **{final_label}**")

    else:
        st.error("Model did not return probabilities. Make sure this is a classification model (yolov8*-cls).")

    # Show annotated image (YOLO overlay)
    annotated = r.plot()  # numpy array with text overlay
    st.image(annotated, caption="Model Output", use_column_width=True)

