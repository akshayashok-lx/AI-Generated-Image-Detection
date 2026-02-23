import os
import gdown

MODEL_PATH = "model/ai_image_detector_large_gpu.h5"
DRIVE_URL = "https://drive.google.com/uc?id=1oTe5iz_ihyjteZJPtICiiarhS9AjQl3M"

os.makedirs("model", exist_ok=True)

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)



import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="AI Image Detector",
    page_icon="🤖",
    layout="centered"
)

# ----------------------------
# Custom Styling
# ----------------------------
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }.201
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_my_model():
    return load_model("ai_image_detector_large_gpu.h5")

model = load_my_model()

IMG_SIZE = 224
THRESHOLD = 0.5

# ----------------------------
# Title Section
# ----------------------------
st.title("🤖 AI Generated Image Detector")
st.write("Upload an image to check whether it is REAL or AI-Generated.")

st.divider()

# ----------------------------
# Prediction Function
# ----------------------------
def predict_image(image):
    image = image.convert("RGB")
    img = np.array(image)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0][0]
    return prediction

# ----------------------------
# Upload Section
# ----------------------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing image..."):
        prediction = predict_image(image)

    st.divider()

    fake_prob = float(prediction)
    real_prob = 1 - fake_prob

    # Display Probability Bars
    st.subheader("Prediction Confidence")

    st.write("Real Probability")
    st.progress(real_prob)

    st.write("Fake Probability")
    st.progress(fake_prob)

    st.divider()

    # Final Result
    if fake_prob > THRESHOLD:
        st.error(f"🚨 AI-GENERATED IMAGE DETECTED")
        st.write(f"Confidence: {fake_prob*100:.2f}%")
    else:
        st.success(f"✅ REAL IMAGE DETECTED")
        st.write(f"Confidence: {real_prob*100:.2f}%")
