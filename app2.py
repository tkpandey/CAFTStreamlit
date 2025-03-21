import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import gdown
import os

st.title("another app")

MODEL_PATH = "model.h5"
MODEL_URL = "https://drive.google.com/uc?id=1Xj-kLaONKILbsqtZ_IqEY25HgY9dzx0c"


# Load the model only once
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model, please wait...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()
st.success("Model loaded successfully!")
class_names = ['Early Blight of Potato', 'Late Blight of Potato','Healthy']

# Initialize session state for page navigation
if "page" not in st.session_state:
    st.session_state.page = "home"

# Navigation functions
def go_to_home():
    st.session_state.page = "home"

def go_to_upload():
    st.session_state.page = "upload"

def go_to_prediction():
    st.session_state.page = "prediction"

# Sidebar Navigation Buttons
st.sidebar.title("Navigation")
st.sidebar.button("Home", on_click=go_to_home)
st.sidebar.button("Upload Image", on_click=go_to_upload)
st.sidebar.button("Prediction", on_click=go_to_prediction)

# Home Page
if st.session_state.page == "home":
    st.title("🐾 Image Classification App")
    st.write("Welcome to the Image Classification App!")
    st.write("Use the sidebar to navigate to different sections.")
    st.image("https://picsum.photos/400/300", caption="Random Image", use_column_width=True)

# Upload Image Page
elif st.session_state.page == "upload":
    st.title("📷 Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        st.session_state["uploaded_image"] = uploaded_file
        st.success("Image uploaded successfully! Click 'Prediction' to see the result.")

# Prediction Page
elif st.session_state.page == "prediction":
    st.title("🔍 Prediction")
    if "uploaded_image" in st.session_state:
        uploaded_file = st.session_state["uploaded_image"]

        # Preprocess the uploaded image
        image = Image.open(uploaded_file).resize((224, 224))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        prediction = model.predict(image)
        predicted_class = np.argmax(prediction, axis=1)

        st.write(f"**Predicted Class:** {class_names[predicted_class[0]]}")
        st.image(uploaded_file, caption="Predicted Image", use_column_width=True)
    else:
        st.warning("No image uploaded! Please upload an image from the 'Upload Image' page.")
