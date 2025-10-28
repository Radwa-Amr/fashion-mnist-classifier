import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from src.utils import class_names

# Load model
model = tf.keras.models.load_model("saved_model/fashion_mnist_model.h5")

st.set_page_config(page_title="Fashion MNIST Classifier", page_icon="🧥", layout="centered")
st.title("🧥 Fashion MNIST Image Classifier")

uploaded_file = st.file_uploader("Upload a grayscale clothing image (28x28)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Convert image to grayscale and resize
    image = Image.open(uploaded_file).convert("L").resize((28, 28))

    # Normalize & reshape same as training data
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, 28, 28)
    
    # لو النموذج متوقع input فيه channel (28,28,1) ضيف البعد الأخير:
    if len(model.input_shape) == 4:
        img_array = np.expand_dims(img_array, -1)

    st.image(image, caption="Uploaded Image", width=150)

    # Predict
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    confidence = np.max(prediction)

    st.subheader(f"Prediction: {class_names[class_idx]}")
    st.caption(f"Confidence: {confidence:.2%}")
