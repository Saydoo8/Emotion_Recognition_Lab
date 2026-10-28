import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('emotion_model.h5')

model = load_model()

class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

st.title("Facial Emotion Recognition App")

choice = st.radio("Select Input Mode", ("Upload Image", "Use Webcam"))

image = None
if choice == "Upload Image":
    uploaded_file = st.file_uploader("Choose a face image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
else:
    camera_image = st.camera_input("Take a picture")
    if camera_image is not None:
        image = Image.open(camera_image)

if image is not None:
    st.image(image, caption="Input Image", use_container_width=True)
    
    image = ImageOps.grayscale(image)
    image = image.resize((48, 48))
    img_array = np.array(image) / 255.0
    img_array = np.reshape(img_array, (1, 48, 48, 1))
    
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = 100 * np.max(predictions)
    
    st.success(f"Prediction: **{predicted_class}** ({confidence:.2f}% confidence)")
    
    st.subheader("Class Probabilities Breakdown")
    for i, class_name in enumerate(class_names):
        prob = float(predictions[0][i])
        st.progress(prob, text=f"{class_name}: {prob*100:.2f}%")