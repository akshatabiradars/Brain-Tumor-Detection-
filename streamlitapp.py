import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions

# Load the pre-trained VGG16 model
model = VGG16(weights='imagenet')

# Function to preprocess the image
def preprocess_image(img):
    img = img.resize((224, 224))  # Resize the image to match VGG16 input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Function to predict the class of the image
def predict_image(img):
    img_array = preprocess_image(img)
    preds = model.predict(img_array)
    return decode_predictions(preds, top=3)[0]  # Return top 3 predictions

# Streamlit UI
st.title("Brain Tumor Detection using CNN,VGG16")
st.write("Upload an image of a brain scan to detect if there's a tumor.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Make prediction
    preds = predict_image(img)
    
    # Display the predictions
    st.write("Predictions:")
    for pred in preds:
        st.write(f"{pred[1]}: {pred[2]*100:.2f}%")
