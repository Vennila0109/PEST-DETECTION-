import numpy as np
import streamlit as st
import cv2
from keras.models import load_model

# Loading the Model
model = load_model('model.H5')

# Name of Classes
CLASS_NAMES = [
    "incertulas",
    "yelow"
]

# Setting Title of App
st.title("Pest Disease Classifier")
st.markdown("Upload an image ")

# Uploading the image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# On predict button click
if uploaded_image is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    # Display the image
    st.image(opencv_image, channels="BGR", caption='Uploaded Image')

    # Preprocess the image
    resized_image = cv2.resize(opencv_image, (200, 200))  # Resize to the appropriate input size of the model
    resized_image = resized_image / 255.0  # Normalize the image

    # Make prediction
    prediction = model.predict(np.expand_dims(resized_image, axis=0))[0]

    # Display the result
    predicted_class_index = np.argmax(prediction)
    print(predicted_class_index)
    predicted_class = CLASS_NAMES[predicted_class_index]
    confidence = prediction[predicted_class_index]

    st.write(f"Predicted Disease: {predicted_class} (Confidence: {confidence:.2f})")
