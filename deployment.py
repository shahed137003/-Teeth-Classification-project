import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load your model
model = tf.keras.models.load_model('teeth_classification.h5')

# Streamlit app layout
st.title('Teeth Classification App')

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Preprocess image
    image = Image.open(uploaded_file)
    image = image.resize((224, 224))  # Resize to the input size expected by the model
    image_array = np.array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Predict
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions, axis=1)[0]

    # Display results
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write(f'Prediction: {predicted_class}')