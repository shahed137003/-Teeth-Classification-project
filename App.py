import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Title and description for the app
st.title("Teeth Classification Model Deployment")
st.write("This app classifies teeth based on an image using a pre-trained machine learning model.")

# Load the saved model
@st.cache(allow_output_mutation=True)  # Cache the model to avoid reloading on every interaction
def load_teeth_classification_model():
    model = load_model('teeth_classification_model.h5')
    return model

model = load_teeth_classification_model()

# Upload image using Streamlit's file uploader
uploaded_file = st.file_uploader("Upload an image of a tooth", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    def preprocess_image(img):
        img = img.resize((224, 224))  # Resize to match model input shape
        img_array = image.img_to_array(img)  # Convert image to array
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize image
        return img_array

    img_preprocessed = preprocess_image(img)

    # Predict using the loaded model
    predictions = model.predict(img_preprocessed)
    predicted_class = np.argmax(predictions, axis=1)[0]  # Get the class with the highest probability
   

    # Display the result
    st.write(f"Predicted Class: {class_name[predicted_class]}")

    # # Show the prediction probabilities
    # st.write("Prediction Probabilities:")
    # for i, prob in enumerate(predictions[0]):
    #     st.write(f"{class_name[i]}: {prob:.2f}")
