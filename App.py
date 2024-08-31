import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
class_name=["CaS","CoS","Gum","MC","OC","OLP","OT"]
#Load your model
# Inject custom HTML to set the favicon
st.markdown(
    """
    <link rel="icon" href="https://img.freepik.com/premium-photo/funny-cartoon-character-white-tooth-colorful-background-pediatric-dentistry-stomatology_115128-6000.jpg" type="image/x-icon">
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
    /* Change background color */
    .stApp {
        background-color: pink;  /* Light blue background */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app layout
st.markdown("# 🦷 Teeth Classification ")


#Load your model
model = tf.keras.models.load_model('saved_model.h5')


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
    image = image.resize((128, 128)) 
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    if st.button('Predict 🦷'):
        st.write(f"Predicted Class: {class_name[predicted_class]}")
        st.success("the prediction is done successfully")
        st.select_slider("Rate my website plz :smile:",["Bad","Ok","Good","Great","Outstanding"])
