import streamlit as st
import numpy as np    
import tensorflow as tf
from PIL import Image
import cv2
import urllib.request
import os


REPO_URL = 'https://raw.githubusercontent.com/anu3592/BrainTumorDetection/main/app.py'

def main():
    selected_box = st.sidebar.selectbox(
        'Choose an option..',
        ('Brain Tumor Detection', 'View Source Code')
    )
        
    if selected_box == 'Brain Tumor Detection':        
        st.sidebar.success('Upload an MRI image to detect the presence of a brain tumor.')
        application()
    if selected_box == 'View Source Code':
        st.code(get_file_content_as_string(REPO_URL))

def get_file_content_as_string(url):
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")

def load_model():
    model = tf.keras.models.load_model('mymodel.h5')
    return model

def application():
    models_load_state = st.text('Loading models...')
    model = load_model()
    models_load_state.text('Models loaded successfully')
    
    file_to_be_uploaded = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])
    
    if file_to_be_uploaded:
        image = Image.open(file_to_be_uploaded)
        st.image(image, caption='Uploaded MRI image', use_column_width=True)
        st.success('Prediction: ' + predict(model, file_to_be_uploaded))

def preprocess_image(image_path):
    
    img = cv2.imread(image_path)

    img = cv2.resize(img, (150, 150))
    img_array = np.array(img)
    img_array = img_array.reshape(1, 150, 150, 3)
    
    return img_array

def predict(model, image_file):
    temp_file_path = "temp_image.jpg"
    with open(temp_file_path, "wb") as f:
        f.write(image_file.getbuffer())
        
    preprocessed_image = preprocess_image(temp_file_path)

    predictions = model.predict(preprocessed_image)

    os.remove(temp_file_path)
    predicted_label = np.argmax(predictions[0]) + 1
    
    if predicted_label in [1, 2, 4]:
        result = "Brain Tumor Detected"
    else:
        result = "No Brain Tumor Detected"
    
    return result

if __name__ == "__main__":
    main()
