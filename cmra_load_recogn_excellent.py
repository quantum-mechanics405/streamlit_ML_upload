import tensorflow as tf
import streamlit as st
from PIL import Image
import numpy as np

# Load the ResNet50 model pre-trained on ImageNet
model = tf.keras.applications.ResNet50(weights="imagenet")

# Function to preprocess the image
def preprocess_image(image):
    # Convert the image to RGB format if not already
    image = image.convert('RGB')
    # Resize image to 224x224
    image = image.resize((224, 224))
    # Convert image to numpy array
    image_array = np.array(image)
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    # Preprocess image for ResNet50
    st.image(image)
    return tf.keras.applications.resnet50.preprocess_input(image_array)

# Upload images
# imag1 = st.file_uploader('Upload image 1', type=['jpg', 'jpeg', 'png'])
imag1 = st.camera_input('capture image ')
imag2 = st.file_uploader('Upload image 2', type=['jpg', 'jpeg', 'png'])

# Process each image if uploaded
if imag1 is not None:
    image1 = Image.open(imag1)
    processed_image1 = preprocess_image(image1)
else:
    processed_image1 = None

if imag2 is not None:
    image2 = Image.open(imag2)
    processed_image2 = preprocess_image(image2)
else:
    processed_image2 = None

# Make predictions if images are available
if processed_image1 is not None or processed_image2 is not None:
    # Prepare inputs for the model
    images_resized = []
    if processed_image1 is not None:
        images_resized.append(processed_image1)
        
    if processed_image2 is not None:
        images_resized.append(processed_image2)
        
    
    images_resized = np.vstack(images_resized)  # Stack images into a single batch
    # Predict
    Y_proba = model.predict(images_resized)
    # Decode predictions
    top_K = tf.keras.applications.resnet50.decode_predictions(Y_proba, top=3)

    # Display predictions
    for image_index, top_k in enumerate(top_K):
        st.write(f"Image #{image_index + 1}:")
        for class_id, name, y_proba in top_k:
            st.write(f"  {class_id} - {name:12s} {y_proba:.2%}")

    
else:
    st.write("Please upload at least one image.")
