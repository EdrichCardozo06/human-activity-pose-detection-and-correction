
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Load the trained model from the .h5 file
model = tf.keras.models.load_model('my_model_final.h5')

# Define the label mapping dictionary
label_mapping = {
    0: 'Adho Mukha Svanasana',
    1: 'Adho Mukha Vrksasana',
    2: 'Alanasana',
    3: 'Anjaneyasana',
    4: 'Ardha Chandrasana',
    5: 'Ardha Matsyendrasana',
    6: 'Ardha Navasana',
    7: 'Ardha Pincha Mayurasana',
    8: 'Ashta Chandrasana',
    9: 'Baddha Konasana',
    10: 'Bakasana',
    11: 'Balasana',
    12: 'Bitilasana',
    13: 'Camatkarasana',
    14: 'Dhanurasana',
    15: 'Eka Pada Rajakapotasana',
    16: 'Garudasana',
    17: 'Halasana',
    18: 'Hanumanasana',
    19: 'Malasana',
    20: 'Marjaryasana',
}

# Function to preprocess the input image
def preprocess_image(image_path):
    image = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(224, 224)
    )
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = tf.keras.applications.mobilenet_v3.preprocess_input(image_array)
    return image_array

# Function to predict the pose from a given image path
def predict_pose(image_path):
    # Preprocess the image
    image_array = preprocess_image(image_path)

    # Make predictions
    predictions = model.predict(image_array)

    # Decode the predictions using the label mapping
    predicted_label_numeric = np.argmax(predictions, axis=1)[0]
    predicted_pose = label_mapping.get(predicted_label_numeric, 'Unknown Pose,please input a valid image')

    return predicted_pose

# Prompt the user for an image path
user_input_image_path = input("Enter the path to your image: ")

# Predict the pose for the user-provided image
predicted_pose = predict_pose(user_input_image_path)

# Display the user-provided image and the predicted pose name
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(plt.imread(user_input_image_path))
ax.set_title(f"Predicted Pose: {predicted_pose}")
plt.show()
