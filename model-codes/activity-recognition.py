import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the saved model from .h5 file
model = tf.keras.models.load_model('/content/my_model_final.h5')

# Define the dataset directory
dataset_dir = '/content/data/data-raw'

# Function to collect image file paths with various extensions
def collect_image_paths(directory):
    image_paths = []
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        image_paths.extend(list(directory.glob(f'**/{ext}')))
    return image_paths

# Collect image file paths and labels
image_paths = collect_image_paths(Path(dataset_dir))
labels = [os.path.split(os.path.split(path)[0])[1] for path in image_paths]

# Create a DataFrame to store image paths and labels
image_df = pd.DataFrame({'Filepath': image_paths, 'Label': labels})

# Function to preprocess and display true and predicted labels for user input
def display_labels_for_input(input_image_path, model):
    # Normalize the provided input_image_path for matching with dataset
    input_image_path = os.path.normpath(input_image_path)

    # Find the row in the DataFrame corresponding to the input image
    row = image_df[image_df['Filepath'].apply(lambda x: os.path.normpath(x)) == input_image_path]

    if not row.empty:
        true_label = row['Label'].values[0]

        # Load and preprocess the user input image for prediction
        img = Image.open(input_image_path)

        # Convert grayscale image to RGB if it has only one channel
        if img.mode != 'RGB':
            img = img.convert('RGB')

        img = img.resize((224, 224))  # Resize the image to the desired dimensions
        input_image = image.img_to_array(img)
        input_image = np.expand_dims(input_image, axis=0)
        input_image = tf.keras.applications.vgg16.preprocess_input(input_image)

        # Make predictions using your loaded model
        pred = model.predict(input_image)
        pred_label_index = np.argmax(pred)

        # Map the predicted label index back to class name
        labels = (train_images.class_indices)
        labels = dict((v, k) for k, v in labels.items())
        predicted_label = labels[pred_label_index]

        # Display the input image and labels
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img)
        ax.set_title(f"True Label: {true_label}\nPredicted Label: {predicted_label}", color="green" if true_label == predicted_label else "red")
        plt.show()
    else:
        print("Image not found in the dataset.")

# Get user input for the image file path
user_input_path = input("Enter the path of the image file: ")

# Call the function to display labels for user input
display_labels_for_input(user_input_path, model)


