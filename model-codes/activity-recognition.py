

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Load the trained model from the .h5 file
model = tf.keras.models.load_model('my_model.h5')

# Load the label mapping from the .pkl file
label_mapping_series = joblib.load('labels.pkl')

# Convert the Pandas Series to a dictionary
label_mapping = {label: number for number, label in enumerate(label_mapping_series.unique())}

# Function to predict a pose from a given image path
def predict_pose(image_path):
    new_image = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(224, 224)
    )
    new_image_array = tf.keras.preprocessing.image.img_to_array(new_image)
    new_image_array = np.expand_dims(new_image_array, axis=0)
    new_image_array = tf.keras.applications.mobilenet_v2.preprocess_input(new_image_array)

    # Make predictions
    predictions = model.predict(new_image_array)

    # Decode the predictions
    predicted_label_numeric = np.argmax(predictions, axis=1)[0]

    # Map the numeric prediction to the corresponding pose name
    predicted_pose = list(label_mapping.keys())[list(label_mapping.values()).index(predicted_label_numeric)]

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
