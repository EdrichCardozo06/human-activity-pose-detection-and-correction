
import os
from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('my_model_final.h5')

# Define the label mapping dictionary
label_mapping = {
    # Define your label mapping here
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
    predicted_pose = label_mapping.get(predicted_label_numeric, 'Unknown Pose, please input a valid image')

    return predicted_pose

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the POST request has a file part
        if 'file' not in request.files:
            return 'No file part'

        file = request.files['file']

        # If the user does not select a file, the browser submits an empty part without a filename
        if file.filename == '':
            return 'No selected file'

        if file:
            # Save the uploaded file to a temporary directory
            filename = os.path.join('temp', file.filename)
            file.save(filename)

            # Predict the pose for the user-provided image
            predicted_pose = predict_pose(filename)

            # Display the user-provided image and the predicted pose name
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(plt.imread(filename))
            ax.set_title(f"Predicted Pose: {predicted_pose}")
            plt.show()

            # Return the prediction result to the HTML template
            return f'Predicted Pose: {predicted_pose}'

    return render_template('edrichindex.html')

if __name__ == '__main__':
    os.makedirs('temp', exist_ok=True)  # Create a temporary directory
    app.run(debug=True)
