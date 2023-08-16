import os
import cv2
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

# Paths to data folders
data_folder = 'data'
raw_data_folder = os.path.join(data_folder, 'raw-data')
processed_folder = os.path.join(data_folder, 'processed-data')

# Load the pre-trained model from TensorFlow Hub
model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
movenet = model.signatures['serving_default']

# Iterate through activity subfolders in the raw_data folder
activity_subfolders = [f for f in os.listdir(raw_data_folder) if os.path.isdir(os.path.join(raw_data_folder, f))]
for activity_subfolder in activity_subfolders:
    activity_folder = os.path.join(raw_data_folder, activity_subfolder)
    
    # Create processed activity folder in processed_data subfolder
    processed_activity_folder = os.path.join(processed_folder, activity_subfolder)
    os.makedirs(processed_activity_folder, exist_ok=True)
    
    # Iterate through images in the activity folder
    for image_filename in os.listdir(activity_folder):
        if image_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(activity_folder, image_filename)
            original_image = cv2.imread(image_path)
            
            # Preprocess the image
            resized_image = cv2.resize(original_image, (192, 192))
            rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            scaled_image = (rgb_image / 255.0) * 255
            int32_image = scaled_image.astype(np.int32)
            input_image = tf.expand_dims(int32_image, axis=0)
            
            # Run model inference
            outputs = movenet(input_image)
            keypoints = outputs['output_0']
            
            # Convert keypoints to numpy array
            keypoints_array = keypoints.numpy()
            keypoints_coordinates = keypoints_array[0, 0, :, :2]
            
            # Scale keypoints to match the original image dimensions
            image_height, image_width, _ = original_image.shape
            adjusted_keypoints = keypoints_coordinates.copy()
            adjusted_keypoints[:, 0] *= image_height  # Multiply by height for y-coordinate
            adjusted_keypoints[:, 1] *= image_width   # Multiply by width for x-coordinate
            
            # Draw keypoints on the original image using OpenCV
            annotated_image = original_image.copy()
            for coord in adjusted_keypoints:
                y, x = coord
                cv2.circle(annotated_image, (int(x), int(y)), 5, (0, 0, 255), -1)
            
            # Save the annotated image in the processed activity folder
            processed_image_path = os.path.join(processed_activity_folder, image_filename)
            cv2.imwrite(processed_image_path, annotated_image)
